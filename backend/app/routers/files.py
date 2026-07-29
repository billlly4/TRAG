import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ..chunking import chunk_text, embed_text
from ..config import Settings
from ..deps import CurrentUserDep, DbDep, SettingsDep, create_user_client
from ..embeddings import embed_documents
from ..extract import extract_text
from ..metadata import DocumentMetadata, extract_metadata
from ..record import config_hash, content_hash, extraction_config
from ..schemas import DocumentOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

BUCKET = "documents"

# Rows per PostgREST insert. A 768-dim vector is ~15 KB as JSON, so this keeps
# each request around 1.5 MB instead of one giant payload for a big document.
_INSERT_BATCH = 100

_DOCUMENT_COLUMNS = (
    "id,filename,mime_type,byte_size,status,error,chunk_count,created_at,"
    "title,doc_type,source_org,published_year,topics,summary"
)

# What the routes fetch: the response columns plus what the record manager
# needs to decide staleness and re-download bytes.
_SELECT_COLUMNS = _DOCUMENT_COLUMNS + ",config_hash,storage_path"


def _safe_storage_name(filename: str) -> str:
    # Storage object keys reject some characters that are fine in filenames.
    # The original name stays untouched in documents.filename.
    return re.sub(r"[^\w.\-]+", "_", filename) or "upload"


def _metadata_columns(meta: DocumentMetadata | None, error: str | None) -> dict:
    """Flatten extracted metadata into the documents columns.

    Written even when extraction failed, so a retry cannot leave last run's
    metadata attached to this run's chunks. The failure reason goes into the
    jsonb column rather than documents.error -- `error` means "this document
    is not searchable", and a document with no metadata still is.
    """
    if meta is None:
        return {
            "title": None, "doc_type": None, "source_org": None, "authors": None,
            "published_on": None, "published_year": None, "language": None,
            "topics": None, "summary": None,
            "metadata": {"metadata_error": error} if error else None,
        }

    # mode="json" turns the `date` into a string; PostgREST is fed JSON and
    # a bare datetime.date is not serialisable.
    dump = meta.model_dump(mode="json")
    return {
        "title": meta.title,
        "doc_type": meta.doc_type,
        "source_org": meta.source_org,
        "authors": meta.authors,
        "published_on": dump["published_on"],
        "published_year": meta.published_year,
        "language": meta.language,
        "topics": meta.topics,
        "summary": meta.summary,
        "metadata": dump,
    }


def _to_out(row: dict, current_hash: str) -> DocumentOut:
    return DocumentOut(
        **{k: row[k] for k in _DOCUMENT_COLUMNS.split(",")},
        stale=row.get("config_hash") != current_hash,
    )


def _ingest(
    document_id: str,
    data: bytes | None,
    storage_path: str | None,
    filename: str,
    mime: str | None,
    user_id: str,
    token: str,
    settings: Settings,
) -> None:
    """Walk a document through extracting -> chunking -> embedding -> ready.

    Each status write is what Supabase Realtime pushes to the UI, so the
    transitions are the progress bar. Runs with the caller's JWT: writes go
    through RLS as the user, same as the request that scheduled it. A
    service_role key would also work, which is exactly why it is not used.

    `data` is passed by the upload path (it already holds the bytes); the
    reprocess path passes `storage_path` instead and the bytes are read back
    here, inside the try, so a Storage failure lands in documents.error.
    """
    db = create_user_client(settings, token)

    # Captured before work starts: these describe the run that actually
    # happened, even if settings change while it is in flight.
    cfg = extraction_config()
    cfg_hash = config_hash()

    def update(**fields) -> None:
        db.table("documents").update(fields).eq("id", document_id).execute()

    try:
        if data is None:
            data = db.storage.from_(BUCKET).download(storage_path)

        update(status="extracting")
        text = extract_text(data, filename, mime)

        # Before chunking, not after: the extracted title is prepended to every
        # chunk's embedding input, so it has to exist before anything is
        # embedded. extract_metadata never raises -- a document with no
        # metadata is still perfectly searchable, so a failure here must not
        # cost the whole ingest.
        update(status="analyzing")
        meta, meta_error = extract_metadata(text, filename)

        update(status="chunking")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No text could be extracted from the document")

        update(status="embedding")
        # What gets embedded is enriched with title and section; what gets
        # stored in chunks.content stays the raw body. See chunking.embed_text.
        title = (meta.title if meta and meta.title else None) or filename
        vectors = embed_documents([embed_text(c, title) for c in chunks])

        rows = [
            {
                "document_id": document_id,
                "user_id": user_id,
                "ordinal": chunk.ordinal,
                "content": chunk.content,
                "section": chunk.section,
                "token_count": chunk.token_count,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        # Replace, not append: on re-ingest the old regime's chunks go here,
        # after the new embeddings exist, so the "document has no chunks"
        # window is milliseconds rather than the whole pipeline run.
        db.table("chunks").delete().eq("document_id", document_id).execute()
        for i in range(0, len(rows), _INSERT_BATCH):
            db.table("chunks").insert(rows[i : i + _INSERT_BATCH]).execute()

        update(
            status="ready",
            chunk_count=len(chunks),
            error=None,
            config_hash=cfg_hash,
            extraction_config=cfg,
            processed_at=datetime.now(timezone.utc).isoformat(),
            # Always written, even on a failed extraction: otherwise a retry
            # leaves the previous run's metadata attached to this run's chunks.
            **_metadata_columns(meta, meta_error),
        )
        log.info(
            "ingested %s: %d chunks, metadata=%s",
            filename, len(chunks), "ok" if meta else f"failed ({meta_error})",
        )

    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed for document %s (%s)", document_id, filename)
        # A row stuck at 'extracting' with no error is indistinguishable from a
        # job still running; 'failed' plus a readable reason is the contract.
        try:
            update(status="failed", error=str(exc)[:500])
        except Exception:
            log.exception("could not record ingest failure for %s", document_id)


def _restart_row(
    db, row: dict, chash: str, mime: str | None, byte_size: int
) -> None:
    """Reset an existing document row for re-ingestion with new bytes."""
    db.table("documents").update(
        {
            "status": "pending",
            "error": None,
            "chunk_count": None,
            "content_hash": chash,
            "mime_type": mime,
            "byte_size": byte_size,
        }
    ).eq("id", row["id"]).execute()
    row.update(
        status="pending", error=None, chunk_count=None,
        mime_type=mime, byte_size=byte_size,
    )


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    response: Response,
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> DocumentOut:
    """Accept a file; ingest it only if something actually changed.

    Record-manager order of checks:
      1. same bytes, current config, not failed  -> no-op, return existing (200)
      2. same bytes otherwise (failed / stale)   -> re-ingest that row (200)
      3. same filename, different bytes          -> update in place (200)
      4. otherwise                               -> new document (201)
    """
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is {len(data):,} bytes, over the "
            f"{settings.max_upload_bytes:,} byte limit",
        )

    filename = file.filename or "upload"
    chash = content_hash(data)
    current = config_hash()

    # RLS scopes both lookups to the caller -- no explicit user filter needed,
    # and no way for one user's upload to match another's document.
    dup = (
        db.table("documents")
        .select(_SELECT_COLUMNS)
        .eq("content_hash", chash)
        .limit(1)
        .execute()
    )
    if dup.data:
        row = dup.data[0]
        if row["status"] != "failed" and row.get("config_hash") == current:
            # Identical bytes, identical processing: nothing to do.
            response.status_code = status.HTTP_200_OK
            return _to_out(row, current)
        # Same bytes but the previous run failed or used an old config: retry
        # in place. Bytes are already in Storage from the first attempt.
        _restart_row(db, row, chash, file.content_type, len(data))
        background.add_task(
            _ingest, row["id"], data, None, row["filename"], file.content_type,
            user.id, user.token, settings,
        )
        response.status_code = status.HTTP_200_OK
        return _to_out(row, current)

    same_name = (
        db.table("documents")
        .select(_SELECT_COLUMNS)
        .eq("filename", filename)
        .limit(1)
        .execute()
    )
    if same_name.data:
        # Modified file: same name, new content. Update in place -- the
        # document id (and old chats' source references) stays stable.
        row = same_name.data[0]
        storage_path = row["storage_path"]
        if not storage_path:
            # Module 1 relic that never had bytes in Storage: adopt it.
            storage_path = f"{user.id}/{row['id']}/{_safe_storage_name(filename)}"
            db.table("documents").update({"storage_path": storage_path}).eq(
                "id", row["id"]
            ).execute()
        db.storage.from_(BUCKET).upload(
            storage_path,
            data,
            file_options={
                "content-type": file.content_type or "application/octet-stream",
                "upsert": "true",
            },
        )
        _restart_row(db, row, chash, file.content_type, len(data))
        background.add_task(
            _ingest, row["id"], data, None, filename, file.content_type,
            user.id, user.token, settings,
        )
        response.status_code = status.HTTP_200_OK
        return _to_out(row, current)

    # Genuinely new document.
    document_id = str(uuid.uuid4())
    storage_path = f"{user.id}/{document_id}/{_safe_storage_name(filename)}"

    db.storage.from_(BUCKET).upload(
        storage_path,
        data,
        file_options={"content-type": file.content_type or "application/octet-stream"},
    )

    res = (
        db.table("documents")
        .insert(
            {
                "id": document_id,
                "user_id": user.id,
                "filename": filename,
                "mime_type": file.content_type,
                "byte_size": len(data),
                "storage_path": storage_path,
                "status": "pending",
                "content_hash": chash,
            }
        )
        .execute()
    )

    background.add_task(
        _ingest, document_id, data, None, filename, file.content_type,
        user.id, user.token, settings,
    )
    return _to_out(res.data[0], current)


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUserDep, db: DbDep) -> list[DocumentOut]:
    current = config_hash()
    res = (
        db.table("documents")
        .select(_SELECT_COLUMNS)
        .order("created_at", desc=True)
        .execute()
    )
    return [_to_out(row, current) for row in res.data]


@router.post("/reprocess-stale", status_code=status.HTTP_202_ACCEPTED)
def reprocess_stale(
    background: BackgroundTasks,
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
) -> dict:
    """Re-ingest every document built under an old config.

    Sequential on purpose: Ollama serializes embedding work anyway, and one
    document at a time keeps the status badges legible.
    """
    current = config_hash()
    rows = db.table("documents").select(_SELECT_COLUMNS).execute().data
    stale = [
        r for r in rows
        if r.get("config_hash") != current and r.get("storage_path")
    ]
    for row in stale:
        db.table("documents").update({"status": "pending", "error": None}).eq(
            "id", row["id"]
        ).execute()
        background.add_task(
            _ingest, row["id"], None, row["storage_path"], row["filename"],
            row["mime_type"], user.id, user.token, settings,
        )
    return {"count": len(stale)}


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_document(
    document_id: str,
    background: BackgroundTasks,
    response: Response,
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
) -> DocumentOut:
    res = (
        db.table("documents")
        .select(_SELECT_COLUMNS)
        .eq("id", document_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    row = res.data[0]

    if not row["storage_path"]:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No stored bytes for this document (it was uploaded before Supabase "
            "Storage existed). Delete it and upload the file again.",
        )

    current = config_hash()
    if row["status"] == "ready" and row.get("config_hash") == current:
        # Already processed under the current config: no-op.
        response.status_code = status.HTTP_200_OK
        return _to_out(row, current)

    db.table("documents").update({"status": "pending", "error": None}).eq(
        "id", document_id
    ).execute()
    background.add_task(
        _ingest, document_id, None, row["storage_path"], row["filename"],
        row["mime_type"], user.id, user.token, settings,
    )
    row["status"] = "pending"
    return _to_out(row, current)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, user: CurrentUserDep, db: DbDep) -> None:
    res = (
        db.table("documents")
        .select("storage_path")
        .eq("id", document_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    storage_path = res.data[0]["storage_path"]

    # Chunks go by ON DELETE CASCADE -- no application code deletes derived
    # data. The original bytes are not derived data, so Storage still needs an
    # explicit (best-effort) removal; the row is the source of truth.
    db.table("documents").delete().eq("id", document_id).execute()
    if storage_path:
        try:
            db.storage.from_(BUCKET).remove([storage_path])
        except Exception:
            log.warning("orphaned storage object %s", storage_path, exc_info=True)
