import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ..config import Settings
from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..extract import SUPPORTED_SUFFIXES
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


def _to_out(row: dict, current_hash: str) -> DocumentOut:
    return DocumentOut(
        **{k: row[k] for k in _DOCUMENT_COLUMNS.split(",")},
        stale=row.get("config_hash") != current_hash,
    )


def _enqueue(db, settings: Settings, document_id: str, user_id: str,
             filename: str, mime: str | None, storage_path: str) -> None:
    """Queue a document for the ingestion worker.

    Replaces the BackgroundTask this used to be. The difference that matters is
    durability: a background task lives inside this process, so a restart or a
    crash abandoned it silently and left the document at 'pending' with nothing
    working on it. A row in ingest_jobs outlives the API entirely.

    The signed URL is minted HERE, while we still hold the caller's JWT. That is
    what lets the worker read one specific object with no Storage credentials of
    its own: it is handed a capability for a single file, not a key.
    """
    signed = db.storage.from_(BUCKET).create_signed_url(
        storage_path, settings.ingest_url_ttl_seconds
    )
    url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url")
    if not url:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Could not create a signed URL for the uploaded file",
        )

    expires = datetime.now(timezone.utc) + timedelta(
        seconds=settings.ingest_url_ttl_seconds
    )
    db.table("ingest_jobs").insert({
        "document_id": document_id,
        "user_id": user_id,
        "filename": filename,
        "mime_type": mime,
        "source_url": url,
        "source_expires_at": expires.isoformat(),
    }).execute()
    log.info("queued %s for ingestion (document %s)", filename, document_id)


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

    # Reject unsupported formats here rather than letting the worker discover it:
    # otherwise the bytes are uploaded to Storage and a document row is created
    # just to end up `failed`, which reads like a processing bug rather than a
    # file the app never accepted.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{suffix or 'This file type'} is not supported. Supported: "
            f"{', '.join(SUPPORTED_SUFFIXES)}",
        )
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is {len(data):,} bytes, over the "
            f"{settings.max_upload_bytes:,} byte limit",
        )

    filename = file.filename or "upload"
    chash = content_hash(data)
    current = config_hash()

    # Storage quota. Enforced for real by a trigger (0006_quotas.sql); checked
    # here so the message names the numbers. Counted AFTER the duplicate and
    # same-filename lookups below would have run, though -- so it is computed
    # excluding any document this upload will replace, since replacing a file
    # is not the same as adding one.
    def _quota_check(replacing_id: str | None) -> None:
        rows = db.table("documents").select("id,byte_size").execute().data or []
        used = sum(
            r["byte_size"] or 0 for r in rows if r["id"] != replacing_id
        )
        if used + len(data) > settings.max_storage_bytes:
            mb = 1024 * 1024
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Storage full: {used / mb:.1f} MB used of "
                f"{settings.max_storage_bytes / mb:.0f} MB, and this file is "
                f"{len(data) / mb:.1f} MB. Delete a document to free space.",
            )

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
        _quota_check(row["id"])
        _restart_row(db, row, chash, file.content_type, len(data))
        _enqueue(db, settings, row["id"], user.id, row["filename"],
                 file.content_type, row["storage_path"])
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
        # Replacing, not adding: the row's current size does not count against
        # the quota, only the new bytes do.
        _quota_check(row["id"])
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
        _enqueue(db, settings, row["id"], user.id, filename,
                 file.content_type, storage_path)
        response.status_code = status.HTTP_200_OK
        return _to_out(row, current)

    # Genuinely new document.
    _quota_check(None)
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

    _enqueue(db, settings, document_id, user.id, filename,
             file.content_type, storage_path)
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
        _enqueue(db, settings, row["id"], user.id, row["filename"],
                 row["mime_type"], row["storage_path"])
    return {"count": len(stale)}


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_document(
    document_id: str,
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
    _enqueue(db, settings, document_id, user.id, row["filename"],
             row["mime_type"], row["storage_path"])
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
