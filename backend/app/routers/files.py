import logging
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from ..chunking import chunk_text
from ..config import Settings
from ..deps import CurrentUserDep, DbDep, SettingsDep, create_user_client
from ..embeddings import embed_documents
from ..extract import extract_text
from ..schemas import DocumentOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

BUCKET = "documents"

# Rows per PostgREST insert. A 768-dim vector is ~15 KB as JSON, so this keeps
# each request around 1.5 MB instead of one giant payload for a big document.
_INSERT_BATCH = 100

_DOCUMENT_COLUMNS = "id,filename,mime_type,byte_size,status,error,chunk_count,created_at"


def _safe_storage_name(filename: str) -> str:
    # Storage object keys reject some characters that are fine in filenames.
    # The original name stays untouched in documents.filename.
    return re.sub(r"[^\w.\-]+", "_", filename) or "upload"


def _ingest(
    document_id: str,
    data: bytes,
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
    """
    db = create_user_client(settings, token)

    def update(**fields) -> None:
        db.table("documents").update(fields).eq("id", document_id).execute()

    try:
        update(status="extracting")
        text = extract_text(data, filename, mime)

        update(status="chunking")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No text could be extracted from the document")

        update(status="embedding")
        vectors = embed_documents([c.content for c in chunks])

        rows = [
            {
                "document_id": document_id,
                "user_id": user_id,
                "ordinal": chunk.ordinal,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        for i in range(0, len(rows), _INSERT_BATCH):
            db.table("chunks").insert(rows[i : i + _INSERT_BATCH]).execute()

        update(status="ready", chunk_count=len(chunks))
        log.info("ingested %s: %d chunks", filename, len(chunks))

    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed for document %s (%s)", document_id, filename)
        # A row stuck at 'extracting' with no error is indistinguishable from a
        # job still running; 'failed' plus a readable reason is the contract.
        try:
            update(status="failed", error=str(exc)[:500])
        except Exception:
            log.exception("could not record ingest failure for %s", document_id)


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> DocumentOut:
    """Accept a file, store the bytes, and return before any processing.

    The pipeline (extract -> chunk -> embed) runs as a background job; the
    client watches documents.status over Realtime instead of holding this
    request open through a multi-minute ingest.
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

    # Generated here rather than by the database because the storage path
    # includes it, and the bytes land in Storage before the row exists.
    document_id = str(uuid.uuid4())
    filename = file.filename or "upload"
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
            }
        )
        .execute()
    )

    background.add_task(
        _ingest, document_id, data, filename, file.content_type,
        user.id, user.token, settings,
    )
    return DocumentOut(**{k: res.data[0][k] for k in _DOCUMENT_COLUMNS.split(",")})


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUserDep, db: DbDep) -> list[DocumentOut]:
    res = (
        db.table("documents")
        .select(_DOCUMENT_COLUMNS)
        .order("created_at", desc=True)
        .execute()
    )
    return [DocumentOut(**row) for row in res.data]


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
