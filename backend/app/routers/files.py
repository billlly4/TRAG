import base64
import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..llm import get_client
from ..schemas import DocumentOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


def _inline_source(data: bytes, mime: str | None) -> dict | None:
    """An inline document source for token counting.

    count_tokens rejects `file` sources outright ("File sources are not
    supported in the token counting endpoint"), so the size check has to be
    done against the bytes we still hold, before the upload.
    """
    mime = (mime or "").lower()
    if mime == "application/pdf":
        return {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(data).decode(),
        }
    if mime.startswith("text/") or mime in {"application/json", "application/xml"}:
        return {
            "type": "text",
            "media_type": "text/plain",
            "data": data.decode("utf-8", errors="replace"),
        }
    return None


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> DocumentOut:
    """Upload a document to Anthropic's Files API.

    Note what does NOT happen here: no chunking, no embedding, no vector index.
    The whole file is handed to Anthropic and later dropped into context in one
    piece. That is the point of Module 1 -- it works, and then it stops working
    once the document is large enough, which is what Module 2 exists to fix.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")

    client = get_client()

    # Measure first, upload second. count_tokens is free and exact, so an
    # oversized document is rejected before it ever reaches Anthropic -- no
    # upload-then-delete dance, and no 400 surfacing mid-conversation days later.
    token_estimate: int | None = None
    source = _inline_source(data, file.content_type)
    if source is not None:
        try:
            counted = client.messages.count_tokens(
                model=settings.anthropic_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": source},
                            {"type": "text", "text": "."},
                        ],
                    }
                ],
            )
            token_estimate = counted.input_tokens
        except Exception:
            # Log it rather than swallowing it -- a silent None here is
            # indistinguishable from "this format cannot be counted".
            log.warning("count_tokens failed for %s", file.filename, exc_info=True)

    if token_estimate is not None and token_estimate > settings.max_document_tokens:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Document is {token_estimate:,} tokens, over the "
            f"{settings.max_document_tokens:,} limit. Module 1 puts whole "
            f"documents in context; chunking arrives in Module 2.",
        )

    uploaded = client.beta.files.upload(
        file=(file.filename, io.BytesIO(data), file.content_type or "application/octet-stream")
    )

    res = (
        db.table("documents")
        .insert(
            {
                "user_id": user.id,
                "filename": file.filename,
                "mime_type": file.content_type,
                "byte_size": len(data),
                "anthropic_file_id": uploaded.id,
                "token_estimate": token_estimate,
            }
        )
        .execute()
    )
    return DocumentOut(**res.data[0])


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUserDep, db: DbDep) -> list[DocumentOut]:
    res = (
        db.table("documents")
        .select(
            "id,filename,mime_type,byte_size,token_estimate,created_at,anthropic_file_id"
        )
        .order("created_at", desc=True)
        .execute()
    )
    return [DocumentOut(**row) for row in res.data]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, user: CurrentUserDep, db: DbDep) -> None:
    res = (
        db.table("documents")
        .select("anthropic_file_id")
        .eq("id", document_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    file_id = res.data[0]["anthropic_file_id"]
    db.table("documents").delete().eq("id", document_id).execute()

    # Best-effort remote cleanup. The local row is the source of truth for what
    # the user can attach, so a failure here must not fail the request.
    try:
        get_client().beta.files.delete(file_id)
    except Exception:
        pass
