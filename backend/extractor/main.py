"""Extraction service: bytes in, Markdown out.

Deliberately holds nothing. No database connection, no Storage credentials, no
user identity -- so there is nothing here to authorize and nothing to leak. That
is what makes moving it out of the API a small change rather than a security
redesign.

    uv run uvicorn backend.extractor.main:app --host 127.0.0.1 --port 8001

BIND TO LOOPBACK. The endpoint is unauthenticated by design; exposed publicly it
is an open CPU sink, since one 25 MB PDF can occupy a core for a minute.
"""

import logging

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, status

from .pipeline import convert, extraction_fingerprint

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

app = FastAPI(title="TRAG extractor", version="0.1.0")
router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness, plus the fingerprint so drift is visible without a conversion."""
    return {"status": "ok", "config_fingerprint": extraction_fingerprint()}


@router.post("/extract")
async def extract(
    file: UploadFile = File(...),
    mime: str | None = Form(default=None),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")

    filename = file.filename or "upload"
    try:
        markdown = convert(data, filename)
    except Exception as exc:  # noqa: BLE001
        # Converted to a 422 rather than a 500: a file docling cannot read is a
        # property of the file, not a fault in this service, and the API stores
        # the reason on the document.
        log.exception("conversion failed for %s", filename)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Could not extract {filename}: {str(exc)[:300]}",
        ) from exc

    return {"markdown": markdown, "config_fingerprint": extraction_fingerprint()}


app.include_router(router)
