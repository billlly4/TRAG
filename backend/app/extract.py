"""Extraction: format policy here, conversion in a separate process.

This module decides WHAT may be extracted; `backend/extractor` does the work.
The split exists because docling wraps native model runtimes that can fault the
whole process -- a SIGSEGV in a PDF conversion used to take the API down with
it, along with every in-flight document. Now it costs one extraction.

Deliberately imports no docling. Keeping the API free of it is most of the
point: no heavyweight model imports at startup, and no way for a conversion
fault to reach the process serving chat.
"""

import logging
from pathlib import Path

import httpx

from .config import get_settings

log = logging.getLogger(__name__)

# Module-level so record.py can fingerprint it, and so the extractor imports the
# same string rather than keeping a second copy. Changing this prompt changes
# what captions (and therefore chunks) a document produces, so it is part of the
# extraction config, not an implementation detail.
VLM_PROMPT = (
    "Describe this figure for a search index. State the chart type, "
    "what is measured, and the key values or trend visible. "
    "Be specific and concise."
)

# Extensions read straight off disk as text. Deliberately NOT "any text/* MIME
# type": text/html and text/csv are both text/*, and passing them through
# unconverted stores markup and raw rows instead of Markdown -- HTML tags then
# get chunked and embedded as if they were prose.
_PASSTHROUGH_SUFFIXES = {".txt", ".md", ".markdown"}
_PASSTHROUGH_MIMES = {"text/plain", "text/markdown", "text/x-markdown"}

# Formats docling can name but this app will not attempt. Audio and video need
# ASR models we do not ship and would otherwise trigger a large model download
# mid-ingest; better to reject in milliseconds with a readable reason.
_REJECTED_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".vtt",
}

# What the UI offers and what the docs claim. Every one of these was verified
# end to end through the real extractor, not merely present in docling's format
# enum. Kept in sync by hand with frontend/src/lib/formats.ts.
SUPPORTED_SUFFIXES = sorted(
    _PASSTHROUGH_SUFFIXES
    | {
        ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    }
)


class UnsupportedFormat(ValueError):
    pass


class ExtractionUnavailable(RuntimeError):
    """The extraction service could not be reached.

    Loud on purpose, like EmbeddingError for Ollama: a document stuck with no
    explanation is worse than one marked failed with a reason naming the thing
    that is not running.
    """


def extract_text(data: bytes, filename: str, mime: str | None = None) -> str:
    """Extract a document's content as Markdown.

    Plain text passes through here without touching the service -- there is no
    structure to recover and a network round trip would only add a way to fail.
    Everything else is converted remotely, including HTML and CSV: they are
    text/* but they are not prose, and storing them verbatim means embedding
    markup and raw rows.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in _REJECTED_SUFFIXES:
        raise UnsupportedFormat(
            f"{suffix} files are not supported (audio and video need speech "
            f"recognition models this app does not ship). Supported: "
            f"{', '.join(SUPPORTED_SUFFIXES)}"
        )

    if suffix in _PASSTHROUGH_SUFFIXES or (mime or "").lower() in _PASSTHROUGH_MIMES:
        return data.decode("utf-8", errors="replace")

    settings = get_settings()
    url = f"{settings.extractor_url.rstrip('/')}/extract"

    try:
        response = httpx.post(
            url,
            files={"file": (filename, data, mime or "application/octet-stream")},
            data={"mime": mime or ""},
            # Generous: a large PDF with VLM captioning legitimately runs for
            # minutes. Module 2 measured 56s for a single page with one figure.
            timeout=httpx.Timeout(settings.extractor_timeout, connect=5.0),
        )
    except httpx.RequestError as exc:
        raise ExtractionUnavailable(
            f"Extraction service unreachable at {settings.extractor_url} -- is it "
            f"running? (`uv run uvicorn backend.extractor.main:app --port 8001`)"
        ) from exc

    if response.status_code >= 400:
        # 422 is the service saying the FILE is bad; anything else is the
        # service itself being unhappy. Both end up on documents.error, but the
        # wording should not blame the user's file for a service fault.
        detail = response.text[:300]
        if response.status_code == 422:
            raise ValueError(detail)
        raise ExtractionUnavailable(
            f"Extraction service error {response.status_code}: {detail}"
        )

    payload = response.json()
    _warn_on_config_drift(payload.get("config_fingerprint"))
    return payload["markdown"]


def _warn_on_config_drift(remote_fingerprint: str | None) -> None:
    if not remote_fingerprint:
        return
    from .record import local_extraction_fingerprint

    local = local_extraction_fingerprint()
    if remote_fingerprint != local:
        log.error(
            "EXTRACTION CONFIG DRIFT: extractor fingerprint %s != API %s. "
            "config_hash now describes a pipeline that did not run -- check that "
            "both processes read the same .env (DESCRIBE_PICTURES, VLM_MODEL).",
            remote_fingerprint[:12], local[:12],
        )
