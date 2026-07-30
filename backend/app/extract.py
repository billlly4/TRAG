"""Document extraction via docling.

Everything becomes Markdown. Headings and tables survive, which is what gives
the chunker meaningful boundaries -- a heading is a topic break, a table is an
atomic unit. Figures optionally become VLM captions written into the document,
so a revenue chart turns into text that chunks and embeds like any other.
"""

import logging
import tempfile
import threading
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .config import get_settings

log = logging.getLogger(__name__)

# Module-level so record.py can fingerprint it: changing this prompt changes
# what captions (and therefore chunks) a document produces, so it is part of
# the extraction config, not an implementation detail.
VLM_PROMPT = (
    "Describe this figure for a search index. State the chart type, "
    "what is measured, and the key values or trend visible. "
    "Be specific and concise."
)

_converter: DocumentConverter | None = None

# Serialises PDF conversion across threads.
#
# _ingest runs as a FastAPI BackgroundTask, and sync background tasks execute
# in a threadpool. Tasks scheduled by ONE request run sequentially, but tasks
# from different requests do not -- an upload arriving during a re-process
# batch, or two batches overlapping, puts two threads through the converter at
# once. The converter wraps PyTorch/ONNX models (layout, TableFormer, OCR)
# that are not thread-safe, and the failure mode is a SIGSEGV that kills the
# whole server: no Python exception, no traceback, nothing written to
# documents.error, and every in-flight document stranded mid-pipeline.
#
# The lock covers construction as well as conversion, so two threads cannot
# race to build the singleton either.
_converter_lock = threading.Lock()


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
# end to end through extract_text, not merely present in docling's format enum.
SUPPORTED_SUFFIXES = sorted(
    _PASSTHROUGH_SUFFIXES
    | {
        ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    }
)


class UnsupportedFormat(ValueError):
    pass


def _build_converter() -> DocumentConverter:
    settings = get_settings()

    pdf_options = PdfPipelineOptions(
        # TableFormer recovers real table structure and emits Markdown tables
        # rather than mangled text runs.
        do_table_structure=True,
        # Recovers text inside scanned pages and screenshots.
        do_ocr=True,
    )

    if settings.describe_pictures:
        pdf_options.do_picture_description = True
        # Without rendered picture images there is nothing to send to the VLM
        # and captioning silently no-ops.
        pdf_options.generate_picture_images = True
        pdf_options.images_scale = 2.0
        # Remote services (the local Ollama daemon counts as one) are opt-in.
        pdf_options.enable_remote_services = True
        pdf_options.picture_description_options = PictureDescriptionApiOptions(
            url=f"{settings.ollama_base_url.rstrip('/')}/v1/chat/completions",
            params={"model": settings.vlm_model},
            prompt=VLM_PROMPT,
            timeout=300,
        )

    return DocumentConverter(
        # Belt and braces with the suffix check in extract_text: docling's
        # default allows every format it knows, including audio and video, so a
        # mislabelled or extensionless file could otherwise reach a pipeline
        # that wants to download ASR models mid-ingest.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.CSV,
            InputFormat.HTML,
            InputFormat.IMAGE,
            InputFormat.MD,
        ],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def _get_converter() -> DocumentConverter:
    # Cached because converter construction loads models (layout, TableFormer,
    # OCR) -- rebuilding per document would dominate small-file ingest time.
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter


def extract_text(data: bytes, filename: str, mime: str | None = None) -> str:
    """Extract a document's content as Markdown.

    Plain text passes through untouched -- docling would only add overhead and
    it has no structure to recover. Everything else goes through conversion,
    including HTML and CSV: they are text/* but they are not prose, and storing
    them verbatim means embedding markup and raw rows.
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

    # docling wants a path or URL, not bytes. delete=False because Windows
    # cannot reopen a NamedTemporaryFile while it is still open.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        # One document through the models at a time -- see _converter_lock.
        # Concurrent ingests queue here rather than crashing the process; the
        # throughput cost is nil, because conversion was already the
        # bottleneck and Ollama serialises the embedding step anyway.
        with _converter_lock:
            result = _get_converter().convert(tmp.name)
            markdown = result.document.export_to_markdown()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    log.info("extracted %s: %d chars of markdown", filename, len(markdown))
    return markdown
