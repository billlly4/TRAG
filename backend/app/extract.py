"""Document extraction via docling.

Everything becomes Markdown. Headings and tables survive, which is what gives
the chunker meaningful boundaries -- a heading is a topic break, a table is an
atomic unit. Figures optionally become VLM captions written into the document,
so a revenue chart turns into text that chunks and embeds like any other.
"""

import logging
import tempfile
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .config import get_settings

log = logging.getLogger(__name__)

_converter: DocumentConverter | None = None


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
            prompt=(
                "Describe this figure for a search index. State the chart type, "
                "what is measured, and the key values or trend visible. "
                "Be specific and concise."
            ),
            timeout=300,
        )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
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
    it has no structure to recover.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".markdown"} or (mime or "").startswith("text/"):
        return data.decode("utf-8", errors="replace")

    # docling wants a path or URL, not bytes. delete=False because Windows
    # cannot reopen a NamedTemporaryFile while it is still open.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        result = _get_converter().convert(tmp.name)
        markdown = result.document.export_to_markdown()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    log.info("extracted %s: %d chars of markdown", filename, len(markdown))
    return markdown
