"""docling conversion. Runs in the extractor process, never in the API.

Everything becomes Markdown. Headings and tables survive, which is what gives
the chunker meaningful boundaries -- a heading is a topic break, a table is an
atomic unit. Figures optionally become VLM captions written into the document,
so a revenue chart turns into text that chunks and embeds like any other.

This module lives here rather than in backend/app because docling wraps native
model runtimes that can fault the whole process. Keeping it out of the API means
a segfault costs an extraction, not the server.
"""

import hashlib
import json
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

from ..app.config import get_settings

# Imported from the API package on purpose: record.py fingerprints this prompt
# as part of config_hash, so the two processes MUST use the same string. One
# definition, imported twice, is the only way that stays true.
from ..app.extract import VLM_PROMPT

log = logging.getLogger(__name__)

_converter: DocumentConverter | None = None

# Serialises conversion across threads.
#
# docling wraps PyTorch/ONNX models (layout, TableFormer, OCR) that are not
# thread-safe, and the failure mode is a SIGSEGV that kills the process: no
# Python exception, no traceback, nothing written anywhere. This lock is why
# concurrent requests queue instead of crashing.
#
# It covers construction as well as conversion, so two threads cannot race to
# build the singleton either.
_converter_lock = threading.Lock()


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
        # Belt and braces with the suffix check in the API: docling's default
        # allows every format it knows, including audio and video, so a
        # mislabelled or extensionless file could otherwise reach a pipeline
        # that wants to download ASR models mid-request.
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


def extraction_settings() -> dict:
    """The settings that shape this service's output.

    Hashed and returned with every response so the API can detect the two
    processes disagreeing. Without it, running the extractor with a different
    DESCRIBE_PICTURES than the API assumes would make config_hash describe a
    pipeline that never ran -- exactly the two-regimes failure Module 3 exists
    to prevent, but now split across process boundaries where nothing looks
    wrong from either side.
    """
    settings = get_settings()
    return {
        "do_ocr": True,
        "do_table_structure": True,
        "describe_pictures": settings.describe_pictures,
        "vlm_model": settings.vlm_model,
        "vlm_prompt_sha256": hashlib.sha256(VLM_PROMPT.encode()).hexdigest(),
    }


def extraction_fingerprint() -> str:
    canonical = json.dumps(extraction_settings(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def convert(data: bytes, filename: str) -> str:
    """Convert bytes to Markdown. The caller has already gated the format."""
    suffix = Path(filename).suffix.lower()

    # docling wants a path or URL, not bytes. delete=False because Windows
    # cannot reopen a NamedTemporaryFile while it is still open.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        # One document through the models at a time -- see _converter_lock.
        with _converter_lock:
            result = _get_converter().convert(tmp.name)
            markdown = result.document.export_to_markdown()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    log.info("extracted %s: %d chars of markdown", filename, len(markdown))
    return markdown
