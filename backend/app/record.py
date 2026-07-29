"""Record manager fingerprints.

Two hashes decide whether work can be skipped:

- content_hash: what the file IS (sha256 of the raw bytes)
- config_hash:  how it would be PROCESSED (everything that shapes the derived
  chunks -- extraction options, prompts, chunking, embedding model)

A document's stored chunks are current only if BOTH match. Hashing the config
alongside the bytes is the rule from CLAUDE.md: without it, changing the
chunking strategy or a prompt silently leaves the corpus processed under two
different regimes with no way to tell which rows are which.
"""

import hashlib
import json

from .chunking import CHARS_PER_TOKEN, EMBED_CONTEXT_VERSION
from .config import get_settings
from .extract import VLM_PROMPT
from .metadata import prompt_fingerprint

# Bump manually when extractor or chunker CODE changes in a way that alters
# output (a bugfix that changes chunk boundaries, a different markdown export).
# Settings changes are picked up automatically below; code changes are not.
#
# 2: Module 4 -- the chunker now tracks section headings and chunk boundaries
#    shifted with it, so every existing chunk was built by different code.
PIPELINE_VERSION = 2


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extraction_config() -> dict:
    """Everything that affects what chunks a given file produces.

    Stored verbatim on the document row (jsonb) so a stale row can show WHY it
    is stale, not just that it is.
    """
    settings = get_settings()
    return {
        "pipeline_version": PIPELINE_VERSION,
        # extraction
        "do_ocr": True,
        "do_table_structure": True,
        "describe_pictures": settings.describe_pictures,
        "vlm_model": settings.vlm_model,
        "vlm_prompt_sha256": hashlib.sha256(VLM_PROMPT.encode()).hexdigest(),
        # chunking
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "chars_per_token": CHARS_PER_TOKEN,
        # metadata -- in here because the extracted title is prepended to every
        # chunk's embedding input, so a different prompt or schema produces
        # different vectors, not just different columns.
        "metadata_model": settings.metadata_model,
        "metadata_fingerprint": prompt_fingerprint(),
        # embedding
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embed_context_version": EMBED_CONTEXT_VERSION,
    }


def config_hash() -> str:
    canonical = json.dumps(extraction_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
