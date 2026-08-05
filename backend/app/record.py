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


# The subset of extraction_config() that the EXTRACTOR process controls. Must
# stay identical to extractor/pipeline.py:extraction_settings() -- same keys,
# same serialisation -- or the drift check cries wolf on every extraction.
_EXTRACTOR_OWNED_KEYS = (
    "do_ocr",
    "do_table_structure",
    "describe_pictures",
    "vlm_model",
    "vlm_prompt_sha256",
)


def local_extraction_fingerprint() -> str:
    """What this process believes the extractor is configured to do.

    Compared against the fingerprint the extractor returns with every
    conversion. They can only disagree if the two processes read different
    configuration -- at which point config_hash is describing a pipeline that
    never ran, and nothing else in the system would notice.
    """
    cfg = extraction_config()
    subset = {k: cfg[k] for k in _EXTRACTOR_OWNED_KEYS}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
