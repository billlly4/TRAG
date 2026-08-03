"""Structured metadata extraction.

One LLM call per document, at ingest time, producing typed fields the database
can index and filter on. This is what turns "search everything" into "search
the 2023 vendor reports": the filters in match_chunks run against these
columns, before the vector ordering.

Two rules shape the schema below:

- doc_type is a CLOSED Literal. An open string makes filtering useless -- the
  model coins a new label for every document ("quarterly-report", "Q4 report",
  "financial report") and no filter ever matches twice.
- Every field is required but nullable. Pydantic marks a field required only
  when it has no default, so omitting defaults forces the model to emit an
  explicit null instead of dropping the key. "Not stated" and "forgot to
  answer" then look different in the output.

Note what the API does NOT enforce: the SDK strips numeric and length
constraints from the schema it sends (anthropic/lib/_parse/_transform.py) and
appends them to the field description as a hint. Field(max_length=...) still
validates locally, but the model is only nudged, never constrained.
"""

import hashlib
import logging
from datetime import date
from typing import Literal, get_args

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .config import get_settings

log = logging.getLogger(__name__)

# Only the head of the document is sent. Titles, authors, publishers and dates
# live at the front; sending a 200-page textbook to answer "what is this?"
# would cost more than the rest of the pipeline combined.
METADATA_INPUT_CHARS = 24_000

# Bounds the structured response. The schema is small and the summary is
# capped at a few sentences, so this is generous.
METADATA_MAX_TOKENS = 1024


class DocumentMetadata(BaseModel):
    """What a document IS, as opposed to what it says."""

    title: str | None = Field(
        description=(
            "The document's own title, as printed on it. Null if the document "
            "has no title -- do not fall back to the filename."
        )
    )

    doc_type: Literal[
        "report",
        "paper",
        "textbook",
        "manual",
        "contract",
        "policy",
        "presentation",
        "correspondence",
        "other",
    ] = Field(
        description=(
            "The closest match from the list. Use 'other' rather than forcing "
            "a poor fit."
        )
    )

    source_org: str | None = Field(
        description=(
            "The organisation that published or issued the document (not the "
            "individual authors). Null if not stated."
        )
    )

    authors: list[str] = Field(
        description="Named individual authors. Empty list if none are stated."
    )

    published_on: date | None = Field(
        description=(
            "Publication date as YYYY-MM-DD, ONLY when a full day-level date "
            "is explicitly printed. If only a year or month is known, leave "
            "this null and set published_year instead. Never invent a day."
        )
    )

    published_year: int | None = Field(
        description=(
            "Four-digit publication year. Set this whenever the year is known, "
            "including when published_on is null. Null if no date is stated."
        )
    )

    language: str | None = Field(
        description="ISO 639-1 code of the main language, e.g. 'en', 'de'."
    )

    topics: list[str] = Field(
        description=(
            "3 to 8 short lowercase subject keywords for filtering, e.g. "
            "'inventory management', 'tariffs'. Prefer general subject terms "
            "over phrases lifted from the text."
        )
    )

    summary: str = Field(
        description="What this document covers, in 1 to 3 plain sentences."
    )


# Derived from the model rather than repeated, so the retrieval tool's enum
# cannot drift from what extraction is actually allowed to produce.
DOC_TYPES: tuple[str, ...] = get_args(
    DocumentMetadata.model_fields["doc_type"].annotation
)


METADATA_PROMPT = (
    "You extract bibliographic metadata from documents. You are given the "
    "beginning of a document; describe the document as a whole.\n\n"
    "Record only what the document actually states. If a field is not stated, "
    "return null (or an empty list) -- a null is correct and useful, a "
    "plausible guess is neither. Do not infer a date from context, do not "
    "expand an abbreviation you are not sure of, and do not treat a filename "
    "as a title.\n\n"
    "The exception is topics and summary, which you should always produce: "
    "they describe content you can see rather than facts that must be printed."
)


def prompt_fingerprint() -> str:
    """sha256 of everything that steers extraction: the prompt and the schema.

    record.py folds this into config_hash, so editing the prompt or adding a
    field marks every document stale rather than silently leaving the corpus
    described under two different regimes.
    """
    schema = DocumentMetadata.model_json_schema()
    payload = METADATA_PROMPT + repr(sorted(schema.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_metadata(
    markdown: str, filename: str
) -> tuple[DocumentMetadata | None, str | None]:
    """Extract metadata from a document's Markdown. Returns (metadata, error).

    Never raises. Metadata is an enhancement to retrieval, not a prerequisite
    for it -- a document with null metadata is still chunked, embedded and
    searchable, so failing the whole ingest over a bad extraction call would
    trade a working document for no document. The caller records the error
    string on the row instead, where it stays visible.
    """
    settings = get_settings()
    head = markdown[:METADATA_INPUT_CHARS]
    if not head.strip():
        return None, "No text to extract metadata from"

    try:
        # `with_structured_output` rather than a raw Anthropic call. The old
        # version had to use `beta.messages.parse` specifically, because
        # LangSmith's wrap_anthropic patches the beta path and not the stable
        # one -- the stable call worked but was invisible in tracing. LangChain
        # traces natively, so that constraint is gone along with the workaround.
        #
        # PROMPT AND SCHEMA ARE UNCHANGED, deliberately: prompt_fingerprint()
        # hashes both into config_hash, so editing either here would mark every
        # document in every corpus stale and force a full re-ingest.
        #
        # include_raw keeps the underlying message reachable, which the
        # truncation check below needs -- the parsed-only form discards
        # stop_reason.
        model = ChatAnthropic(
            model=settings.metadata_model,
            api_key=settings.anthropic_api_key,
            max_tokens=METADATA_MAX_TOKENS,
        ).with_structured_output(DocumentMetadata, include_raw=True)

        result = model.invoke(
            [
                SystemMessage(content=METADATA_PROMPT),
                HumanMessage(
                    content=(
                        f"Filename: {filename}\n\n"
                        f"--- start of document ---\n{head}"
                    )
                ),
            ]
        )

        raw = result.get("raw")
        stop_reason = (getattr(raw, "response_metadata", None) or {}).get("stop_reason")

        # A truncated structured response is unparseable, not merely short, so
        # this is usually caught below -- but check explicitly, because a
        # schema that just fits produces a valid object built from a cut-off
        # reading of the document.
        if stop_reason == "max_tokens":
            return None, "Metadata extraction hit max_tokens"

        if result.get("parsing_error"):
            return None, f"Metadata parse failed: {result['parsing_error']}"[:500]

        meta = result.get("parsed")
        if meta is None:
            return None, f"No structured output (stop_reason={stop_reason})"

        log.info(
            "metadata for %s: type=%s org=%s year=%s topics=%d",
            filename, meta.doc_type, meta.source_org, meta.published_year,
            len(meta.topics),
        )
        return meta, None

    except Exception as exc:  # noqa: BLE001
        log.warning("metadata extraction failed for %s", filename, exc_info=True)
        return None, str(exc)[:500]
