from typing import Any, Literal

from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """The authenticated caller, derived from a verified Supabase JWT."""

    id: str
    email: str | None = None

    # The caller's raw access token. Forwarded to PostgREST so Postgres
    # evaluates RLS as this user rather than us re-implementing authorisation
    # in application code.
    token: str = Field(repr=False)


class HealthResponse(BaseModel):
    status: str
    model: str


class MeResponse(BaseModel):
    id: str
    email: str | None = None


# --- threads -----------------------------------------------------------------


class ThreadCreate(BaseModel):
    title: str | None = None


class ThreadOut(BaseModel):
    id: str
    title: str | None = None
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    thread_id: str
    role: Literal["user", "assistant"]

    # A list of Claude content blocks, not a string. Text, document references
    # and citation spans all live here.
    content: list[dict[str, Any]]

    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    created_at: str


# --- chat --------------------------------------------------------------------


class ChatRequest(BaseModel):
    thread_id: str
    message: str = Field(min_length=1)


# --- documents ---------------------------------------------------------------


class DocumentOut(BaseModel):
    id: str
    filename: str
    mime_type: str | None = None
    byte_size: int | None = None
    created_at: str

    # Ingestion pipeline state: pending -> extracting -> chunking -> embedding
    # -> ready, or failed (with `error` saying why). The UI badge renders this,
    # pushed live over Supabase Realtime.
    status: str
    error: str | None = None
    chunk_count: int | None = None

    # True when the stored chunks were built under a different extraction
    # config than the current one (or under none at all -- legacy rows).
    # Computed server-side against record.config_hash(); never stored.
    stale: bool = False

    # Extracted at ingest (backend/app/metadata.py). All optional: a document
    # whose metadata extraction failed, or one ingested before Module 4, is
    # still fully searchable -- it just cannot be filtered on.
    title: str | None = None
    doc_type: str | None = None
    source_org: str | None = None
    published_year: int | None = None
    topics: list[str] | None = None
    summary: str | None = None
