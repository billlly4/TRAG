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
