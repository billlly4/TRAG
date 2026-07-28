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

    # Documents to attach to this turn. They are persisted into the message
    # content, so replaying history keeps them (and their cache breakpoint) in
    # exactly the same position on every subsequent call.
    document_ids: list[str] = Field(default_factory=list)


# --- documents ---------------------------------------------------------------


class DocumentOut(BaseModel):
    id: str
    filename: str
    mime_type: str | None = None
    byte_size: int | None = None
    token_estimate: int | None = None
    created_at: str

    # Exposed so the UI can match a document against the `document` blocks
    # already present in a thread's history, and stop you attaching the same
    # file twice. It is an opaque handle -- useless without the API key, which
    # never leaves the server.
    anthropic_file_id: str
