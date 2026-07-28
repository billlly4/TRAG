"""Anthropic client, traced by LangSmith.

`wrap_anthropic` is observability, not a framework -- it patches the client so
calls show up in LangSmith and otherwise changes nothing about the SDK surface.
No LangChain is involved.
"""

import os

from anthropic import Anthropic
from langsmith.wrappers import wrap_anthropic

from .config import Settings, get_settings

_client: Anthropic | None = None


def _export_langsmith_env(settings: Settings) -> None:
    """Publish LangSmith config into os.environ.

    pydantic-settings reads .env into a Settings object; it does NOT populate
    os.environ. The LangSmith SDK only reads os.environ, so without this step
    tracing silently no-ops even though the key is sitting in .env.
    """
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id


def get_client() -> Anthropic:
    global _client
    if _client is None:
        settings = get_settings()
        _export_langsmith_env(settings)

        base = Anthropic(api_key=settings.anthropic_api_key)
        _client = wrap_anthropic(base) if settings.langsmith_tracing else base
    return _client


# Fields the Messages API accepts on *input* for each block type. The SDK's
# response models carry extra fields that are not valid to send back -- notably
# `parsed_output`, which streamed text blocks include and non-streamed ones do
# not. Replaying an unsanitised assistant turn fails with
# "content.0.text.parsed_output: Extra inputs are not permitted".
_API_BLOCK_FIELDS: dict[str, set[str]] = {
    "text": {"type", "text", "citations", "cache_control"},
    "document": {"type", "source", "citations", "cache_control", "title", "context"},
    "image": {"type", "source", "cache_control"},
    "tool_use": {"type", "id", "name", "input", "cache_control"},
    "tool_result": {"type", "tool_use_id", "content", "is_error", "cache_control"},
    "thinking": {"type", "thinking", "signature"},
}


def sanitize_for_api(content: list[dict]) -> list[dict]:
    """Strip response-only fields from stored blocks before replaying them.

    We persist the full dump because the UI needs it; this narrows it back to
    what the API accepts as input.

    Citation *spans* on assistant text blocks are dropped entirely rather than
    cleaned field-by-field. They are display metadata -- the model does not need
    its own previous citations to continue, since the document is still in
    context -- and trying to round-trip them fails three different ways:
    `file_id` is not accepted, `document_title` is required-but-nullable, and
    char_location spans are rejected outright when the document is referenced
    by file id. Keeping them in the database and dropping them here gets the UI
    what it needs without fighting the input schema.

    The `citations` key on a *document* block is different -- it is the config
    dict {"enabled": true}, and it must survive or citations stop being
    generated at all.
    """
    out: list[dict] = []
    for block in content:
        allowed = _API_BLOCK_FIELDS.get(block.get("type"))
        if allowed is None:
            out.append(block)
            continue

        clean = {k: v for k, v in block.items() if k in allowed and v is not None}

        if isinstance(clean.get("citations"), list):
            clean.pop("citations")

        out.append(clean)
    return out


SYSTEM_PROMPT = (
    "You are a helpful research assistant. The user's uploaded documents are "
    "searchable with the search_documents tool -- they are NOT in your context. "
    "When a question might depend on their documents, your FIRST action is the "
    "tool call -- write no text before it. Never say what you do or don't have "
    "before searching; the UI already shows the search. "
    "Answer from the passages returned, naming the source file. If the search "
    "finds nothing relevant, say so plainly rather than answering from memory. "
    "For questions clearly unrelated to the user's documents, answer directly "
    "without searching."
)
