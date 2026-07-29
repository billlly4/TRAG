import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..embeddings import EmbeddingError
from ..llm import SYSTEM_PROMPT, get_client, sanitize_for_api
from ..metadata import DOC_TYPES
from ..retrieval import Filters, search
from ..schemas import ChatRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the user's uploaded documents for passages relevant to a question. "
        "Call this whenever answering might depend on their documents — do not answer "
        "from memory when the user refers to their files. "
        "The optional filters narrow the search before ranking; use them only when "
        "the question is explicitly about a kind of document, a source, or a period, "
        "and only with values listed in the corpus summary. Filtering on a value "
        "that is not in the corpus returns nothing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused search query."},
            "top_k": {"type": "integer", "description": "Passages to return, 1-20. Default 5."},
            "doc_type": {
                "type": "array",
                "items": {"type": "string", "enum": list(DOC_TYPES)},
                "description": "Restrict to these document types.",
            },
            "source_org": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict to these publishing organisations. Must match the "
                    "corpus summary exactly."
                ),
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict to documents tagged with any of these topics.",
            },
            "year_min": {"type": "integer", "description": "Earliest publication year."},
            "year_max": {"type": "integer", "description": "Latest publication year."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

# Caps on the corpus summary. A user with hundreds of documents should not
# push the whole vocabulary into every request's system prompt.
_SUMMARY_MAX_VALUES = 30


def corpus_summary(db) -> str:
    """A compact inventory of what is actually in the user's corpus.

    Without this the model invents plausible filter values -- doc_type
    "invoice", source_org "Acme Corp" -- that match nothing, and then reports
    that the corpus has no answer. Injected into the system prompt rather than
    exposed as a tool: a tool would cost an extra round trip on every question,
    and Module 2 already measured that there is no prompt cache to protect
    (cache_read_input_tokens is 0 at this prompt size).
    """
    try:
        res = (
            db.table("documents")
            .select("doc_type,source_org,topics,published_year")
            .eq("status", "ready")
            .execute()
        )
    except Exception:  # noqa: BLE001
        log.warning("corpus summary unavailable", exc_info=True)
        return ""

    rows = res.data or []
    if not rows:
        return "\n\nThe user has no processed documents yet."

    def distinct(key: str) -> list[str]:
        seen = {r[key] for r in rows if r.get(key)}
        return sorted(seen)[:_SUMMARY_MAX_VALUES]

    topics = sorted({t for r in rows for t in (r.get("topics") or [])})
    years = sorted({r["published_year"] for r in rows if r.get("published_year")})

    lines = [f"\n\nThe user's corpus: {len(rows)} document(s)."]
    if types := distinct("doc_type"):
        lines.append(f"Types: {', '.join(types)}.")
    if orgs := distinct("source_org"):
        lines.append(f"Sources: {', '.join(orgs)}.")
    if topics:
        lines.append(f"Topics: {', '.join(topics[:_SUMMARY_MAX_VALUES])}.")
    if years:
        lines.append(f"Years: {years[0]}–{years[-1]}.")
    lines.append(
        "Only ever filter on values from these lists; otherwise search unfiltered."
    )
    return " ".join(lines)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run_search(db, tool_input: dict[str, Any]) -> tuple[str, list[dict], bool]:
    """Execute one search_documents call.

    Returns (text for the model, sources for the UI, is_error). An unreachable
    Ollama comes back as a tool error the model can relay, not a dead stream.
    """
    query = str(tool_input.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required.", [], True
    top_k = tool_input.get("top_k")

    def as_list(key: str) -> list[str]:
        value = tool_input.get(key)
        if isinstance(value, str):  # tolerate a bare string where a list is asked for
            return [value]
        return [str(v) for v in value] if isinstance(value, list) else []

    year_min, year_max = tool_input.get("year_min"), tool_input.get("year_max")
    filters = Filters(
        doc_types=as_list("doc_type"),
        source_orgs=as_list("source_org"),
        topics=as_list("topics"),
        year_min=year_min if isinstance(year_min, int) else None,
        year_max=year_max if isinstance(year_max, int) else None,
    )

    try:
        hits = search(db, query, top_k if isinstance(top_k, int) else None, filters)
    except EmbeddingError as exc:
        return f"Search unavailable: {exc}", [], True

    if not hits:
        # Naming the filters matters: told only "nothing found", the model
        # concludes the corpus cannot answer and stops. Told the filters
        # matched nothing, it retries without them.
        if filters.active():
            return (
                f"No passages matched the filters ({filters.describe()}). "
                f"The filters may not match this corpus — retry without them "
                f"before concluding the documents have no answer.",
                [],
                False,
            )
        return (
            "No relevant passages found in the user's documents for this query.",
            [],
            False,
        )

    parts = []
    sources = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] {h.label()} (chunk {h.ordinal}, similarity {h.similarity:.2f})\n"
            f"{h.content}"
        )
        sources.append(
            {
                "document_id": h.document_id,
                "filename": h.filename,
                "section": h.section,
                "ordinal": h.ordinal,
                "similarity": round(h.similarity, 3),
            }
        )
    return "\n\n".join(parts), sources, False


@router.post("/chat")
def chat(
    body: ChatRequest,
    user: CurrentUserDep,
    db: DbDep,
    settings: SettingsDep,
) -> StreamingResponse:
    # RLS means a thread belonging to someone else simply is not visible here,
    # so "not found" and "not yours" are the same answer.
    thread_res = (
        db.table("threads").select("id,title").eq("id", body.thread_id).execute()
    )
    if not thread_res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")
    thread = thread_res.data[0]

    history_res = (
        db.table("messages")
        .select("role,content")
        .eq("thread_id", body.thread_id)
        .order("created_at")
        .execute()
    )

    # Module 1 turns carry `document` blocks referencing Anthropic's Files API,
    # which this app no longer uses (no beta header, files may be deleted).
    # Dropping them keeps old threads replayable; the text of those turns
    # survives.
    history = []
    for m in history_res.data:
        content = [
            b for b in sanitize_for_api(m["content"]) if b.get("type") != "document"
        ]
        if content:
            history.append({"role": m["role"], "content": content})

    user_content = [{"type": "text", "text": body.message}]

    # Persisted before the call, not after. If the stream dies mid-flight the
    # user still sees their own message on reload, which matches what happened.
    db.table("messages").insert(
        {
            "thread_id": body.thread_id,
            "user_id": user.id,
            "role": "user",
            "content": user_content,
        }
    ).execute()

    messages = history + [{"role": "user", "content": user_content}]
    client = get_client()

    # Once per request, not per tool turn -- the corpus cannot change
    # mid-answer, and re-querying it on every loop iteration would be waste.
    system_prompt = SYSTEM_PROMPT + corpus_summary(db)

    def persist(role: str, content: list[dict], **extra: Any) -> dict:
        res = (
            db.table("messages")
            .insert(
                {
                    "thread_id": body.thread_id,
                    "user_id": user.id,
                    "role": role,
                    "content": content,
                    **extra,
                }
            )
            .execute()
        )
        return res.data[0]

    def event_stream() -> Iterator[str]:
        try:
            final = None
            saved = None
            for turn in range(settings.max_tool_turns):
                kwargs: dict[str, Any] = {
                    "model": settings.anthropic_model,
                    "max_tokens": settings.max_output_tokens,
                    "system": system_prompt,
                    "tools": [SEARCH_TOOL],
                    "messages": messages,
                }
                # Last permitted turn: force prose so the loop cannot end on a
                # dangling tool call.
                if turn == settings.max_tool_turns - 1:
                    kwargs["tool_choice"] = {"type": "none"}

                with client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        yield _sse({"type": "delta", "text": text})
                    final = stream.get_final_message()

                content = [b.model_dump() for b in final.content]
                usage = final.usage.model_dump() if final.usage else None
                saved = persist(
                    "assistant", content, stop_reason=final.stop_reason, usage=usage
                )
                messages.append(
                    {"role": "assistant", "content": sanitize_for_api(content)}
                )

                if final.stop_reason != "tool_use":
                    break

                # All tool_result blocks go in a SINGLE user message. Splitting
                # them across messages trains Claude to stop issuing parallel
                # tool calls.
                results = []
                for block in final.content:
                    if block.type != "tool_use":
                        continue
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    yield _sse(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": tool_input,
                        }
                    )
                    text, sources, is_error = _run_search(db, tool_input)
                    yield _sse(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "sources": sources,
                            "is_error": is_error,
                        }
                    )
                    # `sources` is for the UI on reload; sanitize_for_api
                    # strips it before the blocks are replayed to the API,
                    # same as Module 1 did with citation spans.
                    result: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        "sources": sources,
                    }
                    if is_error:
                        result["is_error"] = True
                    results.append(result)

                persist("user", results)
                messages.append({"role": "user", "content": sanitize_for_api(results)})

            updates: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            if not thread.get("title"):
                updates["title"] = body.message[:60]
            db.table("threads").update(updates).eq("id", body.thread_id).execute()

            usage = final.usage.model_dump() if final and final.usage else None
            yield _sse(
                {
                    "type": "done",
                    "message_id": saved["id"],
                    "stop_reason": final.stop_reason,
                    # 'max_tokens' here means the reply was cut off. On the wire
                    # it is indistinguishable from a finished answer, so the
                    # client must read this field rather than assume completion.
                    "truncated": final.stop_reason == "max_tokens",
                    "content": [b.model_dump() for b in final.content],
                    "usage": usage,
                }
            )

        except Exception as exc:  # noqa: BLE001
            # Headers are already sent, so the status code cannot change. An
            # explicit error frame is the only way to distinguish a failure
            # from a short-but-complete answer.
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies buffering
        },
    )
