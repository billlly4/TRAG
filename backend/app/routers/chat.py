import json
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..embeddings import EmbeddingError
from ..llm import SYSTEM_PROMPT, get_client, sanitize_for_api
from ..retrieval import search
from ..schemas import ChatRequest

router = APIRouter(prefix="/api", tags=["chat"])

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the user's uploaded documents for passages relevant to a question. "
        "Call this whenever answering might depend on their documents — do not answer "
        "from memory when the user refers to their files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused search query."},
            "top_k": {"type": "integer", "description": "Passages to return, 1-20. Default 5."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


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

    try:
        hits = search(db, query, top_k if isinstance(top_k, int) else None)
    except EmbeddingError as exc:
        return f"Search unavailable: {exc}", [], True

    if not hits:
        return (
            "No relevant passages found in the user's documents for this query.",
            [],
            False,
        )

    parts = []
    sources = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] {h.filename} (chunk {h.ordinal}, similarity {h.similarity:.2f})\n"
            f"{h.content}"
        )
        sources.append(
            {
                "document_id": h.document_id,
                "filename": h.filename,
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
                    "system": SYSTEM_PROMPT,
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
