import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from ..agent import blocks_of, build_agent, text_of, to_lc_messages
from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..llm import SYSTEM_PROMPT
from ..schemas import ChatRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

_SUMMARY_MAX_VALUES = 30


def corpus_summary(db) -> str:
    """A compact inventory of what is actually in the user's corpus.

    Without this the model invents plausible filter values -- doc_type
    "invoice", source_org "Acme Corp" -- that match nothing, and then reports
    that the corpus has no answer. Injected into the system prompt rather than
    exposed as a tool: a tool would cost an extra round trip on every question.

    That injection now lands inside the CACHED prefix, so this text is part of
    what prompt caching is preserving between turns. It is recomputed per
    request, which means a document finishing ingestion mid-conversation
    changes the system prompt and invalidates the cache for that thread. That
    is correct -- the model has to see the new corpus -- and it is the reason
    cache hit rates dip while an upload is processing rather than a bug.
    """
    try:
        res = (
            db.table("documents")
            .select("filename,doc_type,source_org,topics,published_year")
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

    # Filenames, because without them the model cannot know a file exists until
    # it has spent a metadata query finding out -- and observed worse, it opens
    # by telling the user they have no such file, then contradicts itself one
    # tool call later. Naming them also lets it pass `document=` on the first
    # search rather than guessing. Capped: past a few dozen this stops being a
    # summary, and the metadata tool is the right instrument for a large corpus.
    if files := distinct("filename"):
        shown = ", ".join(files)
        more = len(rows) - len(files)
        lines.append(f"Files: {shown}{f' (+{more} more)' if more > 0 else ''}.")

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


def _web_results_from_block(block: dict) -> tuple[list[dict], str | None]:
    """Pull (results, error) out of a web_search_tool_result block.

    Takes a plain dict now: ChatAnthropic hands back Anthropic-shaped blocks
    already parsed, where the raw SDK gave typed objects.

    Web search failures arrive as a successful HTTP 200 whose `content` is a
    single error object rather than a list -- there is no exception to catch.
    Treating that shape as an empty result set would render a failed search as
    "found nothing", which is a different and much more misleading claim.
    """
    content = block.get("content")
    if not isinstance(content, list):
        code = (content or {}).get("error_code") if isinstance(content, dict) else None
        return [], str(code or "unknown_error")

    results = []
    for item in content:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        results.append(
            {
                "url": url,
                "title": item.get("title") or url,
                "page_age": item.get("page_age"),
            }
        )
    return results, None


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
        .select("role,content", count="exact")
        .eq("thread_id", body.thread_id)
        .order("created_at")
        .execute()
    )

    # Checked before anything is written. A turn inserts several rows (the
    # question, the tool call, the tool results, the answer), so refusing the
    # whole turn up front is the only way to avoid ending it half-persisted when
    # the trigger fires partway through. The trigger is still the real limit --
    # this is what turns it into a sentence the user can act on.
    if (history_res.count or 0) >= settings.max_messages_per_thread:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This chat has reached its {settings.max_messages_per_thread}-message "
            f"limit. Start a new chat to continue.",
        )

    # Module 1 turns carry `document` blocks referencing Anthropic's Files API,
    # which this app no longer uses (no beta header, files may be deleted).
    # Dropping them keeps old threads replayable; the text of those turns
    # survives. Conversion to LangChain messages happens in `to_lc_messages`.
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

    lc_messages = to_lc_messages(history_res.data) + [HumanMessage(content=body.message)]

    # Once per request, not per tool turn -- the corpus cannot change
    # mid-answer, and re-querying it on every loop iteration would be waste.
    system_prompt = SYSTEM_PROMPT + corpus_summary(db)

    # Compiled per request because the tools close over this caller's
    # RLS-scoped client, and because web search is gated by capability: when
    # body.web_search is false the tool is not in the graph at all.
    agent = build_agent(settings, db, system_prompt, web_search=body.web_search)

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
        """Drive the LangGraph agent and translate its stream into SSE frames.

        The graph owns the loop now. What is left here is the boundary work the
        graph deliberately does not do: persisting each turn to Postgres (the
        UI reads that table, not a checkpointer) and shaping frames the frontend
        already understands, so the rewrite needs no frontend change.

        Two stream modes, because they answer different questions:
          "messages" - token chunks, for the typing effect
          "updates"  - completed messages per node, which is where tool calls,
                       tool results and their artifacts become visible
        """
        try:
            saved: dict[str, Any] | None = None
            last_ai: AIMessage | None = None

            for mode, chunk in agent.stream(
                {"messages": lc_messages}, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    msg, _meta = chunk
                    if isinstance(msg, AIMessageChunk):
                        text = text_of(msg)
                        if text:
                            yield _sse({"type": "delta", "text": text})
                    continue

                # mode == "updates": one entry per node that just finished.
                for _node, value in (chunk or {}).items():
                    for m in (value or {}).get("messages", []) or []:
                        if isinstance(m, AIMessage):
                            last_ai = m
                            blocks = blocks_of(m)
                            usage = m.usage_metadata or None
                            saved = persist(
                                "assistant",
                                blocks,
                                stop_reason=(m.response_metadata or {}).get("stop_reason"),
                                usage=dict(usage) if usage else None,
                            )
                            # Web search runs on Anthropic's side, so its results
                            # arrive on the assistant message rather than through
                            # the tool node -- surfaced separately so the UI can
                            # attribute a web claim to a URL and not to the
                            # user's documents.
                            for b in blocks:
                                if b.get("type") == "web_search_tool_result":
                                    results, error = _web_results_from_block(b)
                                    yield _sse(
                                        {
                                            "type": "web_results",
                                            "tool_use_id": b.get("tool_use_id"),
                                            "results": results,
                                            "error": error,
                                        }
                                    )
                            for call in m.tool_calls or []:
                                yield _sse(
                                    {
                                        "type": "tool_use",
                                        "id": call.get("id"),
                                        "name": call.get("name"),
                                        "input": call.get("args") or {},
                                    }
                                )

                        elif isinstance(m, ToolMessage):
                            # The artifact is how a tool returns UI metadata
                            # without showing it to the model -- `sources` for
                            # retrieval, `sql` for a metadata query.
                            art = m.artifact if isinstance(m.artifact, dict) else {}
                            sources = art.get("sources") or []
                            sql_meta = art.get("sql")
                            # An exact content count. Carried separately from
                            # `sources` because it has none -- rendering it
                            # through the passage list would show "no relevant
                            # passages found" over the top of a complete answer.
                            count = art if "count" in art else None
                            is_error = bool(art.get("is_error")) or m.status == "error"
                            yield _sse(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": m.tool_call_id,
                                    "sources": sources,
                                    "sql": sql_meta,
                                    "count": count,
                                    "is_error": is_error,
                                }
                            )
                            stored: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content if isinstance(m.content, str) else str(m.content),
                                "sources": sources,
                            }
                            if sql_meta is not None:
                                stored["sql"] = sql_meta
                            if count is not None:
                                stored["count"] = count
                            if is_error:
                                stored["is_error"] = True
                            persist("user", [stored])

            updates: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            if not thread.get("title"):
                updates["title"] = body.message[:60]
            db.table("threads").update(updates).eq("id", body.thread_id).execute()

            stop_reason = (last_ai.response_metadata or {}).get("stop_reason") if last_ai else None
            usage = last_ai.usage_metadata if last_ai else None
            yield _sse(
                {
                    "type": "done",
                    "message_id": saved["id"] if saved else None,
                    "stop_reason": stop_reason,
                    # 'max_tokens' here means the reply was cut off. On the wire
                    # it is indistinguishable from a finished answer, so the
                    # client must read this field rather than assume completion.
                    "truncated": stop_reason == "max_tokens",
                    "content": blocks_of(last_ai) if last_ai else [],
                    "usage": dict(usage) if usage else None,
                }
            )

        except Exception as exc:  # noqa: BLE001
            # Headers are already sent, so the status code cannot change. An
            # explicit error frame is the only way to distinguish a failure
            # from a short-but-complete answer.
            log.exception("chat stream failed")
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
