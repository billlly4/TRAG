import json
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..llm import SYSTEM_PROMPT, document_block, get_client, sanitize_for_api
from ..schemas import ChatRequest

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _build_user_content(text: str, file_ids: list[str]) -> list[dict[str, Any]]:
    """Assemble one user turn: documents first, question last.

    Order matters for caching. The cache breakpoint goes on the final document
    block, so everything before the question is a stable prefix that can be
    read back on later turns. Putting the question first would make the prefix
    change every time and the cache would never hit.
    """
    blocks: list[dict[str, Any]] = []
    for i, fid in enumerate(file_ids):
        blocks.append(document_block(fid, cache=(i == len(file_ids) - 1)))
    blocks.append({"type": "text", "text": text})
    return blocks


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

    file_ids: list[str] = []
    if body.document_ids:
        docs = (
            db.table("documents")
            .select("anthropic_file_id")
            .in_("id", body.document_ids)
            .execute()
        )
        file_ids = [d["anthropic_file_id"] for d in docs.data]
        if len(file_ids) != len(body.document_ids):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document id")

    history_res = (
        db.table("messages")
        .select("role,content")
        .eq("thread_id", body.thread_id)
        .order("created_at")
        .execute()
    )
    history = [
        {"role": m["role"], "content": sanitize_for_api(m["content"])}
        for m in history_res.data
    ]

    user_content = _build_user_content(body.message, file_ids)

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

    def event_stream() -> Iterator[str]:
        try:
            with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=settings.max_output_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield _sse({"type": "delta", "text": text})

                final = stream.get_final_message()

            content = [b.model_dump() for b in final.content]
            usage = final.usage.model_dump() if final.usage else None

            saved = (
                db.table("messages")
                .insert(
                    {
                        "thread_id": body.thread_id,
                        "user_id": user.id,
                        "role": "assistant",
                        "content": content,
                        "stop_reason": final.stop_reason,
                        "usage": usage,
                    }
                )
                .execute()
            )

            updates: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            if not thread.get("title"):
                updates["title"] = body.message[:60]
            db.table("threads").update(updates).eq("id", body.thread_id).execute()

            yield _sse(
                {
                    "type": "done",
                    "message_id": saved.data[0]["id"],
                    "stop_reason": final.stop_reason,
                    # 'max_tokens' here means the reply was cut off. On the wire
                    # it is indistinguishable from a finished answer, so the
                    # client must read this field rather than assume completion.
                    "truncated": final.stop_reason == "max_tokens",
                    "content": content,
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
