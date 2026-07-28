from fastapi import APIRouter, HTTPException, status

from ..deps import CurrentUserDep, DbDep
from ..schemas import MessageOut, ThreadCreate, ThreadOut

router = APIRouter(prefix="/api/threads", tags=["threads"])


@router.post("", response_model=ThreadOut, status_code=status.HTTP_201_CREATED)
def create_thread(body: ThreadCreate, user: CurrentUserDep, db: DbDep) -> ThreadOut:
    res = (
        db.table("threads")
        .insert({"user_id": user.id, "title": body.title})
        .execute()
    )
    return ThreadOut(**res.data[0])


@router.get("", response_model=list[ThreadOut])
def list_threads(user: CurrentUserDep, db: DbDep) -> list[ThreadOut]:
    # No .eq("user_id", ...) filter here on purpose: RLS already restricts the
    # rows to this user. Adding one would imply the database needs our help.
    res = (
        db.table("threads")
        .select("id,title,created_at,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return [ThreadOut(**row) for row in res.data]


@router.get("/{thread_id}/messages", response_model=list[MessageOut])
def list_messages(thread_id: str, user: CurrentUserDep, db: DbDep) -> list[MessageOut]:
    """Replay a conversation.

    The Messages API is stateless -- Anthropic keeps nothing between calls. If
    this table is empty, the conversation is simply gone.
    """
    res = (
        db.table("messages")
        .select("id,thread_id,role,content,stop_reason,usage,created_at")
        .eq("thread_id", thread_id)
        .order("created_at")
        .execute()
    )
    return [MessageOut(**row) for row in res.data]


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(thread_id: str, user: CurrentUserDep, db: DbDep) -> None:
    # Messages go with it via ON DELETE CASCADE, not application code.
    res = db.table("threads").delete().eq("id", thread_id).execute()
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")
