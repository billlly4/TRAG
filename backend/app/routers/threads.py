from fastapi import APIRouter, HTTPException, status

from ..deps import CurrentUserDep, DbDep, SettingsDep
from ..schemas import MessageOut, ThreadCreate, ThreadOut, UsageResponse

router = APIRouter(prefix="/api/threads", tags=["threads"])

# Mounted separately from the /api/threads prefix -- usage spans threads and
# documents, so it belongs to neither.
usage_router = APIRouter(prefix="/api", tags=["usage"])


@usage_router.get("/usage", response_model=UsageResponse)
def get_usage(user: CurrentUserDep, db: DbDep, settings: SettingsDep) -> UsageResponse:
    """Current consumption against the quotas. RLS scopes both counts."""
    threads = db.table("threads").select("id", count="exact").execute()
    docs = db.table("documents").select("byte_size").execute().data or []
    return UsageResponse(
        threads_used=threads.count or 0,
        threads_limit=settings.max_threads_per_user,
        storage_used_bytes=sum(d["byte_size"] or 0 for d in docs),
        storage_limit_bytes=settings.max_storage_bytes,
        messages_per_thread_limit=settings.max_messages_per_thread,
    )


@router.post("", response_model=ThreadOut, status_code=status.HTTP_201_CREATED)
def create_thread(
    body: ThreadCreate, user: CurrentUserDep, db: DbDep, settings: SettingsDep
) -> ThreadOut:
    # The real enforcement is a BEFORE INSERT trigger (0006_quotas.sql) -- this
    # check exists so the user reads "you have 5 of 5 chats" instead of a raw
    # Postgres check_violation. RLS scopes the count to the caller.
    existing = db.table("threads").select("id", count="exact").execute()
    if (existing.count or 0) >= settings.max_threads_per_user:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"You have {existing.count} of {settings.max_threads_per_user} chats. "
            f"Delete one before starting another.",
        )

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
