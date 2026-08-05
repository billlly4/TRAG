-- Per-user quotas, enforced by triggers.
--
-- The API checks the same limits first, purely so the user gets a readable
-- message instead of a raised Postgres exception. The trigger is the actual
-- enforcement; the handler is the manners.
--
-- ALL THREE TRIGGER FUNCTIONS ARE `security definer`, for one reason: the count
-- must be correct regardless of who is asking. Under RLS a user sees only their
-- own rows, which happens to give the right answer today -- but a quota that
-- silently under-counts when a policy changes is worse than no quota. Each
-- function still filters explicitly by NEW.user_id or NEW.thread_id, so it can
-- only ever count one user's rows or one conversation.
--
-- Limits live in a table rather than in the function bodies, so changing one is
-- an UPDATE rather than a rewrite. Read-only to users: no policy grants write
-- access, and none is granted below.

create table public.quotas (
  key   text primary key,
  value bigint not null
);

insert into public.quotas (key, value) values
  ('max_threads_per_user',    5),
  ('max_storage_bytes',       104857600),   -- 100 MiB
  -- Message ROWS, not conversational turns. One exchange that uses the search
  -- tool writes four: the question, the tool call, the tool results, the
  -- answer. So 200 is roughly 50 exchanges.
  ('max_messages_per_thread', 200);

alter table public.quotas enable row level security;

create policy quotas_readable on public.quotas
  for select to authenticated using (true);

grant select on public.quotas to authenticated;

-- ---------------------------------------------------------------------------
-- Thread count
-- ---------------------------------------------------------------------------

create function public.enforce_thread_limit()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value   bigint;
  current_count bigint;
begin
  select value into limit_value from public.quotas where key = 'max_threads_per_user';
  if limit_value is null then
    return new;   -- no quota configured: do not invent one
  end if;

  select count(*) into current_count
    from public.threads
   where user_id = new.user_id;

  if current_count >= limit_value then
    raise exception 'thread_quota_exceeded: % of % chats already exist',
      current_count, limit_value
      using errcode = 'check_violation';
  end if;

  return new;
end;
$$;

create trigger threads_enforce_limit
  before insert on public.threads
  for each row execute function public.enforce_thread_limit();

-- ---------------------------------------------------------------------------
-- Storage
--
-- Counts documents.byte_size -- the original file in Storage. Chunks and
-- embeddings are derived data and deliberately not counted: the user did not
-- choose to create them, and their size follows from chunk settings the user
-- cannot see.
-- ---------------------------------------------------------------------------

create function public.enforce_storage_quota()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value bigint;
  used_bytes  bigint;
begin
  if new.byte_size is null then
    return new;
  end if;

  select value into limit_value from public.quotas where key = 'max_storage_bytes';
  if limit_value is null then
    return new;
  end if;

  -- Excluding NEW.id matters on UPDATE: an in-place re-ingest must be measured
  -- as a replacement, not as an addition on top of the row it replaces.
  select coalesce(sum(byte_size), 0) into used_bytes
    from public.documents
   where user_id = new.user_id
     and id <> new.id;

  if used_bytes + new.byte_size > limit_value then
    raise exception
      'storage_quota_exceeded: % bytes used, % requested, % allowed',
      used_bytes, new.byte_size, limit_value
      using errcode = 'check_violation';
  end if;

  return new;
end;
$$;

-- Fires on UPDATE as well as INSERT: re-ingesting a modified file replaces
-- byte_size in place, which would otherwise be a way to grow past the quota
-- without a single INSERT.
create trigger documents_enforce_quota
  before insert or update of byte_size on public.documents
  for each row execute function public.enforce_storage_quota();

-- ---------------------------------------------------------------------------
-- Chat length
--
-- Chats live in Postgres, not in the Storage bucket, so the file quota does not
-- constrain them at all -- and they are a real consumer. Every search stores the
-- full text of the retrieved passages in a tool_result block so the
-- conversation can be replayed to a stateless API, which makes a single message
-- row up to ~60 KB. Measured: 26 messages already outweighed 40 chunks.
-- ---------------------------------------------------------------------------

create function public.enforce_thread_length()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value   bigint;
  current_count bigint;
begin
  select value into limit_value
    from public.quotas where key = 'max_messages_per_thread';
  if limit_value is null then
    return new;
  end if;

  select count(*) into current_count
    from public.messages
   where thread_id = new.thread_id;

  if current_count >= limit_value then
    raise exception 'thread_length_exceeded: % of % messages already in this chat',
      current_count, limit_value
      using errcode = 'check_violation';
  end if;

  return new;
end;
$$;

create trigger messages_enforce_length
  before insert on public.messages
  for each row execute function public.enforce_thread_length();
