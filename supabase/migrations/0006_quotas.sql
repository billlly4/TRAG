-- Module: per-user quotas
--
-- Enforced in the database, not only in FastAPI. The frontend holds a real user
-- JWT and the RLS policies let it write to `threads` and `documents` through
-- PostgREST directly, so a limit checked only in a request handler is advice,
-- not a limit. Same reasoning as every other rule in this schema: authorization
-- and invariants belong where the data is.
--
-- The API checks the same limits first, purely so the user gets a readable
-- message instead of a raised Postgres exception. The trigger is the actual
-- enforcement; the handler is the manners.

-- ---------------------------------------------------------------------------
-- Configuration
--
-- Kept in a table rather than hardcoded in the functions so a limit can be
-- changed without rewriting a trigger. Read-only to users: no RLS policy grants
-- them write access, and none is granted below.
-- ---------------------------------------------------------------------------

create table if not exists public.quotas (
  key   text primary key,
  value bigint not null
);

insert into public.quotas (key, value) values
  ('max_threads_per_user', 5),
  ('max_storage_bytes', 104857600)   -- 100 MiB
on conflict (key) do nothing;

alter table public.quotas enable row level security;

drop policy if exists quotas_readable on public.quotas;
create policy quotas_readable on public.quotas
  for select to authenticated using (true);

grant select on public.quotas to authenticated;

-- ---------------------------------------------------------------------------
-- Thread limit
--
-- SECURITY DEFINER because the count must be correct regardless of who is
-- asking. Under RLS a user sees only their own threads, which happens to give
-- the right answer today -- but a quota that silently under-counts when a
-- policy changes is worse than no quota. The function still filters explicitly
-- by NEW.user_id, so it can only ever count one user's rows.
-- ---------------------------------------------------------------------------

create or replace function public.enforce_thread_limit()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value bigint;
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

drop trigger if exists threads_enforce_limit on public.threads;
create trigger threads_enforce_limit
  before insert on public.threads
  for each row execute function public.enforce_thread_limit();

-- ---------------------------------------------------------------------------
-- Storage quota
--
-- Counts documents.byte_size, which is the size of the original file in
-- Storage. Chunks and embeddings are derived data and deliberately not counted:
-- the user did not choose to create them, and their size is a consequence of
-- chunk settings the user cannot see.
--
-- Fires on UPDATE as well as INSERT because re-ingesting a modified file
-- replaces byte_size in place (Module 3), which would otherwise be a way to
-- grow past the quota without a single INSERT.
-- ---------------------------------------------------------------------------

create or replace function public.enforce_storage_quota()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value bigint;
  used_bytes bigint;
begin
  if new.byte_size is null then
    return new;
  end if;

  select value into limit_value from public.quotas where key = 'max_storage_bytes';
  if limit_value is null then
    return new;
  end if;

  -- Excluding NEW.id matters on UPDATE: an in-place re-ingest must be measured
  -- as a replacement, not as an addition on top of the row it is replacing.
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

drop trigger if exists documents_enforce_quota on public.documents;
create trigger documents_enforce_quota
  before insert or update of byte_size on public.documents
  for each row execute function public.enforce_storage_quota();

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. Quotas are readable
--   select * from public.quotas;
--
--   -- 2. Triggers are attached
--   select tgname, tgrelid::regclass
--     from pg_trigger
--    where tgname in ('threads_enforce_limit', 'documents_enforce_quota');
--
--   -- 3. Current usage per user (admin view; run as the table owner)
--   select user_id,
--          (select count(*) from public.threads t where t.user_id = d.user_id) as threads,
--          pg_size_pretty(sum(byte_size)) as storage
--     from public.documents d group by user_id;
--
--   -- 4. The limit actually bites. As a user with 5 threads, this must fail
--   --    with 'thread_quota_exceeded':
--   --      insert into public.threads (user_id, title)
--   --      values (auth.uid(), 'sixth');
--
--   -- To change a limit later, update the row -- no function rewrite needed:
--   --   update public.quotas set value = 10 where key = 'max_threads_per_user';
-- ---------------------------------------------------------------------------
