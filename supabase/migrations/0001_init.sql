-- Module 1: App Shell
--
-- Claude's Messages API is stateless: there are no server-side threads and no
-- managed retrieval. Every call resends the full conversation. That is why a
-- `messages` table exists here at all -- if we do not store the transcript,
-- nothing does.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

create table if not exists public.threads (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  title       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create table if not exists public.messages (
  id          uuid primary key default gen_random_uuid(),
  thread_id   uuid not null references public.threads(id) on delete cascade,

  -- Denormalised from threads on purpose. RLS runs per row on every query, and
  -- a local column comparison is far cheaper than a subquery back to threads.
  user_id     uuid not null references auth.users(id) on delete cascade,

  role        text not null check (role in ('user', 'assistant')),

  -- jsonb, not text. Claude messages are arrays of typed blocks (text,
  -- document, citations). Flattening to a string loses the citation spans and
  -- makes replaying history back into the API lossy.
  content     jsonb not null,

  -- Populated for assistant rows. stop_reason = 'max_tokens' means the reply
  -- was truncated; without storing it, a cut-off answer is indistinguishable
  -- from a complete one on reload.
  stop_reason text,
  usage       jsonb,

  created_at  timestamptz not null default now()
);

create table if not exists public.documents (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  filename           text not null,
  mime_type          text,
  byte_size          bigint,

  -- Anthropic's Files API owns the bytes; we only keep the handle.
  anthropic_file_id  text not null,

  -- Checked before a document is attached to a request, so an oversized file
  -- fails at upload instead of as a context-window 400 mid-conversation.
  token_estimate     int,

  created_at         timestamptz not null default now(),

  unique (user_id, anthropic_file_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

create index if not exists messages_thread_created_idx
  on public.messages (thread_id, created_at);

create index if not exists threads_user_updated_idx
  on public.threads (user_id, updated_at desc);

create index if not exists documents_user_created_idx
  on public.documents (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Row-Level Security
--
-- `for all` covers select / insert / update / delete in one policy:
--   USING      -> which existing rows are visible (select, update, delete)
--   WITH CHECK -> which rows may be written (insert, update)
--
-- Both are required. USING alone would let a user insert a row owned by
-- someone else -- they just could not read it back afterwards.
--
-- The `anon` role is blocked implicitly: auth.uid() is null for an
-- unauthenticated request, so `auth.uid() = user_id` is never true.
-- ---------------------------------------------------------------------------

alter table public.threads   enable row level security;
alter table public.messages  enable row level security;
alter table public.documents enable row level security;

drop policy if exists threads_owner   on public.threads;
drop policy if exists messages_owner  on public.messages;
drop policy if exists documents_owner on public.documents;

create policy threads_owner on public.threads
  for all
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy messages_owner on public.messages
  for all
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy documents_owner on public.documents
  for all
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

grant select, insert, update, delete
  on public.threads, public.messages, public.documents
  to authenticated;

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. RLS is on for all three
--   select relname, relrowsecurity
--     from pg_class
--    where relname in ('threads','messages','documents');
--   -- expect relrowsecurity = true for each
--
--   -- 2. Policies exist
--   select tablename, policyname, cmd
--     from pg_policies
--    where schemaname = 'public';
--
--   -- 3. Isolation: as an unauthenticated session, this must return 0 rows
--   set role anon;
--   select count(*) from public.threads;
--   reset role;
-- ---------------------------------------------------------------------------
