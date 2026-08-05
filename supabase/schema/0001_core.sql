-- TRAG core schema: tables, indexes, RLS, storage, realtime.
--
-- Claude's Messages API is stateless -- no server-side threads, no managed
-- retrieval. Every call resends the full conversation. That is why `messages`
-- exists at all: if this database does not store the transcript, nothing does.
-- For the same reason there is no embeddings endpoint and no managed vector
-- store, so `chunks` and pgvector are the app's own.
--
-- Two rules hold everywhere below and are not restated per object:
--
--   1. Authorization lives in the database. Every table has RLS keyed on
--      auth.uid(); the API is a convenience layer, not a gate. The frontend
--      holds a real user JWT and can reach PostgREST directly, so a check that
--      exists only in a request handler is advice.
--   2. No service_role key anywhere. Where a caller genuinely needs more than
--      its own rows, it gets a SECURITY DEFINER function with a narrow
--      signature -- never a blanket credential.

create extension if not exists pgcrypto;
create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- threads / messages
-- ---------------------------------------------------------------------------

create table public.threads (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  title       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create table public.messages (
  id          uuid primary key default gen_random_uuid(),
  thread_id   uuid not null references public.threads(id) on delete cascade,

  -- Denormalised from threads. RLS runs per row on every query, and a local
  -- column comparison beats a subquery back to threads.
  user_id     uuid not null references auth.users(id) on delete cascade,

  role        text not null check (role in ('user', 'assistant')),

  -- jsonb, not text. Claude messages are arrays of typed blocks (text,
  -- tool_use, tool_result, citations). Flattening to a string loses citation
  -- spans and makes replaying history back into the API lossy.
  content     jsonb not null,

  -- 'max_tokens' means the reply was truncated. Without storing it, a cut-off
  -- answer is indistinguishable from a complete one on reload.
  stop_reason text,
  usage       jsonb,

  created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- documents
--
-- Metadata is typed columns rather than one jsonb blob: the text-to-SQL tool
-- (0005) needs a schema to introspect and an allowlist to scope, and "how many
-- documents from this vendor" is an aggregate, not a search. `metadata` carries
-- the leftovers and any extraction error.
--
-- Two fingerprints make ingestion idempotent: content_hash is what the file IS,
-- config_hash is how it was PROCESSED. Chunks are current only when both match.
-- Both nullable -- an unhashed row reads as stale, which is the right default.
-- ---------------------------------------------------------------------------

create table public.documents (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  filename           text not null,
  mime_type          text,
  byte_size          bigint,
  token_estimate     int,

  -- Supabase Storage holds the bytes. Path: {user_id}/{document_id}/{filename}.
  storage_path       text,

  -- 'analyzing' sits between extracting and chunking: the extracted title is
  -- prepended to every chunk's embedding input, so metadata must exist before
  -- anything is embedded.
  status             text not null default 'pending'
                     constraint documents_status_check check (status in (
                       'pending','extracting','analyzing','chunking',
                       'embedding','ready','failed')),
  error              text,
  chunk_count        int,

  content_hash       text,
  config_hash        text,
  -- The config verbatim, not just its hash, so a stale row can show WHICH knob
  -- changed rather than only that something did.
  extraction_config  jsonb,
  processed_at       timestamptz,

  title              text,
  doc_type           text,
  source_org         text,
  authors            text[],
  -- Day-level, and set only when a full date is printed. published_year is set
  -- whenever the year is known, so a year-only source (most textbooks) is not
  -- recorded as January 1st.
  published_on       date,
  published_year     int,
  language           text,
  topics             text[],
  summary            text,
  metadata           jsonb,

  created_at         timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------

create table public.chunks (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,

  -- Denormalised for the same reason as messages.user_id.
  user_id     uuid not null references auth.users(id) on delete cascade,

  ordinal     int  not null,
  content     text not null,
  token_count int,

  -- The heading path this chunk starts under ("Chapter 3 > Inventory Models").
  -- Derived from Markdown by the chunker, so it costs no LLM call.
  section     text,

  -- 768 is nomic-embed-text's output size. EMBEDDING_DIM in config must match;
  -- a different model means a new migration and re-embedding the corpus.
  embedding   vector(768) not null,

  -- Maintained by Postgres on every write, so no application code can forget to
  -- refresh it and the ingest pipeline has no second write to get wrong.
  --
  -- The 'english' regconfig MUST be spelled out: one-argument to_tsvector()
  -- reads default_text_search_config at runtime, making it STABLE rather than
  -- IMMUTABLE, and a generated column refuses that.
  --
  -- section is folded in so the keyword channel sees heading context exactly as
  -- the vector channel does -- a chunk under "Chapter 7 > Holt's Model" is then
  -- findable by "Holt" even if the body never repeats the name.
  content_tsv tsvector generated always as (
    to_tsvector('english', coalesce(section, '') || ' ' || content)
  ) stored,

  created_at  timestamptz not null default now(),

  unique (document_id, ordinal)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

create index messages_thread_created_idx on public.messages (thread_id, created_at);
create index threads_user_updated_idx    on public.threads  (user_id, updated_at desc);

create index documents_user_created_idx  on public.documents (user_id, created_at desc);
-- Serves the duplicate check on upload: has this user already ingested these
-- exact bytes?
create index documents_user_content_idx  on public.documents (user_id, content_hash);
create index documents_user_doctype_idx  on public.documents (user_id, doc_type);
create index documents_user_org_idx      on public.documents (user_id, source_org);
create index documents_user_year_idx     on public.documents (user_id, published_year);
-- GIN, because topics is an array and the filter is the overlap operator (&&).
create index documents_topics_idx        on public.documents using gin (topics);

-- The HNSW opclass must match the query operator: vector_cosine_ops serves
-- `<=>`. Mismatched, Postgres silently ignores the index and sequentially scans
-- -- correct results, quietly slow.
create index chunks_embedding_idx on public.chunks using hnsw (embedding vector_cosine_ops);

-- GIN is right for tsvector containment (@@). GiST is lossy and slower to
-- search but faster to update -- the wrong trade here, since chunks are written
-- once per ingest and read on every query.
create index chunks_content_tsv_idx on public.chunks using gin (content_tsv);

-- ---------------------------------------------------------------------------
-- Row-Level Security
--
-- `for all` covers select / insert / update / delete in one policy:
--   USING      -> which existing rows are visible (select, update, delete)
--   WITH CHECK -> which rows may be written (insert, update)
--
-- Both are required. USING alone would let a user insert a row owned by someone
-- else -- they just could not read it back afterwards.
--
-- The `anon` role is blocked implicitly: auth.uid() is null for an
-- unauthenticated request, so `auth.uid() = user_id` is never true.
-- ---------------------------------------------------------------------------

alter table public.threads   enable row level security;
alter table public.messages  enable row level security;
alter table public.documents enable row level security;
alter table public.chunks    enable row level security;

create policy threads_owner on public.threads
  for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy messages_owner on public.messages
  for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy documents_owner on public.documents
  for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy chunks_owner on public.chunks
  for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update, delete
  on public.threads, public.messages, public.documents, public.chunks
  to authenticated;

-- ---------------------------------------------------------------------------
-- Storage: private bucket for the original bytes
--
-- The first path segment is the owner's uuid, which is what the policies key
-- on.
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

create policy documents_bucket_select on storage.objects
  for select to authenticated
  using (bucket_id = 'documents'
         and (storage.foldername(name))[1] = auth.uid()::text);

create policy documents_bucket_insert on storage.objects
  for insert to authenticated
  with check (bucket_id = 'documents'
              and (storage.foldername(name))[1] = auth.uid()::text);

create policy documents_bucket_update on storage.objects
  for update to authenticated
  using (bucket_id = 'documents'
         and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'documents'
              and (storage.foldername(name))[1] = auth.uid()::text);

create policy documents_bucket_delete on storage.objects
  for delete to authenticated
  using (bucket_id = 'documents'
         and (storage.foldername(name))[1] = auth.uid()::text);

-- ---------------------------------------------------------------------------
-- Realtime: documents.status transitions push to the UI, which is what lets a
-- worker on a different machine drive the badges in the browser.
-- ---------------------------------------------------------------------------

do $$
begin
  alter publication supabase_realtime add table public.documents;
exception when duplicate_object then
  raise notice 'documents already in supabase_realtime';
end $$;
