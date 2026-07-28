-- Module 2: BYO Retrieval
--
-- Anthropic has no embeddings endpoint and no managed retrieval, so this
-- migration is where the app starts owning it: chunks live in pgvector, the
-- document bytes move from Anthropic's Files API to Supabase Storage, and
-- vector search is exposed through a database function because PostgREST
-- cannot express `order by embedding <=> $1`.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- documents: ingestion pipeline state
-- ---------------------------------------------------------------------------

alter table public.documents
  add column storage_path text,
  add column status text not null default 'pending'
    check (status in ('pending','extracting','chunking','embedding','ready','failed')),
  add column error text,
  add column chunk_count int;

-- The hard cut: Anthropic no longer holds the bytes, Supabase Storage does.
alter table public.documents drop constraint if exists documents_user_id_anthropic_file_id_key;
alter table public.documents drop column anthropic_file_id;

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------

create table public.chunks (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,

  -- Denormalised from documents on purpose, same as messages.user_id in 0001:
  -- RLS runs per row, and a local column comparison beats a subquery back to
  -- documents on every chunk of every search.
  user_id     uuid not null references auth.users(id) on delete cascade,

  ordinal     int  not null,
  content     text not null,
  token_count int,

  -- 768 is nomic-embed-text's output size. EMBEDDING_DIM in config must match;
  -- a different model means a new migration and re-embedding the corpus.
  embedding   vector(768) not null,

  created_at  timestamptz not null default now(),

  unique (document_id, ordinal)
);

-- HNSW opclass must match the query operator: vector_cosine_ops serves `<=>`.
-- Mismatched, Postgres silently ignores the index and sequentially scans --
-- correct results, quietly slow.
create index chunks_embedding_idx
  on public.chunks using hnsw (embedding vector_cosine_ops);

alter table public.chunks enable row level security;

drop policy if exists chunks_owner on public.chunks;

create policy chunks_owner on public.chunks
  for all
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

grant select, insert, update, delete on public.chunks to authenticated;

-- ---------------------------------------------------------------------------
-- Retrieval RPC
--
-- security invoker is load-bearing: the function runs as the caller, so RLS on
-- chunks still applies inside it. security definer would run as the function
-- owner and bypass RLS entirely -- exactly the "work around a policy" move
-- AGENTS.md forbids.
-- ---------------------------------------------------------------------------

create or replace function public.match_chunks(
  query_embedding vector(768),
  match_count int default 5,
  min_similarity float default 0.0
)
returns table (id uuid, document_id uuid, filename text, ordinal int,
               content text, similarity float)
language sql stable
security invoker
set search_path = public
as $$
  -- <=> is cosine DISTANCE (0 = identical): similarity is 1 - distance and
  -- ordering is ascending. Reversed, this returns the least relevant chunks.
  select c.id, c.document_id, d.filename, c.ordinal, c.content,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  join public.documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_chunks(vector, int, float) to authenticated;

-- ---------------------------------------------------------------------------
-- Storage: private bucket for the original bytes
--
-- Path convention: {user_id}/{document_id}/{filename}. The first folder is the
-- owner's uuid, which is what the policies key on.
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

drop policy if exists documents_bucket_select on storage.objects;
drop policy if exists documents_bucket_insert on storage.objects;
drop policy if exists documents_bucket_update on storage.objects;
drop policy if exists documents_bucket_delete on storage.objects;

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
-- Realtime: documents.status transitions push to the UI
-- ---------------------------------------------------------------------------

alter publication supabase_realtime add table public.documents;

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. RLS is on for chunks
--   select relname, relrowsecurity
--     from pg_class
--    where relname = 'chunks';
--   -- expect relrowsecurity = true
--
--   -- 2. Policies exist (chunks_owner plus four documents_bucket_*)
--   select tablename, policyname, cmd
--     from pg_policies
--    where policyname like '%chunks%' or policyname like 'documents_bucket%';
--
--   -- 3. Isolation THROUGH THE RPC, not just the table -- the function is the
--   --    new way data leaves the database. Insert chunks as two different
--   --    users (via the API, so RLS applies), then as each user confirm both:
--   --      select count(*) from public.chunks;
--   --      select * from public.match_chunks((select embedding from public.chunks limit 1), 10, 0.0);
--   --    Neither may return the other user's rows.
--
--   -- 4. anthropic_file_id is gone
--   select column_name from information_schema.columns
--    where table_name = 'documents';
-- ---------------------------------------------------------------------------
