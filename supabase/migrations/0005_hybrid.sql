-- Module 6: Hybrid Search
--
-- Adds a keyword retrieval channel alongside the vector one. The two fail in
-- close to opposite ways: an embedding is a lossy summary, so a query for a
-- rare literal token -- a part number, "Holt's model", "13,000" -- can miss the
-- chunk that literally contains it, because the surrounding semantics dominate
-- the vector. Full-text search is exact where embeddings are fuzzy.
--
-- Fusion of the two rankings happens in Python, not here. Doing it in SQL would
-- hardcode the pair; CLAUDE.md requires fusion over a LIST of ranked lists so a
-- third channel can be added without reopening the fusion code.

-- ---------------------------------------------------------------------------
-- 1. Searchable text vector
--
-- Generated and stored, so it is maintained by Postgres on every insert and
-- update -- there is no application code that can forget to refresh it, and no
-- second write for the ingest pipeline to get wrong.
-- ---------------------------------------------------------------------------

alter table public.chunks
  add column content_tsv tsvector
  generated always as (
    -- The 'english' regconfig MUST be spelled out. The one-argument
    -- to_tsvector() reads default_text_search_config at runtime, which makes it
    -- STABLE rather than IMMUTABLE, and a generated column refuses it.
    --
    -- section is folded in so the keyword channel sees heading context, exactly
    -- as chunking.embed_text prepends it for the vector channel. A chunk under
    -- "Chapter 7 > Holt's Model" is then findable by "Holt" even if the body
    -- never repeats the name.
    to_tsvector('english', coalesce(section, '') || ' ' || content)
  ) stored;

-- GIN is the right index for tsvector containment (@@). GiST would also work
-- but is lossy and slower to search, faster to update -- the wrong trade here,
-- since chunks are written once per ingest and read on every query.
create index chunks_content_tsv_idx on public.chunks using gin (content_tsv);

-- ---------------------------------------------------------------------------
-- 2. Keyword retrieval
--
-- Deliberately mirrors match_chunks: same five metadata filters, same
-- security invoker, same return columns. Two channels that disagree about
-- filtering or RLS would be a silent correctness hole -- a filter that only
-- narrowed the vector channel would quietly widen results the moment hybrid
-- search is switched on.
-- ---------------------------------------------------------------------------

create or replace function public.match_chunks_keyword(
  query_text         text,
  match_count        int    default 5,
  filter_doc_types   text[] default null,
  filter_source_orgs text[] default null,
  filter_topics      text[] default null,
  filter_year_min    int    default null,
  filter_year_max    int    default null
)
returns table (id uuid, document_id uuid, filename text, ordinal int,
               content text, similarity float, section text,
               doc_type text, source_org text, published_year int,
               keyword_rank float)
language sql stable
security invoker
set search_path = public
as $$
  select c.id, c.document_id, d.filename, c.ordinal, c.content,
         -- Keyword search computes no cosine. The column exists so both RPCs
         -- return an identical shape and one Hit constructor serves both;
         -- 0.0 is "no vector evidence", not "dissimilar".
         0.0::float as similarity,
         c.section, d.doc_type, d.source_org, d.published_year,
         ts_rank_cd(c.content_tsv,
                    websearch_to_tsquery('english', query_text))::float
           as keyword_rank
  from public.chunks c
  join public.documents d on d.id = c.document_id
  -- websearch_to_tsquery accepts what users actually type: quoted phrases,
  -- OR, and -excluded terms. On stopwords-only input it yields an EMPTY
  -- tsquery, and `@@` against an empty tsquery is false for every row -- so a
  -- query like "the of and" returns nothing rather than erroring.
  where c.content_tsv @@ websearch_to_tsquery('english', query_text)
    and (filter_doc_types   is null or d.doc_type   = any(filter_doc_types))
    and (filter_source_orgs is null or d.source_org = any(filter_source_orgs))
    and (filter_topics      is null or d.topics    && filter_topics)
    and (filter_year_min    is null or d.published_year >= filter_year_min)
    and (filter_year_max    is null or d.published_year <= filter_year_max)
  order by keyword_rank desc
  limit match_count;
$$;

grant execute on function public.match_chunks_keyword(
  text, int, text[], text[], text[], int, int
) to authenticated;

-- No RLS changes: chunks_owner and documents_owner are row-level, the new
-- column is covered by them, and this function is security invoker so the
-- policies apply inside it exactly as they do for match_chunks.

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. Column exists and is populated for every chunk
--   select count(*) as chunks,
--          count(content_tsv) as with_tsv,
--          count(*) filter (where content_tsv = '') as empty_tsv
--     from public.chunks;
--   -- expect chunks = with_tsv, empty_tsv = 0
--
--   -- 2. The GIN index exists
--   select indexname from pg_indexes
--    where tablename = 'chunks' and indexname = 'chunks_content_tsv_idx';
--
--   -- 3. A term you know is in the corpus returns rows, nonsense returns none
--   select filename, ordinal, round(keyword_rank::numeric, 4) as rank
--     from public.match_chunks_keyword('forecasting', 5);
--   select count(*) from public.match_chunks_keyword('zzzqqqxyz', 5);
--   -- expect rows, then 0
--
--   -- 4. Stopwords-only input is empty, not an error
--   select count(*) from public.match_chunks_keyword('the of and', 5);
--   -- expect 0 rows, no exception
--
--   -- 5. One signature only, and the grant landed
--   select pg_get_function_identity_arguments(oid)
--     from pg_proc where proname = 'match_chunks_keyword';
--   select has_function_privilege('authenticated',
--     'public.match_chunks_keyword(text,int,text[],text[],text[],int,int)',
--     'execute');
-- ---------------------------------------------------------------------------
