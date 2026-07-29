-- Module 4: Metadata Extraction
--
-- Retrieval so far ranks the whole corpus by cosine similarity and takes the
-- top k. This migration gives it something to narrow WITH: typed metadata
-- columns on documents, a section path on chunks, and a match_chunks that
-- applies filters in the WHERE clause -- before the vector ordering, not after
-- it. Filtering after the top-k is the tempting version and it silently
-- destroys recall: it can only ever remove rows the ranking already chose,
-- so a filter that excludes the top 5 returns nothing instead of the next 5.

-- ---------------------------------------------------------------------------
-- 1. Metadata columns
--
-- Typed columns rather than one jsonb blob because Module 7's text-to-SQL
-- needs a schema to introspect and an allowlist to scope, and because
-- "how many documents from this vendor" is an aggregate, not a search. The
-- jsonb column carries the leftovers (and any extraction error).
--
-- All nullable: legacy rows have no metadata and read as unfiltered. No
-- backfill needed -- every row is already stale from the config_hash change,
-- so one "Re-process stale" populates them.
-- ---------------------------------------------------------------------------

alter table public.documents
  add column title           text,
  add column doc_type        text,
  add column source_org      text,
  add column authors         text[],

  -- published_on is day-level and only set when a full date is printed;
  -- published_year is set whenever the year is known. Two columns so a
  -- year-only source (most textbooks) is not recorded as January 1st.
  add column published_on    date,
  add column published_year  int,

  add column language        text,
  add column topics          text[],
  add column summary         text,
  add column metadata        jsonb;

-- The heading path the chunk starts under ("Chapter 3 > Inventory Models").
-- Derived by the chunker from Markdown, so it costs no LLM call.
alter table public.chunks
  add column section text;

create index documents_user_doctype_idx on public.documents (user_id, doc_type);
create index documents_user_org_idx     on public.documents (user_id, source_org);
create index documents_user_year_idx    on public.documents (user_id, published_year);

-- GIN, because topics is an array and the filter is the overlap operator (&&).
create index documents_topics_idx on public.documents using gin (topics);

-- ---------------------------------------------------------------------------
-- 2. New pipeline status: 'analyzing'
--
-- Metadata extraction runs BETWEEN extraction and chunking, because the
-- extracted title is prepended to each chunk's embedding input -- so it has
-- to exist before anything is embedded.
--
-- 0002 declared the status check inline, so its name was generated. Find and
-- drop it by definition rather than guessing the name.
-- ---------------------------------------------------------------------------

do $$
declare
  cname text;
begin
  select conname into cname
    from pg_constraint
   where conrelid = 'public.documents'::regclass
     and contype = 'c'
     and pg_get_constraintdef(oid) ilike '%status%';

  if cname is not null then
    execute format('alter table public.documents drop constraint %I', cname);
  end if;
end $$;

alter table public.documents
  add constraint documents_status_check
  check (status in ('pending','extracting','analyzing','chunking',
                    'embedding','ready','failed'));

-- ---------------------------------------------------------------------------
-- 3. match_chunks gains metadata filters
--
-- DROP first, do not CREATE OR REPLACE. Adding parameters -- even defaulted
-- ones -- creates an OVERLOAD rather than replacing the function, and the
-- existing three-argument call then matches both signatures and fails as
-- ambiguous. Dropping also drops the grant, so it is re-issued below.
-- ---------------------------------------------------------------------------

drop function if exists public.match_chunks(vector, int, float);

create function public.match_chunks(
  query_embedding    vector(768),
  match_count        int   default 5,
  min_similarity     float default 0.0,
  filter_doc_types   text[] default null,
  filter_source_orgs text[] default null,
  filter_topics      text[] default null,
  filter_year_min    int    default null,
  filter_year_max    int    default null
)
returns table (id uuid, document_id uuid, filename text, ordinal int,
               content text, similarity float, section text,
               doc_type text, source_org text, published_year int)
language sql stable
security invoker
set search_path = public
as $$
  select c.id, c.document_id, d.filename, c.ordinal, c.content,
         1 - (c.embedding <=> query_embedding) as similarity,
         c.section, d.doc_type, d.source_org, d.published_year
  from public.chunks c
  join public.documents d on d.id = c.document_id
  -- Every filter is null-tolerant: a null argument means "do not filter on
  -- this", so the unfiltered call behaves exactly as it did before.
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
    and (filter_doc_types   is null or d.doc_type   = any(filter_doc_types))
    and (filter_source_orgs is null or d.source_org = any(filter_source_orgs))
    and (filter_topics      is null or d.topics    && filter_topics)
    and (filter_year_min    is null or d.published_year >= filter_year_min)
    and (filter_year_max    is null or d.published_year <= filter_year_max)
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_chunks(
  vector, int, float, text[], text[], text[], int, int
) to authenticated;

-- HNSW walks the index in similarity order and stops once it has match_count
-- candidates. With a selective filter, most of what it visits is discarded and
-- it can return fewer rows than asked for. Iterative scan (pgvector 0.8+) makes
-- it keep going until the limit is genuinely met.
--
-- Attached separately and tolerantly: on pgvector < 0.8 the parameter does not
-- exist and would make the whole CREATE FUNCTION fail.
do $$
begin
  execute 'alter function public.match_chunks(vector, int, float, text[], '
          'text[], text[], int, int) set hnsw.iterative_scan = ''relaxed_order''';
  raise notice 'hnsw.iterative_scan enabled';
exception when others then
  raise notice 'hnsw.iterative_scan unavailable (pgvector < 0.8): filtered '
               'searches may return fewer than match_count rows';
end $$;

-- No RLS changes. chunks_owner and documents_owner are row-level, so the new
-- columns are already covered, and match_chunks stays `security invoker` so
-- the policies still apply inside it.

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. Columns exist (expect 10 document columns + chunks.section)
--   select table_name, column_name, data_type
--     from information_schema.columns
--    where (table_name = 'documents'
--           and column_name in ('title','doc_type','source_org','authors',
--                               'published_on','published_year','language',
--                               'topics','summary','metadata'))
--       or (table_name = 'chunks' and column_name = 'section');
--
--   -- 2. The status constraint accepts 'analyzing'
--   select pg_get_constraintdef(oid)
--     from pg_constraint
--    where conname = 'documents_status_check';
--
--   -- 3. Exactly ONE match_chunks signature (an overload would be ambiguous)
--   select pronargs, pg_get_function_identity_arguments(oid)
--     from pg_proc where proname = 'match_chunks';
--   -- expect exactly 1 row, with 8 arguments
--
--   -- 4. The grant survived the drop
--   select has_function_privilege(
--     'authenticated',
--     'public.match_chunks(vector,int,float,text[],text[],text[],int,int)',
--     'execute');
--   -- expect true
--
--   -- 5. Which iterative-scan path this database took
--   select extversion from pg_extension where extname = 'vector';
--   -- 0.8.0+ means the notice above said "enabled"
-- ---------------------------------------------------------------------------