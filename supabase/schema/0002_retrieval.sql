-- Retrieval: two search channels and one exact counter.
--
-- These are functions rather than PostgREST queries because PostgREST cannot
-- express `order by embedding <=> $1`.
--
-- ALL THREE ARE `security invoker`, and that is load-bearing. The function runs
-- as the caller, so RLS on chunks and documents still applies inside it.
-- `security definer` would run as the owner and bypass RLS entirely.
--
-- THE TWO CHANNELS MUST MIRROR EACH OTHER -- same filters, same security mode,
-- same return columns. They fail in close to opposite ways (an embedding is a
-- lossy summary and misses rare literal tokens; full-text search is exact but
-- misses paraphrase), which is the whole reason both exist. Two channels that
-- disagreed about filtering would be a silent correctness hole: a filter that
-- only narrowed one of them would quietly widen results the moment fusion
-- combined the two.
--
-- FUSION IS NOT HERE. Reciprocal-rank fusion runs in Python over a LIST of
-- ranked lists, so a third channel can be added without reopening it. Doing it
-- in SQL would hardcode the pair.
--
-- EVERY FILTER IS NULL-TOLERANT: a null argument means "do not filter on this",
-- so an unfiltered call behaves as though the parameter did not exist.
--
-- FILTERS GO IN THE WHERE CLAUSE, NEVER IN THE CALLER. Filtering the top-k
-- afterwards can only discard rows the ranking already picked, so a filter that
-- excludes the top 5 returns nothing instead of the next 5. Scoping to a single
-- document after the fact is the worst case: for a corpus of related chapters
-- the winners usually come from other files, so it returns nothing almost
-- every time.

-- ---------------------------------------------------------------------------
-- 1. Vector channel
-- ---------------------------------------------------------------------------

create function public.match_chunks(
  query_embedding     vector(768),
  match_count         int    default 5,
  min_similarity      float  default 0.0,
  filter_doc_types    text[] default null,
  filter_source_orgs  text[] default null,
  filter_topics       text[] default null,
  filter_year_min     int    default null,
  filter_year_max     int    default null,
  filter_document_ids uuid[] default null
)
returns table (id uuid, document_id uuid, filename text, ordinal int,
               content text, similarity float, section text,
               doc_type text, source_org text, published_year int)
language sql stable
security invoker
set search_path = public
as $$
  -- <=> is cosine DISTANCE (0 = identical): similarity is 1 - distance and the
  -- ordering is ASCENDING. Reversed, this returns the least relevant chunks.
  select c.id, c.document_id, d.filename, c.ordinal, c.content,
         1 - (c.embedding <=> query_embedding) as similarity,
         c.section, d.doc_type, d.source_org, d.published_year
  from public.chunks c
  join public.documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
    and (filter_doc_types    is null or d.doc_type    = any(filter_doc_types))
    and (filter_source_orgs  is null or d.source_org  = any(filter_source_orgs))
    and (filter_topics       is null or d.topics     && filter_topics)
    and (filter_year_min     is null or d.published_year >= filter_year_min)
    and (filter_year_max     is null or d.published_year <= filter_year_max)
    and (filter_document_ids is null or c.document_id = any(filter_document_ids))
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_chunks(
  vector, int, float, text[], text[], text[], int, int, uuid[]
) to authenticated;

-- HNSW walks the index in similarity order and stops once it has match_count
-- candidates. With a selective filter -- and a single-document filter is about
-- as selective as it gets -- most of what it visits is discarded and it returns
-- fewer rows than asked for. Correct results, quietly short. Iterative scan
-- (pgvector 0.8+) makes it keep going until the limit is genuinely met.
--
-- IF THIS FUNCTION IS EVER DROPPED AND RECREATED, THIS SETTING GOES WITH IT.
-- So does the grant above. The grant fails loudly (every search 404s); this
-- fails silently.
--
-- Attached tolerantly: on pgvector < 0.8 the parameter does not exist and a
-- bare ALTER would fail the whole migration.
do $$
begin
  execute 'alter function public.match_chunks(vector, int, float, text[], '
          'text[], text[], int, int, uuid[]) '
          'set hnsw.iterative_scan = ''relaxed_order''';
  raise notice 'hnsw.iterative_scan enabled';
exception when others then
  raise notice 'hnsw.iterative_scan unavailable (pgvector < 0.8): filtered '
               'searches may return fewer than match_count rows';
end $$;

-- ---------------------------------------------------------------------------
-- 2. Keyword channel
-- ---------------------------------------------------------------------------

create function public.match_chunks_keyword(
  query_text          text,
  match_count         int    default 5,
  filter_doc_types    text[] default null,
  filter_source_orgs  text[] default null,
  filter_topics       text[] default null,
  filter_year_min     int    default null,
  filter_year_max     int    default null,
  filter_document_ids uuid[] default null
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
         -- 0.0 means "no vector evidence", not "dissimilar".
         0.0::float as similarity,
         c.section, d.doc_type, d.source_org, d.published_year,
         ts_rank_cd(c.content_tsv,
                    websearch_to_tsquery('english', query_text))::float
           as keyword_rank
  from public.chunks c
  join public.documents d on d.id = c.document_id
  -- websearch_to_tsquery accepts what users actually type: quoted phrases, OR,
  -- and -excluded terms. On stopwords-only input it yields an EMPTY tsquery,
  -- and `@@` against an empty tsquery is false for every row -- so "the of and"
  -- returns nothing rather than erroring.
  where c.content_tsv @@ websearch_to_tsquery('english', query_text)
    and (filter_doc_types    is null or d.doc_type    = any(filter_doc_types))
    and (filter_source_orgs  is null or d.source_org  = any(filter_source_orgs))
    and (filter_topics       is null or d.topics     && filter_topics)
    and (filter_year_min     is null or d.published_year >= filter_year_min)
    and (filter_year_max     is null or d.published_year <= filter_year_max)
    and (filter_document_ids is null or c.document_id = any(filter_document_ids))
  order by keyword_rank desc
  limit match_count;
$$;

grant execute on function public.match_chunks_keyword(
  text, int, text[], text[], text[], int, int, uuid[]
) to authenticated;

-- ---------------------------------------------------------------------------
-- 3. Exact content counts
--
-- "How many of my files mention X" has no correct answer from the other two:
-- search returns the top k passages, so counting what comes back measures k,
-- not the corpus -- right on a 7-document corpus, silently wrong on 200. And
-- the metadata tool (0005) cannot see content at all, deliberately.
--
-- THE RETURN TYPE IS THE BOUNDARY. There is no content column, so this cannot
-- carry passage text no matter what is asked of it. The corpus-dumping concern
-- that keeps `chunks` out of the text-to-SQL view does not apply to a function
-- that can only ever emit a document id, a filename and an integer.
--
-- Costs one GIN index probe -- content_tsv and its index already exist, so this
-- adds no storage and no ingest work.
-- ---------------------------------------------------------------------------

create function public.count_documents_matching(
  search_term text,
  max_rows    int default 200
)
returns table (document_id uuid, filename text, chunk_matches bigint)
language sql stable
security invoker
set search_path = public
as $$
  select c.document_id,
         d.filename,
         count(*)::bigint as chunk_matches
  from public.chunks c
  join public.documents d on d.id = c.document_id
  -- The same matching the keyword channel uses, so a term that counts here is a
  -- term that would retrieve there.
  where c.content_tsv @@ websearch_to_tsquery('english', search_term)
  group by c.document_id, d.filename
  -- Most-mentioned first, so a truncated result keeps the strongest matches.
  order by count(*) desc, d.filename
  limit max_rows;
$$;

grant execute on function public.count_documents_matching(text, int)
  to authenticated;
