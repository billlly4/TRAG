-- Module 3: Record Manager
--
-- Two fingerprints make ingestion idempotent: content_hash says what the file
-- IS, config_hash says how it was PROCESSED. Chunks are current only if both
-- match the incoming upload / current settings. Nullable on purpose: legacy
-- rows have no hashes and therefore read as stale until re-processed.

alter table public.documents
  add column content_hash text,
  add column config_hash  text,

  -- The config verbatim, not just its hash, so a stale row can show WHY it is
  -- stale (which knob changed), not merely that it is.
  add column extraction_config jsonb,

  add column processed_at timestamptz;

-- Serves the duplicate-check on upload: "has this user already ingested these
-- exact bytes?"
create index documents_user_content_idx
  on public.documents (user_id, content_hash);

-- No RLS changes: policies are row-level, so new columns are covered by the
-- existing documents_owner policy from 0001.

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. Columns exist
--   select column_name, data_type
--     from information_schema.columns
--    where table_name = 'documents'
--      and column_name in ('content_hash','config_hash','extraction_config','processed_at');
--   -- expect 4 rows
--
--   -- 2. Legacy rows are unhashed (stale by definition)
--   select count(*) filter (where content_hash is null) as unhashed
--     from public.documents;
-- ---------------------------------------------------------------------------
