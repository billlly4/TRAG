# Consolidated schema

The database as it should be built today, in five files, organised by concern
rather than by the order things were discovered.


| file | contents |
|---|---|
| `0001_core.sql` | extensions, `threads`, `messages`, `documents`, `chunks`, indexes, RLS, storage bucket, realtime |
| `0002_retrieval.sql` | `match_chunks`, `match_chunks_keyword`, `count_documents_matching` |
| `0003_quotas.sql` | `quotas` table and the three enforcement triggers |
| `0004_ingest_queue.sql` | `ingest_workers`, `ingest_jobs`, the five worker functions |
| `0005_metadata_sql.sql` | `documents_queryable`, the `rag_readonly` role, `query_documents` |

Apply in order. The dependencies are real: `0002` needs `chunks.content_tsv`
from `0001`, and `0005` revokes on `ingest_jobs` from `0004`.

## What changed from `migrations/`

Nothing semantic. The final state is byte-for-byte equivalent in behaviour —
same signatures, same grants, same policies, same defaults. What went away is
the archaeology:

- `match_chunks` was created once, then dropped and recreated twice as
  parameters were added. Here it is created once, with all nine.
- `documents` was built across four migrations — a column added in one, dropped
  in the next (`anthropic_file_id`), metadata bolted on later. Here it is one
  `create table`.
- `chunks.content_tsv` was an `alter table` after the fact; it is now declared
  inline where its constraints are visible next to the columns it reads.
- The `do $$ ... drop constraint by definition ... $$` block that renamed an
  inline status check is gone — the constraint is simply named
  `documents_status_check` from the start.
- `claim_ingest_job` and `fail_ingest_job` were each written twice, the second
  time to add backoff. Only the final version is here.

One comment was wrong and is corrected: `migrations/0011` line 213 said *"The
view is security_invoker"*, contradicting its own line 47 (*"DEFINER-rights
view, NOT security_invoker"*). The code was always definer-rights; only the
comment was stale.

## After applying

`0004` needs one manual step or the queue never drains and every upload sits at
`pending` — register the worker account. The instructions are at the foot of that
file.

Each file ends with a commented verification block meant to be pasted into the
SQL editor. They are worth running: a file that applied cleanly and a schema
that works are different claims. The ones that matter most —

- `select proname, pg_get_function_identity_arguments(oid) from pg_proc where
  proname in ('match_chunks','match_chunks_keyword')` — **exactly one row each**.
  Two means an overload survived, and PostgREST fails every search at once with
  "could not choose the best candidate function".
- `select proname, pg_get_userbyid(proowner) from pg_proc where proname =
  'query_documents'` — must read `rag_readonly`. If it reads `postgres`, the
  text-to-SQL privilege drop is gone and the model's SQL has superuser reach.
- `select proconfig from pg_proc where proname = 'match_chunks'` — must contain
  `hnsw.iterative_scan=relaxed_order` on pgvector 0.8+. Missing, filtered
  searches quietly return fewer rows than asked for.

The reasoning behind the numbers and the failures that produced them is in
[`findings.md`](../../findings.md).
