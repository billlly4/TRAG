-- Text-to-SQL over document metadata.
--
-- Vector search retrieves passages; it cannot count rows. "How many documents
-- from each source", "what did I upload last year", "which document types are
-- in my corpus" are `count`/`group by` over the metadata columns -- which is
-- why those are typed columns rather than one jsonb blob.
--
-- THE TRUST BOUNDARY IS POSTGRES GRANTS, NOT STRING PARSING.
--
-- The model writes the SQL. Validating model-written SQL with regexes is the
-- tempting version and it leaks: every allowlist of forbidden keywords has a
-- form it does not catch. So the SQL runs under a role granted SELECT on
-- exactly one view and nothing else -- `delete from documents` fails because
-- rag_readonly has no such grant, not because a pattern matched. The structural
-- checks in the function are defence in depth and are ALLOWED TO BE IMPERFECT,
-- precisely because they are not what holds the line.
--
-- This is `security definer` used to DROP privilege -- the same shape as the
-- ingest-worker functions, not the service_role-style escalation that would
-- move authorization out of the database.
--
-- KNOWN AND ACCEPTED: the system catalogs stay readable. `select * from pg_proc`
-- through this tool works, because PostgreSQL grants pg_catalog to PUBLIC by
-- design and revoking it breaks the server. The model can enumerate table and
-- role names; it cannot read a single row of anyone's data that way, since
-- catalogs describe the schema and not its contents. Blocking it would mean
-- pattern-matching on 'pg_', which is the leaky approach this design exists to
-- avoid. Verified, not assumed: a catalog query returns metadata, while
-- `select ... from documents` returns 42501.

-- ---------------------------------------------------------------------------
-- 1. The allowlist, as a view
--
-- Adding a column to `documents` does NOT surface it here. That is the point --
-- but it means this list needs revisiting whenever metadata grows, or new
-- columns are silently unqueryable.
--
-- Excluded deliberately:
--   user_id                      nothing needs it once the view is self-scoped
--   storage_path, content_hash,
--   config_hash, token_estimate  internals; leak processing details
--   error, metadata              free text that can carry extraction fragments
--   summary                      long text that would bloat every result row;
--                                content questions belong to search_documents
--
-- DEFINER-RIGHTS VIEW (the default), NOT security_invoker.
--
-- The first version used `security_invoker = on`, wanting RLS as a second scope
-- on top of the where-clause. That does not work here, and the failure is
-- instructive: an invoker-rights view requires the CALLER to hold privileges on
-- the base table, so reading it as rag_readonly gave
--   42501: permission denied for table documents
-- The only fix is `grant select on documents to rag_readonly` -- which would let
-- the model read documents directly and select the very columns this view
-- exists to withhold. The belt-and-braces would have cost the allowlist.
--
-- Definer rights instead: the view runs as its owner, so rag_readonly needs no
-- base-table grant at all. The scoping is still layered, just differently --
--   1. rag_readonly cannot reach ANY base table, only this view
--   2. the view exposes only the columns below
--   3. the where-clause scopes every read to the caller's own rows
-- and (1) enforces the column allowlist, which RLS never would have.
-- ---------------------------------------------------------------------------

create view public.documents_queryable as
select
  id,
  filename,
  mime_type,
  title,
  doc_type,
  source_org,
  authors,
  published_on,
  published_year,
  language,
  topics,
  byte_size,
  chunk_count,
  status,
  created_at
from public.documents
where user_id = auth.uid();

comment on view public.documents_queryable is
  'Allowlisted, self-scoped projection of documents for the text-to-SQL tool. '
  'Definer-rights: the where-clause is what scopes rows to the caller, so it '
  'must never be removed. rag_readonly has no grant on documents itself.';

-- ---------------------------------------------------------------------------
-- 2. The read-only role
--
-- SELECT on one view. No grant on documents, chunks, threads, messages or
-- ingest_jobs -- so a query reaching for them fails on permissions, which is the
-- guarantee the structural checks are not asked to provide.
--
-- `chunks` is withheld on purpose and it is not an oversight: a SQL-readable
-- view over chunks would let the model `select content from chunks` and pull the
-- corpus wholesale, bypassing retrieval, ranking and the abstention gate.
-- Counting over content has its own narrow function instead
-- (count_documents_matching, 0002), whose return type cannot carry text.
-- ---------------------------------------------------------------------------

do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'rag_readonly') then
    create role rag_readonly nologin;
  end if;

  -- Required before the ownership transfer below: PostgreSQL only lets you hand
  -- an object to a role you are a member of, and Supabase's `postgres` is NOT a
  -- superuser --
  --   42501: must be able to SET ROLE "rag_readonly"
  -- Granted to current_user rather than hardcoding `postgres`, so this works
  -- whoever applies it. Membership also keeps `create or replace function`
  -- working on a re-run, once rag_readonly owns the function.
  execute format('grant rag_readonly to %I', current_user);
end
$$;

grant usage  on schema public              to rag_readonly;
grant select on public.documents_queryable to rag_readonly;

-- Explicit, so a future careless blanket grant does not widen this by accident.
revoke all on public.documents   from rag_readonly;
revoke all on public.chunks      from rag_readonly;
revoke all on public.threads     from rag_readonly;
revoke all on public.messages    from rag_readonly;
revoke all on public.ingest_jobs from rag_readonly;

-- ---------------------------------------------------------------------------
-- 3. The executor
--
-- THE PRIVILEGE DROP IS OWNERSHIP, NOT A ROLE SWITCH.
--
-- The obvious version -- `set local role rag_readonly` inside the function --
-- does not work: PostgreSQL rejects it with
--   42501: cannot set parameter "role" within security-definer function
-- Found by running it, not by reading about it.
--
-- A `security definer` function already executes with its OWNER's privileges,
-- so making rag_readonly the owner IS the drop -- and it is stronger than a SET
-- ROLE would have been: no window before the switch, no exception path that
-- could skip it, nothing to restore afterwards.
-- ---------------------------------------------------------------------------

create or replace function public.query_documents(
  sql      text,
  max_rows int default 200
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  result  jsonb;
  trimmed text;
begin
  trimmed := btrim(coalesce(sql, ''));

  -- Cheap structural guards. NOT the security boundary -- see the header. A
  -- false positive here (a semicolon inside a string literal) costs the model
  -- one retry; a false negative costs nothing, because the grants still hold.
  if trimmed = '' then
    raise exception 'empty_query';
  end if;
  if trimmed !~* '^(select|with)\s' then
    raise exception 'only_select_allowed';
  end if;
  if position(';' in rtrim(trimmed, '; ')) > 0 then
    raise exception 'single_statement_only';
  end if;

  -- Bounds cost. statement_timeout is an ordinary GUC, so unlike `role` it CAN
  -- be set inside a security-definer function.
  set local statement_timeout = '5s';

  -- The outer LIMIT caps a query that forgot one; jsonb_agg over the capped set
  -- keeps a runaway result from becoming a runaway response body.
  execute format(
    'select coalesce(jsonb_agg(_capped), ''[]''::jsonb) '
    'from (select * from (%s) _rows limit %s) _capped',
    trimmed, greatest(1, least(max_rows, 200))
  ) into result;

  return result;
end;
$$;

-- THE LINE THAT MAKES THE WHOLE DESIGN WORK. Without it the function runs as
-- postgres and the model's SQL has superuser reach. Owned by rag_readonly, the
-- same SQL can touch exactly one view.
--
-- PostgreSQL requires the incoming owner to hold CREATE on the object's schema,
-- otherwise:
--   42501: permission denied for schema public
-- Checked only at transfer time, so the grant is temporary. Leaving it would
-- make rag_readonly able to create objects in public -- far more than a role
-- that exists to run SELECTs needs.
grant create on schema public to rag_readonly;

alter function public.query_documents(text, int) owner to rag_readonly;

revoke create on schema public from rag_readonly;

-- auth.uid() reads request.jwt.claims, a GUC PostgREST sets per request, so it
-- resolves the same regardless of which role is executing. Supabase already
-- grants auth.uid() to PUBLIC, making this usually redundant -- it is here so
-- the dependency is explicit rather than inherited by luck.
--
-- The `auth` schema is owned by supabase_auth_admin, so `postgres` may not be
-- able to grant on it. That is fine and not worth failing over; if aggregates
-- come back empty, verification query 7 is the one to run.
do $$
begin
  grant usage   on schema auth         to rag_readonly;
  grant execute on function auth.uid() to rag_readonly;
exception when insufficient_privilege then
  raise notice 'auth grants skipped (owned by another role); relying on the '
               'PUBLIC grant Supabase ships. Verify with query 7.';
end
$$;

revoke execute on function public.query_documents(text, int) from public;
grant  execute on function public.query_documents(text, int) to authenticated;

comment on function public.query_documents(text, int) is
  'Runs a read-only SELECT for the text-to-SQL tool under the rag_readonly '
  'role, which can read only documents_queryable. Errors are meant to be '
  'returned to the model so it can correct its own SQL.';