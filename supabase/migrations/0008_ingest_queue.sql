-- Durable ingestion queue
--
-- Ingestion used to run as a FastAPI BackgroundTask, so a restart, a deploy or
-- a crash silently abandoned every document in flight -- left at 'pending' or
-- 'extracting' forever, indistinguishable from one that is merely slow. That
-- happened twice in this project. There were also no retries: a single Ollama
-- blip failed a document permanently.
--
-- THE AUTHORIZATION PROBLEM, and why this file looks the way it does.
--
-- A worker process has no user JWT, but needs Storage reads and documents /
-- chunks writes. The easy answer is a service_role key, and _ingest's own
-- docstring records why this codebase refuses it: "A service_role key would
-- also work, which is exactly why it is not used."
--
-- So the worker gets no blanket key and no stored user credentials. Instead:
--
--   * its ENTIRE database surface is the four functions below
--   * every one of them refuses a caller that is not a registered worker
--   * every one takes a JOB ID, never a user_id -- the caller cannot name
--     whose data it wants to touch, it can only act on a job it was handed
--   * Storage is read through a signed URL for ONE object, minted by the API
--     while it still holds the uploader's JWT
--
-- These functions are now the most security-sensitive code in the project. RLS
-- is no longer the backstop for this path; the job_id scoping is.

-- ---------------------------------------------------------------------------
-- Who is allowed to drain the queue
--
-- Populated by hand, once, after creating the worker's account. Deliberately
-- not self-registering: a table that accepts whoever calls first is a
-- first-caller-wins hole.
-- ---------------------------------------------------------------------------

create table if not exists public.ingest_workers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  label   text,
  created_at timestamptz not null default now()
);

alter table public.ingest_workers enable row level security;
-- No policy: unreachable from the API entirely. Only the SECURITY DEFINER
-- functions below read it, and they run as the owner.

-- ---------------------------------------------------------------------------
-- The queue
-- ---------------------------------------------------------------------------

create table if not exists public.ingest_jobs (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  user_id     uuid not null references auth.users(id) on delete cascade,

  -- Everything the worker needs to do the job without querying anything else.
  filename    text not null,
  mime_type   text,
  -- Signed URL for exactly one object. A capability, not a credential: it
  -- grants reading this file until it expires, and nothing else.
  source_url  text not null,
  source_expires_at timestamptz,

  status      text not null default 'queued'
              check (status in ('queued','running','done','failed')),
  attempts    int  not null default 0,
  max_attempts int not null default 3,

  claimed_at  timestamptz,
  claimed_by  text,
  error       text,

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Serves the claim query: oldest queued job first.
create index if not exists ingest_jobs_queue_idx
  on public.ingest_jobs (status, created_at);

alter table public.ingest_jobs enable row level security;

drop policy if exists ingest_jobs_owner_read on public.ingest_jobs;
drop policy if exists ingest_jobs_owner_insert on public.ingest_jobs;

-- Owners can see their own queue (so the UI can show depth later) and enqueue.
-- Deliberately NO update policy: a job's state machine belongs to the functions
-- below, not to whoever holds a user token.
create policy ingest_jobs_owner_read on public.ingest_jobs
  for select to authenticated using (auth.uid() = user_id);

create policy ingest_jobs_owner_insert on public.ingest_jobs
  for insert to authenticated with check (auth.uid() = user_id);

grant select, insert on public.ingest_jobs to authenticated;

-- ---------------------------------------------------------------------------
-- The worker's four functions
-- ---------------------------------------------------------------------------

create or replace function public.is_ingest_worker()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (select 1 from public.ingest_workers where user_id = auth.uid());
$$;

-- 1. Claim.
--
-- The `or (status = 'running' and claimed_at < ...)` clause IS the durability
-- mechanism. A worker that dies holding a job does not strand it: the job goes
-- stale and the next claim picks it up. Everything else here is bookkeeping.
--
-- SKIP LOCKED so several workers can drain the queue without fighting over the
-- same row.
create or replace function public.claim_ingest_job(
  worker_label text,
  stale_after_seconds int default 3600
)
returns public.ingest_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
  job public.ingest_jobs;
begin
  if not public.is_ingest_worker() then
    raise exception 'not_an_ingest_worker' using errcode = 'insufficient_privilege';
  end if;

  select * into job
    from public.ingest_jobs
   where attempts < max_attempts
     and (
       status = 'queued'
       or (status = 'running'
           and claimed_at < now() - make_interval(secs => stale_after_seconds))
     )
   order by created_at
   for update skip locked
   limit 1;

  if job.id is null then
    return null;
  end if;

  update public.ingest_jobs
     set status = 'running',
         attempts = attempts + 1,
         claimed_at = now(),
         claimed_by = worker_label,
         updated_at = now()
   where id = job.id
   returning * into job;

  return job;
end;
$$;

-- 2. Progress. Writes documents.status, which is what Supabase Realtime
-- publishes to the UI -- the badges keep working even though the writer is now
-- a different process.
create or replace function public.report_ingest_status(
  job_id uuid,
  new_status text,
  error_text text default null
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  doc_id uuid;
begin
  if not public.is_ingest_worker() then
    raise exception 'not_an_ingest_worker' using errcode = 'insufficient_privilege';
  end if;

  -- The document is DERIVED from the job. The caller never names it, which is
  -- what stops a compromised worker from touching an arbitrary row.
  select document_id into doc_id from public.ingest_jobs where id = job_id;
  if doc_id is null then
    raise exception 'unknown_job';
  end if;

  update public.documents
     set status = new_status,
         error = error_text
   where id = doc_id;
end;
$$;

-- 3. Finish.
--
-- One transaction on purpose. Today the chunk swap and the status write are
-- separate statements, so a crash between them leaves a document marked 'ready'
-- with half its chunks and nothing able to detect it. Here, either all of it
-- lands or none of it does.
create or replace function public.complete_ingest_job(
  job_id uuid,
  chunks jsonb,
  doc_fields jsonb
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  job public.ingest_jobs;
begin
  if not public.is_ingest_worker() then
    raise exception 'not_an_ingest_worker' using errcode = 'insufficient_privilege';
  end if;

  select * into job from public.ingest_jobs where id = job_id;
  if job.id is null then
    raise exception 'unknown_job';
  end if;

  -- Replace, not append. Old chunks go after the new ones are known-good,
  -- inside the same transaction, so there is no window with zero chunks.
  delete from public.chunks where document_id = job.document_id;

  insert into public.chunks
    (document_id, user_id, ordinal, content, section, token_count, embedding)
  select job.document_id,
         job.user_id,
         (c->>'ordinal')::int,
         c->>'content',
         c->>'section',
         (c->>'token_count')::int,
         (c->>'embedding')::vector
    from jsonb_array_elements(chunks) as c;

  update public.documents
     set status            = 'ready',
         error             = null,
         chunk_count       = jsonb_array_length(chunks),
         config_hash       = doc_fields->>'config_hash',
         extraction_config = doc_fields->'extraction_config',
         processed_at      = now(),
         title             = doc_fields->>'title',
         doc_type          = doc_fields->>'doc_type',
         source_org        = doc_fields->>'source_org',
         authors           = case when doc_fields->'authors' = 'null'::jsonb then null
                                  else array(select jsonb_array_elements_text(doc_fields->'authors')) end,
         published_on      = nullif(doc_fields->>'published_on','')::date,
         published_year    = nullif(doc_fields->>'published_year','')::int,
         language          = doc_fields->>'language',
         topics            = case when doc_fields->'topics' = 'null'::jsonb then null
                                  else array(select jsonb_array_elements_text(doc_fields->'topics')) end,
         summary           = doc_fields->>'summary',
         metadata          = doc_fields->'metadata'
   where id = job.document_id;

  update public.ingest_jobs
     set status = 'done', error = null, updated_at = now()
   where id = job_id;
end;
$$;

-- 4. Fail. Requeues while attempts remain, so a transient Ollama outage
-- recovers itself; gives up permanently after that rather than looping on a
-- document that will never work.
create or replace function public.fail_ingest_job(job_id uuid, error_text text)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  job public.ingest_jobs;
begin
  if not public.is_ingest_worker() then
    raise exception 'not_an_ingest_worker' using errcode = 'insufficient_privilege';
  end if;

  select * into job from public.ingest_jobs where id = job_id;
  if job.id is null then
    raise exception 'unknown_job';
  end if;

  if job.attempts < job.max_attempts then
    update public.ingest_jobs
       set status = 'queued', error = error_text, claimed_at = null,
           claimed_by = null, updated_at = now()
     where id = job_id;
    -- Document goes back to pending: it IS still queued, and showing 'failed'
    -- between retries would make a recovering document look broken.
    update public.documents set status = 'pending', error = null
     where id = job.document_id;
  else
    update public.ingest_jobs
       set status = 'failed', error = error_text, updated_at = now()
     where id = job_id;
    update public.documents set status = 'failed', error = error_text
     where id = job.document_id;
  end if;
end;
$$;

grant execute on function public.claim_ingest_job(text, int)          to authenticated;
grant execute on function public.report_ingest_status(uuid, text, text) to authenticated;
grant execute on function public.complete_ingest_job(uuid, jsonb, jsonb) to authenticated;
grant execute on function public.fail_ingest_job(uuid, text)          to authenticated;
grant execute on function public.is_ingest_worker()                   to authenticated;

-- Granted to `authenticated` because that is the only role a signed-in worker
-- can have. The guard is inside each function, not in the grant -- any other
-- user calling these gets insufficient_privilege.

-- ---------------------------------------------------------------------------
-- MANUAL SETUP (once, after applying this migration)
--
--   1. Create the worker account -- sign up normally with the email and
--      password you will put in .env as INGEST_WORKER_EMAIL / _PASSWORD.
--
--   2. Register it:
--        insert into public.ingest_workers (user_id, label)
--        select id, 'local-worker' from auth.users
--         where email = 'worker@yourdomain.example';
--
--   3. Confirm exactly one worker is registered:
--        select w.label, u.email from public.ingest_workers w
--          join auth.users u on u.id = w.user_id;
--
-- Verification
--
--   -- Functions exist and are SECURITY DEFINER (prosecdef = true)
--   select proname, prosecdef from pg_proc
--    where proname in ('claim_ingest_job','report_ingest_status',
--                      'complete_ingest_job','fail_ingest_job');
--
--   -- As a NORMAL user this must raise not_an_ingest_worker:
--   --   select public.claim_ingest_job('probe');
--
--   -- Queue state at a glance
--   select status, count(*), max(attempts) from public.ingest_jobs group by status;
-- ---------------------------------------------------------------------------
