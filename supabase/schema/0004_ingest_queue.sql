-- Durable ingestion queue.
--
-- Ingestion cannot be a BackgroundTask in the API process: a restart, a deploy
-- or a crash silently abandons every document in flight, left at 'pending' or
-- 'extracting' forever and indistinguishable from one that is merely slow. That
-- happened twice before this table existed. A queue row outlives the process
-- that created it, which is also what lets the worker run on a different
-- machine from the API entirely -- it never talks to the API, only to these
-- functions.
--
-- THE AUTHORIZATION PROBLEM, and why this file looks the way it does.
--
-- A worker process has no user JWT but needs Storage reads and documents /
-- chunks writes. The easy answer is a service_role key, which is exactly why it
-- is not used. Instead:
--
--   * the worker's ENTIRE database surface is the five functions below
--   * every one refuses a caller that is not a registered worker
--   * every one takes a JOB ID, never a user_id -- the caller cannot name whose
--     data it wants to touch, only act on a job it was handed
--   * Storage is read through a signed URL for ONE object, minted by the API
--     while it still holds the uploader's JWT: a capability, not a credential
--
-- These are the most security-sensitive functions in the project. RLS is not
-- the backstop on this path; the job-id scoping is.

-- ---------------------------------------------------------------------------
-- Who may drain the queue
--
-- Populated by hand, once, after creating the worker's account (see MANUAL
-- SETUP at the foot of this file). Deliberately not self-registering: a table
-- that accepts whoever calls first is a first-caller-wins hole.
-- ---------------------------------------------------------------------------

create table public.ingest_workers (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  label      text,
  created_at timestamptz not null default now()
);

-- RLS on with NO policy: unreachable from the API entirely. Only the SECURITY
-- DEFINER functions below read it, and they run as the owner.
alter table public.ingest_workers enable row level security;

-- ---------------------------------------------------------------------------
-- The queue
-- ---------------------------------------------------------------------------

create table public.ingest_jobs (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  user_id     uuid not null references auth.users(id) on delete cascade,

  -- Everything the worker needs to do the job without querying anything else.
  filename    text not null,
  mime_type   text,
  source_url  text not null,
  source_expires_at timestamptz,

  status       text not null default 'queued'
               check (status in ('queued','running','done','failed')),
  attempts     int  not null default 0,
  max_attempts int  not null default 3,

  -- Sets when a retry becomes eligible. Retries without a delay are nearly
  -- useless: a dependency being down fails FAST (connection refused takes
  -- milliseconds), so a job that requeues immediately burns all three attempts
  -- in about ten seconds -- while the extraction service takes roughly a minute
  -- to boot. The retry was firing exactly when it could not help and giving up
  -- exactly when it would have.
  not_before  timestamptz not null default now(),

  claimed_at  timestamptz,
  claimed_by  text,
  error       text,

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Serves the claim query: oldest eligible job first.
create index ingest_jobs_queue_idx on public.ingest_jobs (status, created_at);

alter table public.ingest_jobs enable row level security;

-- Owners can watch their own queue and enqueue. Deliberately NO update policy:
-- a job's state machine belongs to the functions below, not to whoever holds a
-- user token.
create policy ingest_jobs_owner_read on public.ingest_jobs
  for select to authenticated using (auth.uid() = user_id);

create policy ingest_jobs_owner_insert on public.ingest_jobs
  for insert to authenticated with check (auth.uid() = user_id);

grant select, insert on public.ingest_jobs to authenticated;

-- ---------------------------------------------------------------------------
-- The worker's functions
-- ---------------------------------------------------------------------------

create function public.is_ingest_worker()
returns boolean
language sql stable
security definer
set search_path = public
as $$
  select exists (select 1 from public.ingest_workers where user_id = auth.uid());
$$;

-- 1. Claim.
--
-- The `or (status = 'running' and claimed_at < ...)` clause IS the durability
-- mechanism: a worker that dies holding a job does not strand it -- the job goes
-- stale and the next claim picks it up. Everything else here is bookkeeping.
--
-- stale_after_seconds must exceed the longest legitimate job (measured: 421s
-- for a figure-heavy PDF) or two workers claim the same document.
--
-- SKIP LOCKED so several workers can drain the queue without fighting over the
-- same row.
create function public.claim_ingest_job(
  worker_label        text,
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
     and not_before <= now()
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
     set status     = 'running',
         attempts   = attempts + 1,
         claimed_at = now(),
         claimed_by = worker_label,
         updated_at = now()
   where id = job.id
   returning * into job;

  return job;
end;
$$;

-- 2. Progress. Writes documents.status, which Supabase Realtime publishes to
-- the UI -- so the badges keep working even though the writer is a different
-- process on a different machine.
create function public.report_ingest_status(
  job_id     uuid,
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
         error  = error_text
   where id = doc_id;
end;
$$;

-- 3. Finish.
--
-- One transaction on purpose. As separate statements, a crash between the chunk
-- swap and the status write leaves a document marked 'ready' with half its
-- chunks and nothing able to detect it. Here, either all of it lands or none.
create function public.complete_ingest_job(
  job_id     uuid,
  chunks     jsonb,
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

  -- Replace, not append. The delete is inside the same transaction as the
  -- insert, so there is no window with zero chunks.
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

-- 4. Fail. Requeues with an exponential delay while attempts remain, so a
-- transient outage recovers itself; gives up permanently after that rather than
-- looping on a document that will never work.
--
-- 60s then 120s, so a dependency has three minutes across the whole cycle to
-- come back. Long enough for a service restart; short enough that a user
-- watching a badge does not conclude the app is broken.
create function public.fail_ingest_job(job_id uuid, error_text text)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  job           public.ingest_jobs;
  delay_seconds int;
begin
  if not public.is_ingest_worker() then
    raise exception 'not_an_ingest_worker' using errcode = 'insufficient_privilege';
  end if;

  select * into job from public.ingest_jobs where id = job_id;
  if job.id is null then
    raise exception 'unknown_job';
  end if;

  if job.attempts < job.max_attempts then
    delay_seconds := 60 * power(2, greatest(job.attempts - 1, 0))::int;

    update public.ingest_jobs
       set status     = 'queued',
           error      = error_text,
           claimed_at = null,
           claimed_by = null,
           not_before = now() + make_interval(secs => delay_seconds),
           updated_at = now()
     where id = job_id;

    -- 'pending', not 'failed': the document IS still queued, and showing
    -- failure between retries makes a recovering document look broken.
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

-- Granted to `authenticated` because that is the only role a signed-in worker
-- can have. The guard is INSIDE each function, not in the grant -- any other
-- user calling these gets insufficient_privilege.
grant execute on function public.is_ingest_worker()                     to authenticated;
grant execute on function public.claim_ingest_job(text, int)            to authenticated;
grant execute on function public.report_ingest_status(uuid, text, text) to authenticated;
grant execute on function public.complete_ingest_job(uuid, jsonb, jsonb) to authenticated;
grant execute on function public.fail_ingest_job(uuid, text)            to authenticated;