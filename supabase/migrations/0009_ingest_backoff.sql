-- Retry backoff
--
-- Found by testing 0008: retries without a delay are nearly useless.
--
-- A dependency being down fails FAST -- connection refused takes milliseconds --
-- so a job that requeues immediately burns all three attempts in about ten
-- seconds. The extraction service takes roughly a minute to boot (it loads
-- docling's models), so by the time it was back, the document had already
-- failed permanently. The retry mechanism was firing exactly when it could not
-- help, and giving up exactly when it would have.
--
-- Backoff is what makes a retry mean "try again once the world has had a chance
-- to change" rather than "try again immediately, twice, and give up".

alter table public.ingest_jobs
  add column if not exists not_before timestamptz not null default now();

-- Claim: skip jobs that are still serving their backoff.
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
     and not_before <= now()
     and (
       status = 'queued'
       -- Stale reclaim: a worker that died holding this job does not strand it.
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

-- Fail: requeue with an exponential delay, or give up.
--
-- 60s then 120s, so a dependency has three minutes across the whole cycle to
-- come back before the document is declared failed. Long enough for a service
-- restart; short enough that a user watching a badge does not conclude the app
-- is broken.
create or replace function public.fail_ingest_job(job_id uuid, error_text text)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  job public.ingest_jobs;
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
       set status = 'queued',
           error = error_text,
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

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- Column exists
--   select column_name from information_schema.columns
--    where table_name = 'ingest_jobs' and column_name = 'not_before';
--
--   -- Anything waiting out a backoff right now
--   select id, status, attempts, not_before - now() as waits_for, error
--     from public.ingest_jobs where not_before > now();
-- ---------------------------------------------------------------------------
