"""Ingestion worker: drains the queue, one document at a time.

    uv run python -m backend.worker.main

Exists because ingestion used to run as a FastAPI BackgroundTask, which meant a
restart or a crash silently abandoned whatever was in flight. Here the state
lives in the database, so a dead worker loses nothing: its job goes stale and is
re-claimed.

AUTHORIZATION. This process holds no service_role key and no user credentials.
It signs in as its own account, and every database write goes through one of
four SECURITY DEFINER functions that take a JOB ID -- it cannot name whose data
it wants to touch, only act on work it was handed. Storage is read through a
signed URL for a single object, minted by the API when the job was created.
"""

import logging
import time
from typing import Any

import httpx

from ..app.chunking import chunk_text, embed_text
from ..app.config import get_settings
from ..app.deps import create_user_client
from ..app.embeddings import embed_documents
from ..app.extract import extract_text
from ..app.metadata import extract_metadata
from ..app.record import config_hash, extraction_config

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("worker")


class WorkerAuth:
    """The worker's own Supabase session, refreshed when it expires.

    The worker's token expires like anyone else's. Unlike a user's, it can
    simply sign in again -- which is the whole reason this design does not need
    to store somebody else's credentials.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        if not (self.settings.ingest_worker_email and self.settings.ingest_worker_password):
            raise SystemExit(
                "Set INGEST_WORKER_EMAIL and INGEST_WORKER_PASSWORD. The account "
                "must also be registered in public.ingest_workers -- see the "
                "manual setup notes in 0008_ingest_queue.sql."
            )
        self._client = None
        self._expires_at = 0.0

    def client(self):
        if self._client is None or time.time() > self._expires_at - 120:
            self._sign_in()
        return self._client

    def _sign_in(self) -> None:
        s = self.settings
        res = httpx.post(
            f"{s.supabase_url.rstrip('/')}/auth/v1/token?grant_type=password",
            headers={"apikey": s.supabase_publishable_key,
                     "Content-Type": "application/json"},
            json={"email": s.ingest_worker_email, "password": s.ingest_worker_password},
            timeout=30,
        )
        if res.status_code >= 300 or not res.json().get("access_token"):
            raise SystemExit(
                f"Worker sign-in failed ({res.status_code}). Check "
                f"INGEST_WORKER_EMAIL / INGEST_WORKER_PASSWORD."
            )
        payload = res.json()
        self._client = create_user_client(self.settings, payload["access_token"])
        self._expires_at = time.time() + payload.get("expires_in", 3600)
        log.info("worker authenticated as %s", self.settings.ingest_worker_email)


def _download(url: str) -> bytes:
    """Fetch the source bytes from the signed URL the API minted."""
    res = httpx.get(url, timeout=120, follow_redirects=True)
    res.raise_for_status()
    return res.content


def process(db, job: dict[str, Any]) -> None:
    """Run one document through the pipeline.

    Body moved from files._ingest. The only structural difference is that every
    write goes through an RPC instead of a table, because this process is not
    the user and must not be able to act as though it were.
    """
    job_id = job["id"]
    filename = job["filename"]
    mime = job.get("mime_type")

    def status(value: str) -> None:
        db.rpc("report_ingest_status",
               {"job_id": job_id, "new_status": value, "error_text": None}).execute()

    # Captured before work starts: these describe the run that actually
    # happened, even if settings change while it is in flight.
    cfg = extraction_config()
    cfg_hash = config_hash()

    data = _download(job["source_url"])

    status("extracting")
    text = extract_text(data, filename, mime)

    # Before chunking, not after: the extracted title is prepended to every
    # chunk's embedding input, so it has to exist before anything is embedded.
    status("analyzing")
    meta, meta_error = extract_metadata(text, filename)

    status("chunking")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text could be extracted from the document")

    status("embedding")
    title = (meta.title if meta and meta.title else None) or filename
    vectors = embed_documents([embed_text(c, title) for c in chunks])

    payload = [
        {
            "ordinal": c.ordinal,
            "content": c.content,
            "section": c.section,
            "token_count": c.token_count,
            # pgvector accepts its text form; the RPC casts it back.
            "embedding": "[" + ",".join(repr(float(x)) for x in v) + "]",
        }
        for c, v in zip(chunks, vectors)
    ]

    dump = meta.model_dump(mode="json") if meta else {}
    doc_fields = {
        "config_hash": cfg_hash,
        "extraction_config": cfg,
        "title": dump.get("title"),
        "doc_type": dump.get("doc_type"),
        "source_org": dump.get("source_org"),
        "authors": dump.get("authors"),
        "published_on": dump.get("published_on"),
        "published_year": dump.get("published_year"),
        "language": dump.get("language"),
        "topics": dump.get("topics"),
        "summary": dump.get("summary"),
        "metadata": dump or ({"metadata_error": meta_error} if meta_error else None),
    }

    # One transaction: chunks replaced and the document marked ready together,
    # so a crash cannot leave 'ready' with half a corpus behind it.
    db.rpc("complete_ingest_job",
           {"job_id": job_id, "chunks": payload, "doc_fields": doc_fields}).execute()

    log.info("ingested %s: %d chunks, metadata=%s",
             filename, len(chunks), "ok" if meta else f"failed ({meta_error})")


def main() -> None:
    settings = get_settings()
    auth = WorkerAuth()
    log.info("polling every %.1fs, stale after %ds",
             settings.ingest_poll_seconds, settings.ingest_stale_after_seconds)

    while True:
        try:
            db = auth.client()
            res = db.rpc("claim_ingest_job", {
                "worker_label": settings.ingest_worker_label,
                "stale_after_seconds": settings.ingest_stale_after_seconds,
            }).execute()
            job = res.data if isinstance(res.data, dict) else None

            if not job or not job.get("id"):
                time.sleep(settings.ingest_poll_seconds)
                continue

            log.info("claimed %s (%s, attempt %s)",
                     job["id"][:8], job["filename"], job["attempts"])
            try:
                process(db, job)
            except Exception as exc:  # noqa: BLE001
                # Any failure requeues while attempts remain, so a transient
                # Ollama or extractor outage heals itself without the user
                # touching anything.
                log.warning("job %s failed: %s", job["id"][:8], exc)
                db.rpc("fail_ingest_job",
                       {"job_id": job["id"], "error_text": str(exc)[:500]}).execute()

        except KeyboardInterrupt:
            log.info("stopping")
            return
        except Exception:  # noqa: BLE001
            # The loop itself must not die -- a database blip should cost a
            # poll, not the worker. Whatever it was holding goes stale and is
            # re-claimed.
            log.exception("worker loop error; retrying")
            time.sleep(settings.ingest_poll_seconds)


if __name__ == "__main__":
    main()
