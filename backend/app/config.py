import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application config, loaded from the repo-root .env.

    Field names map to env vars case-insensitively, so `anthropic_api_key`
    reads ANTHROPIC_API_KEY. Measurements behind the defaults: findings.md.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5"

    # Separate from anthropic_model: metadata feeds the embedding input, so it
    # is part of config_hash. Tying them would make a chat-model swap re-embed
    # the whole corpus.
    metadata_model: str = "claude-haiku-4-5"

    supabase_url: str
    supabase_publishable_key: str

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "trag-module-1"

    # An EU workspace returns 403 against the US default, which looks exactly
    # like an invalid key.
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Org-scoped keys need this; without it /runs 403s while /orgs still 200s.
    langsmith_workspace_id: str | None = None

    cors_origins: str = "http://localhost:5173"

    # Bounds runaway generation. Not a cost control -- you pay for tokens
    # generated, not for the limit.
    max_output_tokens: int = 8192

    # Bounds ingestion work, not context.
    max_upload_bytes: int = 26_214_400  # 25 MB

    # Enforced by the `quotas` table (0006), not here -- the frontend holds a
    # real JWT and can reach PostgREST directly. These exist to produce a
    # readable message before the database raises a bare exception.
    max_threads_per_user: int = 5
    max_storage_bytes: int = 104_857_600  # 100 MB

    # Message ROWS, not turns: one search-using exchange writes four.
    max_messages_per_thread: int = 200

    # 127.0.0.1, never `localhost`: on Windows that resolves to IPv6 first and
    # waits for the timeout. Measured 2198ms vs 157ms.
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Must match the vector(768) column in 0002 -- changing the model means a
    # migration and re-embedding the corpus.
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Off = much faster ingest, but charts become unsearchable.
    vlm_model: str = "qwen2.5vl:7b"
    describe_pictures: bool = True

    # --- ingestion worker ----------------------------------------------------

    # No service_role key: the worker signs in as its own account and reaches
    # the database only through four SECURITY DEFINER functions (0008).
    ingest_worker_email: str | None = None
    ingest_worker_password: str | None = None
    ingest_worker_label: str = "local-worker"

    ingest_poll_seconds: float = 3.0

    # Must exceed the longest legitimate job (measured: 421s) or two workers
    # claim the same document.
    ingest_stale_after_seconds: int = 3600

    # Generous: a bulk re-process can queue for hours, and a URL expiring
    # mid-backlog fails the job.
    ingest_url_ttl_seconds: int = 604_800  # 7 days

    # Separate process because docling wraps native runtimes that can segfault
    # -- one did, taking the API with it. Loopback only: unauthenticated.
    extractor_url: str = "http://127.0.0.1:8001"
    # Measured 56s for one page with one figure.
    extractor_timeout: float = 600.0

    # Chosen for retrieval precision, not model limits: the embedding window is
    # 10x this.
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 120

    retrieval_top_k: int = 5

    # Inert across 0.15-0.50 on this corpus. Do not tune without a golden-set
    # run.
    retrieval_min_similarity: float = 0.30

    # --- hybrid retrieval ----------------------------------------------------

    # Strictly better than vector alone: the channels fail on disjoint
    # questions. findings.md
    retrieval_hybrid: bool = True

    # On for ABSTENTION, not ranking. Cosine cannot separate a real question
    # from gibberish here (0.500 vs 0.483, both clearing the 0.30 gate); the
    # cross-encoder separates them by ~2.6 logits.
    rerank_enabled: bool = True

    # RRF damping, from the original paper. Larger rewards cross-channel
    # agreement over a strong showing in one.
    rrf_k: int = 60

    # Wider than top_k: fusion can only reorder what it is given, and the
    # reranker can only rescue a passage that survived retrieval.
    rerank_candidates: int = 20

    # "cuda" warns rather than crashes if torch was built without it. rerank.py
    # demotes itself to CPU on a CUDA OOM -- the card is shared with Ollama.
    rerank_device: Literal["auto", "cpu", "cuda"] = "auto"

    # Per rerank call, CPU only. Measured on 16 cores; the arithmetic answer (2)
    # is wrong at both ends. Re-measure on a different core count.
    rerank_torch_threads: int = 3

    # 22M params, ~90MB. bge-reranker-v2-m3 scores better but is XLM-R-large --
    # seconds per query on CPU.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Measured: answerable -5.53..+7.28, absent -10.94..-8.14. No overlap. The
    # "natural" guess of 0.0 would reject four of seven real questions.
    rerank_min_score: float | None = -7.0

    # Distance below the BEST hit, not an absolute floor -- a good passage's
    # score depends on how answerable the question is at all. None disables.
    rerank_relative_drop: float | None = 8.0

    max_tool_turns: int = 5

    # --- tools ---------------------------------------------------------------


    sql_tool_enabled: bool = True

    count_tool_enabled: bool = True

    web_search_enabled: bool = True

    # The basic variant. 20260209 needs Opus 4.6+/Sonnet 4.6+ and 400s on
    # claude-haiku-4-5.
    web_search_tool_version: str = "web_search_20250305"

    # Each search bills on top of tokens.
    web_search_max_uses: int = 5

    # --- prompt caching ------------------------------------------------------

    # The Messages API is stateless, so every turn re-sends the whole thread at
    # full price. Caching charges 10% for re-reads. findings.md
    prompt_cache_enabled: bool = True

    # 1h writes at 2x and reads at 0.1x. 5m pays off within a turn but is a coin
    # flip across user messages, and an expired cache means paying the write
    # premium for nothing.
    prompt_cache_ttl: Literal["5m", "1h"] = "1h"

    # 0 = always tag and let the API decide. Below Anthropic's floor (2048
    # tokens on Haiku) nothing caches, at no penalty.
    prompt_cache_min_messages: int = 0

    # Three, not one: the agent is non-deterministic and a single sample reads a
    # lucky run as a pass. findings.md
    answer_eval_reps: int = 3

    # Read through Settings, not os.environ: pydantic-settings loads .env into
    # this object and does not export it.
    eval_user: str | None = None
    eval_password: str | None = None

    @property
    def jwks_url(self) -> str:
        # /auth/v1/jwks returns 404; the keys are at the RFC 8615 well-known
        # location and need no apikey header.
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def jwt_issuer(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def _export_langsmith(settings: Settings) -> None:
    if not settings.langsmith_tracing:
        return
    for key, value in (
        ("LANGSMITH_TRACING", "true"),
        ("LANGSMITH_API_KEY", settings.langsmith_api_key),
        ("LANGSMITH_ENDPOINT", settings.langsmith_endpoint),
        ("LANGSMITH_PROJECT", settings.langsmith_project),
        ("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id),
    ):
        if value:
            os.environ.setdefault(key, str(value))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _export_langsmith(settings)
    return settings
