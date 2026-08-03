from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application config, loaded from the repo-root .env.

    Field names map to env vars case-insensitively, so `anthropic_api_key`
    reads ANTHROPIC_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5"

    # Deliberately separate from anthropic_model. Metadata feeds the embedding
    # input (title + section are prepended to each chunk), so it is part of
    # config_hash -- and tying it to the chat model would mean swapping the
    # chat model marks the entire corpus stale and re-embeds it.
    metadata_model: str = "claude-haiku-4-5"

    supabase_url: str
    supabase_publishable_key: str

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "trag-module-1"

    # LangSmith is region-partitioned and the SDK defaults to US. An EU-hosted
    # workspace (eu.smith.langchain.com) returns 403 against the US endpoint,
    # which looks exactly like an invalid key.
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # An org-scoped key must name the workspace explicitly; the SDK turns this
    # into the X-Tenant-Id header. Without it, /runs and /sessions return 403
    # while /workspaces and /orgs still return 200 -- so the key looks valid
    # right up until tracing silently fails.
    langsmith_workspace_id: str | None = None

    cors_origins: str = "http://localhost:5173"

    # Output ceiling for chat. Bounds runaway generation; it does NOT cap input
    # and is not a cost control -- you pay for tokens generated, not the limit.
    max_output_tokens: int = 8192

    # Bounds ingestion work (extraction + embedding time), not context. The old
    # max_document_tokens gate is gone: chunking is exactly what removed the
    # context ceiling it guarded against.
    max_upload_bytes: int = 26_214_400  # 25 MB

    # Per-user quotas. These mirror the values in the `quotas` table, which is
    # where they are actually ENFORCED (0006_quotas.sql) -- the frontend holds a
    # real user JWT and can reach PostgREST directly, so a check that lives only
    # here is advice. Their job in the API is to produce a readable message
    # before the database raises a bare exception.
    max_threads_per_user: int = 5
    max_storage_bytes: int = 104_857_600  # 100 MB

    # Message ROWS, not conversational turns: one search-using exchange writes
    # four rows (question, tool call, tool results, answer), so 200 is roughly
    # 50 exchanges. Chats live in Postgres, so this bounds the database, which
    # max_storage_bytes -- counting only the Storage bucket -- does not.
    max_messages_per_thread: int = 200

    # Anthropic has no embeddings endpoint, so embeddings come from Ollama.
    # EMBEDDING_DIM must match the vector(768) column in 0002_retrieval.sql --
    # changing the model means a new migration and re-embedding the corpus.
    # 127.0.0.1, never `localhost`. Measured: resolving `localhost` on Windows
    # costs ~2.08s per connection -- an IPv6 ::1 attempt that times out before
    # falling back to IPv4. That is not a theory: `GET /api/version`, which
    # returns a string and touches no model, took 2198ms via `localhost` and
    # 157ms via `127.0.0.1`. It made every query embedding look 40x slower than
    # the model actually is. Ollama binds loopback IPv4 by default; point a real
    # host here via OLLAMA_BASE_URL if it runs elsewhere.
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # VLM captioning of figures during extraction. Off = much faster ingest,
    # but charts become unsearchable.
    vlm_model: str = "qwen2.5vl:7b"
    describe_pictures: bool = True

    # Extraction runs in a SEPARATE PROCESS (backend/extractor). docling wraps
    # native model runtimes that can segfault, and one did -- taking the API
    # down with it. Out of process, that costs an extraction instead of the
    # server. Must stay on loopback: the service is unauthenticated by design.
    # --- ingestion worker ----------------------------------------------------
    # Ingestion runs in a THIRD process draining a queue, so a restart or crash
    # no longer abandons documents mid-pipeline. The worker holds no
    # service_role key and no user credentials: it signs in as its own account
    # and reaches the database only through four SECURITY DEFINER functions
    # that take a job id (0008_ingest_queue.sql).
    ingest_worker_email: str | None = None
    ingest_worker_password: str | None = None
    ingest_worker_label: str = "local-worker"

    # How often to ask for work when the queue is empty.
    ingest_poll_seconds: float = 3.0

    # How long a claimed job may go silent before another worker may take it.
    # MUST exceed the longest legitimate job: a figure-heavy PDF measured 421s
    # on CPU. Too low and two workers process the same document; too high and a
    # crashed worker's job waits that long to be retried.
    ingest_stale_after_seconds: int = 3600

    # Lifetime of the per-job signed URL. Generous because a bulk re-process can
    # queue for hours, and a URL that expires mid-backlog fails the job.
    ingest_url_ttl_seconds: int = 604_800  # 7 days

    extractor_url: str = "http://127.0.0.1:8001"
    # Generous on purpose. A large PDF with VLM captioning legitimately runs for
    # minutes; Module 2 measured 56s for one page with a single figure.
    extractor_timeout: float = 600.0

    # 800 tokens is chosen for retrieval precision, not model limits -- the
    # embedding model's 8192-token window is 10x this target.
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 120

    retrieval_top_k: int = 5

    # A guess. Do not tune without a golden-set run (Module 9); log scores now.
    # Measured 2026-07-30: inert across 0.15-0.50 on the current corpus -- every
    # relevant chunk scores 0.53+, so this neither helps nor hurts yet. It stays
    # until an unanswerable golden question exists to show the cost of lowering it.
    retrieval_min_similarity: float = 0.30

    # --- Module 6: hybrid retrieval -----------------------------------------
    # ON, from measurement (2026-07-31, config_hash 1a38fa64b0e1, 7 questions):
    #   vector    hit 100%  passage  83.3%  mrr 0.929
    #   keyword   hit  71.4% passage  83.3%  mrr 0.714
    #   rrf       hit 100%  passage 100.0%  mrr 1.000
    # The channels fail on disjoint questions: vector ranked "intrafunctional
    # view" 2nd with the wrong passage (a rare literal token embeddings blur),
    # while keyword missed the two paraphrase questions entirely. Fused, both
    # are answered. Strictly better than vector on every metric, so it is on.
    retrieval_hybrid: bool = True

    # ON -- for abstention, not for ranking.
    #
    # On the golden set the reranker scored identically to rrf (100% on
    # everything), because rrf already saturates a 7-question set. Ranking is
    # not why it is enabled. This is:
    #
    #   nonsense query, cosine:  0.483 / 0.414   (both clear the 0.30 gate)
    #   real question,  cosine:  0.500 / 0.381   (the SAME range)
    #
    # nomic-embed-text compresses everything into a narrow high band, so no
    # cosine threshold can separate a real question from random characters --
    # the relevance gate has never actually worked. The cross-encoder separates
    # them by ~2.6 logits with no overlap (see rerank_min_score), and is the
    # only component in the pipeline that can.
    rerank_enabled: bool = True

    # RRF's damping constant, from the original paper. Larger flattens the
    # advantage of top ranks, so agreement across channels matters more.
    rrf_k: int = 60

    # How many candidates each channel fetches before fusion, and how many the
    # reranker reads. Wider than top_k on purpose: fusion can only reorder what
    # it is given, and the reranker can only rescue a passage that survived
    # retrieval in the first place.
    rerank_candidates: int = 20

    # "auto" uses the GPU when torch reports one, else CPU. "cpu" forces CPU;
    # "cuda" asks for GPU and warns (rather than crashes) if torch was built
    # without it -- which is the state this project shipped in until measuring
    # showed the reranker was 90% of retrieval time purely for being on the
    # wrong device.
    #
    # The card is shared with Ollama, so rerank.py demotes itself to CPU on a
    # CUDA OOM rather than failing the request.
    rerank_device: Literal["auto", "cpu", "cuda"] = "auto"

    # Torch threads PER rerank call. CPU only -- ignored on GPU. Reranking runs concurrently now, so this is
    # a share of the machine rather than a cap on its one exclusive user.
    #
    # Measured on 16 cores, 20 candidates, rather than reasoned about -- the
    # arithmetic answer (threads x callers ~= cores, so 2) is wrong at both ends:
    #
    #   threads   c=1 p50   c=10 p50   c=10 max
    #         1     3080ms     6185ms     6769ms
    #         2     1752ms     4992ms     5700ms
    #         3     1287ms     4753ms     5072ms   <-- best at BOTH
    #         4     1047ms     5336ms     5669ms
    #         8      909ms     5474ms     5715ms
    #
    # 3 is not a compromise: c=1 lands at 1287ms against the old locked build's
    # 1281ms, so single-user latency is unchanged while the ten-user tail falls
    # from 13.3s to 5.1s. Above 3 the extra threads win at c=1 and lose under
    # load; below it, everything is worse. Re-measure on a different core count
    # -- this number is a property of the machine, not of the model.
    rerank_torch_threads: int = 3

    # 22M params, ~90MB, English. Roughly 30-80ms for 20 passages on CPU, which
    # is what makes it viable in the chat path. bge-reranker-v2-m3 is cached
    # locally and scores better, but it is XLM-R-large on a CPU-only torch
    # build -- seconds per query.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Measured 2026-07-31 over 7 answerable + 4 absent queries, scoring the top
    # hit of the fused pipeline:
    #
    #   answerable   -5.53 .. +7.28   (min: "what is the chapter2 pdf about?")
    #   absent      -10.94 .. -8.14   (max: an EOQ question, a near-miss)
    #
    # No overlap; -7.0 sits in the empty band. Note that 0.0 -- the "natural"
    # decision boundary of the classifier, and the number a reasonable person
    # would guess -- would reject FOUR of the seven real questions. This value
    # had to be measured; it could not be reasoned to.
    #
    # Re-measure as the corpus grows: the gap is only 2.6 logits wide and rests
    # on 11 data points.
    rerank_min_score: float | None = -7.0

    # How far below the BEST hit a passage may score and still be returned.
    #
    # Absolute per-hit gating does not work here -- a good passage's score
    # depends on how well the question is answerable at all, so q004's best hit
    # (-5.53) scores lower than q007's fifth (-2.43). Relative distance from the
    # winner is the stable signal.
    #
    # Measured 2026-07-31 over the golden set plus a real multi-document query:
    #   must keep  gap 4.29  (q001's rank-2 hit, which carries its must_contain)
    #   must drop  gap 11.14 (unrelated policy chunks riding along on a name query)
    # 8.0 sits in that band. None disables the cutoff.
    rerank_relative_drop: float | None = 8.0

    max_tool_turns: int = 5

    # --- Module 7 tools ------------------------------------------------------

    # Text-to-SQL over document metadata. Requires 0011_text_to_sql.sql to have
    # been applied; without it every call comes back as a missing-function error
    # the model cannot fix, so turn this off rather than half-install it.
    sql_tool_enabled: bool = True

    # Web search is requested PER MESSAGE by the user (ChatRequest.web_search).
    # This flag is the operator's off switch on top of that -- when false the
    # tool is never declared no matter what the client asks for.
    #
    # Deliberately not a fallback for failed retrieval: a document question the
    # corpus cannot answer must still abstain, or the app quietly stops being
    # grounded in the user's files. Web search is something the user opts into.
    web_search_enabled: bool = True

    # 20250305 is the BASIC variant. The newer 20260209 (dynamic filtering) needs
    # Opus 4.6+/Sonnet 4.6+ and 400s on claude-haiku-4-5 -- verified, not assumed:
    # "'claude-haiku-4-5-20251001' does not support programmatic tool calling".
    # Raising anthropic_model to a 4.6+ model is the only reason to change this.
    web_search_tool_version: str = "web_search_20250305"

    # Each search is billed on top of tokens, so this bounds what one message can
    # spend without the user seeing it happen.
    web_search_max_uses: int = 5

    # The account backend/evaluation signs in as. Read through Settings rather
    # than os.environ because pydantic-settings loads .env into this object and
    # does NOT export it to the process environment -- the same trap llm.py
    # documents for the LangSmith SDK.
    eval_user: str | None = None
    eval_password: str | None = None

    @property
    def jwks_url(self) -> str:
        # Note the path: /auth/v1/jwks returns 404. The public keys are served
        # from the RFC 8615 well-known location, and unlike the rest of the
        # Supabase API it needs no apikey header.
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def jwt_issuer(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
