from functools import lru_cache
from pathlib import Path

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

    # Anthropic has no embeddings endpoint, so embeddings come from Ollama.
    # EMBEDDING_DIM must match the vector(768) column in 0002_retrieval.sql --
    # changing the model means a new migration and re-embedding the corpus.
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # VLM captioning of figures during extraction. Off = much faster ingest,
    # but charts become unsearchable.
    vlm_model: str = "qwen2.5vl:7b"
    describe_pictures: bool = True

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
