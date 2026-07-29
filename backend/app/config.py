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
    retrieval_min_similarity: float = 0.30

    max_tool_turns: int = 5

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
