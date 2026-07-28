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

    # Rejects oversized uploads at ingest. Without this the failure surfaces as
    # a context-window 400 mid-conversation, long after the upload "succeeded".
    # Deliberately below the 200k window to leave room for history and output.
    max_document_tokens: int = 150_000

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
