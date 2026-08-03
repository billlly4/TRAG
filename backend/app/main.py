import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .deps import CurrentUserDep
from .routers import chat, files, threads
from .schemas import HealthResponse, MeResponse

# uvicorn configures its own loggers but leaves the root logger without a
# handler, so anything our modules log silently disappears. That is how a
# count_tokens failure could look identical to "this format cannot be counted".
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

# httpx logs every request at INFO. The worker polls the queue every 3s, so at
# INFO its log is almost entirely "claim_ingest_job 200 OK" heartbeats and a
# real failure scrolls past between them. Warnings and errors still surface --
# this hides the successes, not the problems.
logging.getLogger("httpx").setLevel(logging.WARNING)

settings = get_settings()

app = FastAPI(title="TRAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Unauthenticated liveness check."""
    return HealthResponse(status="ok", model=settings.anthropic_model)


@app.get("/api/me", response_model=MeResponse)
def me(user: CurrentUserDep) -> MeResponse:
    """Echoes the verified caller. The auth smoke test for Checkpoint A."""
    return MeResponse(id=user.id, email=user.email)


app.include_router(threads.router)
app.include_router(threads.usage_router)
app.include_router(files.router)
app.include_router(chat.router)
