"""Embeddings via Ollama's /api/embed.

Anthropic has no embeddings endpoint, so this is the one place the app talks
to a second model runtime. nomic-embed-text REQUIRES asymmetric task prefixes:
passages embed as `search_document: ...`, queries as `search_query: ...`.
Ollama does not add them, and omitting them raises no error -- retrieval just
quietly gets worse. Two functions exist so the call site cannot get the
prefix wrong.
"""

import threading

import httpx

from .config import get_settings

_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

_client: httpx.Client | None = None
_client_url: str | None = None
_client_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    pass


def _get_client(base_url: str) -> httpx.Client:
    global _client, _client_url
    with _client_lock:
        if _client is None or _client_url != base_url:
            if _client is not None:
                _client.close()
            _client = httpx.Client(
                base_url=base_url,
                timeout=httpx.Timeout(300.0, connect=5.0),
            )
            _client_url = base_url
    return _client


def _embed(inputs: list[str]) -> list[list[float]]:
    settings = get_settings()
    base_url = settings.ollama_base_url.rstrip("/")
    client = _get_client(base_url)
    payload = {"model": settings.embedding_model, "input": inputs}

    # Fail loudly. A document stuck at status='embedding' because the daemon is
    # down should surface as a readable error in documents.error, not a hang.
    #
    # The retry is the cost of pooling: a kept-alive connection can be closed by
    # the far end -- an Ollama restart, or its idle timeout -- and the next use
    # of that dead socket fails before the request is even sent. Retrying once
    # on a fresh connection turns that into a blip instead of a failed job.
    # ConnectError is deliberately NOT retried: it means the daemon is down, so
    # a second attempt only delays a message the caller needs now.
    for attempt in (1, 2):
        try:
            resp = client.post("/api/embed", json=payload)
            resp.raise_for_status()
            break
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as exc:
            if attempt == 1:
                continue
            raise EmbeddingError(
                f"Ollama connection kept failing at {base_url}: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Ollama unreachable at {base_url} -- is the daemon "
                f"running? (`ollama serve`)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama embed failed ({exc.response.status_code}): "
                f"{exc.response.text[:300]}"
            ) from exc

    embeddings = resp.json().get("embeddings")
    if not embeddings or len(embeddings) != len(inputs):
        raise EmbeddingError(
            f"Ollama returned {len(embeddings or [])} embeddings for {len(inputs)} inputs"
        )

    dim = len(embeddings[0])
    if dim != settings.embedding_dim:
        raise EmbeddingError(
            f"Embedding dimension {dim} does not match EMBEDDING_DIM="
            f"{settings.embedding_dim} (and the vector({settings.embedding_dim}) column)"
        )
    return embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage. One batched call, not one per chunk."""
    return _embed([_DOCUMENT_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    """Embed a search query against stored passages."""
    return _embed([_QUERY_PREFIX + text])[0]
