"""Embeddings via Ollama's /api/embed.

Anthropic has no embeddings endpoint, so this is the one place the app talks
to a second model runtime. nomic-embed-text REQUIRES asymmetric task prefixes:
passages embed as `search_document: ...`, queries as `search_query: ...`.
Ollama does not add them, and omitting them raises no error -- retrieval just
quietly gets worse. Two functions exist so the call site cannot get the
prefix wrong.
"""

import httpx

from .config import get_settings

_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    pass


def _embed(inputs: list[str]) -> list[list[float]]:
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"

    # Fail loudly. A document stuck at status='embedding' because the daemon is
    # down should surface as a readable error in documents.error, not a hang.
    try:
        resp = httpx.post(
            url,
            json={"model": settings.embedding_model, "input": inputs},
            timeout=httpx.Timeout(300.0, connect=5.0),
        )
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise EmbeddingError(
            f"Ollama unreachable at {settings.ollama_base_url} -- is the daemon "
            f"running? (`ollama serve`)"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(
            f"Ollama embed failed ({exc.response.status_code}): {exc.response.text[:300]}"
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
