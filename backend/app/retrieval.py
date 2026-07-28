"""Vector search over the user's chunks, via the match_chunks RPC.

PostgREST cannot express `order by embedding <=> $1`, so ranking lives in the
database function; this module embeds the query and applies the relevance
threshold. RLS applies inside the RPC (security invoker), so a user can only
ever search their own chunks.
"""

import logging
from dataclasses import dataclass

from supabase import Client

from .config import get_settings
from .embeddings import embed_query

log = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    document_id: str
    filename: str
    ordinal: int
    content: str
    similarity: float


def search(db: Client, query: str, top_k: int | None = None) -> list[Hit]:
    settings = get_settings()
    k = max(1, min(top_k or settings.retrieval_top_k, 20))

    vector = embed_query(query)

    # min_similarity=0 on purpose: the threshold is applied here, after
    # logging, so near-misses are visible. Scores the RPC filtered out would
    # never appear anywhere, and those logs are the raw material for the
    # Module 9 golden set.
    res = db.rpc(
        "match_chunks",
        {"query_embedding": vector, "match_count": k, "min_similarity": 0.0},
    ).execute()
    hits = [Hit(**row) for row in res.data]

    threshold = settings.retrieval_min_similarity
    for h in hits:
        log.info(
            "retrieval query=%r %s#%d similarity=%.3f %s",
            query, h.filename, h.ordinal, h.similarity,
            "PASS" if h.similarity >= threshold else "below-threshold",
        )

    return [h for h in hits if h.similarity >= threshold]
