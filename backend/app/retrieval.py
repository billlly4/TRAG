"""Vector search over the user's chunks, via the match_chunks RPC.

PostgREST cannot express `order by embedding <=> $1`, so ranking lives in the
database function; this module embeds the query, passes metadata filters
through, and applies the relevance threshold. RLS applies inside the RPC
(security invoker), so a user can only ever search their own chunks.

Filters are applied by the RPC in its WHERE clause, never here. That ordering
is the point: filtering in Python after the top-k comes back can only remove
rows the ranking already picked, so a filter excluding the top 5 would return
nothing instead of the next 5.
"""

import logging
from dataclasses import dataclass, field

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

    # Returned by the RPC alongside the chunk so provenance and the metadata
    # that justified a filter are available without a second round trip.
    section: str | None = None
    doc_type: str | None = None
    source_org: str | None = None
    published_year: int | None = None

    def label(self) -> str:
        """Where this came from: 'report.pdf › Chapter 3 › Inventory Models'."""
        return f"{self.filename} › {self.section}" if self.section else self.filename


@dataclass
class Filters:
    """Metadata narrowing, as requested by the model in its tool call."""

    doc_types: list[str] = field(default_factory=list)
    source_orgs: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None

    def active(self) -> bool:
        return bool(
            self.doc_types
            or self.source_orgs
            or self.topics
            or self.year_min is not None
            or self.year_max is not None
        )

    def describe(self) -> str:
        """The filters in words -- for logs, and to tell the model what it asked."""
        parts = []
        if self.doc_types:
            parts.append(f"type={'/'.join(self.doc_types)}")
        if self.source_orgs:
            parts.append(f"source={'/'.join(self.source_orgs)}")
        if self.topics:
            parts.append(f"topics={'/'.join(self.topics)}")
        if self.year_min is not None or self.year_max is not None:
            parts.append(f"years={self.year_min or '…'}–{self.year_max or '…'}")
        return ", ".join(parts) or "none"

    def to_rpc(self) -> dict:
        # An empty list must become null, not an empty array: `= any('{}')` is
        # false for every row, so an empty array would filter everything out
        # rather than filtering nothing.
        return {
            "filter_doc_types": self.doc_types or None,
            "filter_source_orgs": self.source_orgs or None,
            "filter_topics": self.topics or None,
            "filter_year_min": self.year_min,
            "filter_year_max": self.year_max,
        }


def search(
    db: Client,
    query: str,
    top_k: int | None = None,
    filters: Filters | None = None,
    min_similarity: float | None = None,
) -> list[Hit]:
    """Rank the user's chunks against a query.

    `min_similarity` overrides the configured relevance threshold. The
    evaluation harness passes 0.0 to get the unthresholded ranking: ranking
    quality and threshold choice are separate questions, and measuring through
    the threshold cannot tell you whether the threshold itself is right.
    """
    settings = get_settings()
    k = max(1, min(top_k or settings.retrieval_top_k, 20))
    filters = filters or Filters()

    vector = embed_query(query)

    # min_similarity=0 on purpose: the threshold is applied here, after
    # logging, so near-misses are visible. Scores the RPC filtered out would
    # never appear anywhere, and those logs are the raw material for the
    # Module 9 golden set.
    res = db.rpc(
        "match_chunks",
        {
            "query_embedding": vector,
            "match_count": k,
            "min_similarity": 0.0,
            **filters.to_rpc(),
        },
    ).execute()
    hits = [Hit(**row) for row in res.data]

    threshold = (
        settings.retrieval_min_similarity if min_similarity is None else min_similarity
    )
    for h in hits:
        log.info(
            "retrieval query=%r filters=[%s] %s#%d similarity=%.3f %s",
            query, filters.describe(), h.filename, h.ordinal, h.similarity,
            "PASS" if h.similarity >= threshold else "below-threshold",
        )
    if not hits:
        # A filter that matched no documents is a different event from a corpus
        # with nothing relevant in it, and the Module 9 golden set needs to be
        # able to tell them apart after the fact.
        log.info(
            "retrieval query=%r filters=[%s] matched no rows",
            query, filters.describe(),
        )

    return [h for h in hits if h.similarity >= threshold]
