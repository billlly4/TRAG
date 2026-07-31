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

    # Per-channel scores, kept separate rather than squeezed into `similarity`
    # because they are NOT comparable: cosine is 0-1, ts_rank_cd is unbounded
    # and usually tiny, RRF is ~0.02, and a cross-encoder logit runs about
    # -10..+10. Collapsing them into one field would make any threshold
    # meaningless the moment more than one channel is involved.
    keyword_rank: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    def label(self) -> str:
        """Where this came from: 'report.pdf › Chapter 3 › Inventory Models'."""
        return f"{self.filename} › {self.section}" if self.section else self.filename

    def gate_score(self) -> float:
        """The score that decides whether this hit counts as relevant.

        Which one that is depends on how the hit was produced, so the pipeline
        that ranked it owns the answer. A reranked hit is judged by the
        cross-encoder -- the only score here produced by actually reading the
        query and passage together -- and a keyword-only hit must never be
        judged by `similarity`, which is 0.0 for it by construction.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.keyword_rank is not None and self.similarity == 0.0:
            return self.keyword_rank
        return self.similarity


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


def search_keyword(
    db: Client,
    query: str,
    top_k: int | None = None,
    filters: Filters | None = None,
) -> list[Hit]:
    """Postgres full-text search over the same chunks.

    Complements the vector channel rather than competing with it: embeddings
    summarise meaning and lose rare literal tokens, while this matches them
    exactly. No relevance threshold is applied -- ts_rank_cd has no meaningful
    absolute scale, and the fused pipeline gates on the reranker instead.
    """
    settings = get_settings()
    k = max(1, min(top_k or settings.retrieval_top_k, 50))
    filters = filters or Filters()

    res = db.rpc(
        "match_chunks_keyword",
        {"query_text": query, "match_count": k, **filters.to_rpc()},
    ).execute()
    hits = [Hit(**row) for row in res.data]

    log.info(
        "keyword query=%r filters=[%s] hits=%d",
        query, filters.describe(), len(hits),
    )
    return hits


def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion over N ranked lists.

    Takes a LIST of ranked lists, never a (vector, keyword) pair -- that is the
    CLAUDE.md rule, and the reason is that adding or removing a channel later
    must not require reopening this function.

    RRF scores by RANK, not by score: sum of 1/(k + rank) across the lists a
    chunk appears in. That is exactly why it can merge channels whose scores
    are incommensurable -- cosine and ts_rank_cd cannot be added or averaged
    meaningfully, but "was 3rd here and 1st there" always can.

    k=60 is the constant from the original paper. It damps the difference
    between top ranks, so appearing in several lists beats ranking first in one
    -- which is the behaviour that makes fusion worth doing.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            # Keep the richest copy of each chunk: channels return the same row
            # but populate different score fields, so merge rather than
            # overwrite or the keyword rank is lost when vector wins.
            if hit.id in best:
                existing = best[hit.id]
                existing.similarity = max(existing.similarity, hit.similarity)
                if hit.keyword_rank is not None:
                    existing.keyword_rank = hit.keyword_rank
            else:
                best[hit.id] = hit

    for chunk_id, score in scores.items():
        best[chunk_id].rrf_score = score

    return sorted(best.values(), key=lambda h: h.rrf_score or 0.0, reverse=True)


def hybrid_search(
    db: Client,
    query: str,
    top_k: int | None = None,
    filters: Filters | None = None,
) -> list[Hit]:
    """Vector + keyword, fused by RRF, optionally reranked.

    The candidate pool is deliberately wider than the final k: fusion can only
    reorder what it is given, and a reranker can only rescue a passage that
    survived retrieval. Both channels therefore fetch `rerank_candidates`
    before anything is cut.
    """
    settings = get_settings()
    k = max(1, min(top_k or settings.retrieval_top_k, 20))
    depth = max(k, settings.rerank_candidates)

    # Unthresholded on purpose: the gate is applied once, at the end, on
    # whichever score the final pipeline stage produced.
    channels = [
        search(db, query, top_k=depth, filters=filters, min_similarity=0.0),
        search_keyword(db, query, top_k=depth, filters=filters),
    ]
    fused = rrf_fuse(channels, k=settings.rrf_k)

    # The relevance gate. Fusion reorders, it does not judge -- and the vector
    # channel was queried unthresholded so RRF could see a full ranking, so
    # without this every query returns k chunks no matter how irrelevant, and
    # the model stops declining. Module 2 verified that declining behaviour
    # explicitly; hybrid must not quietly delete it.
    #
    # A hit survives on EITHER channel's evidence: a cosine above the
    # threshold, or a genuine lexical match. The union matters -- gating the
    # fused list on cosine alone would discard every keyword-only hit, which is
    # the entire reason the keyword channel exists.
    fused = [
        h for h in fused
        if h.similarity >= settings.retrieval_min_similarity or h.keyword_rank is not None
    ]

    if settings.rerank_enabled and fused:
        from .rerank import rerank  # imported lazily: loads torch on first use

        fused = rerank(query, fused[:depth], top_k=k)

        # Abstention is a QUERY-level decision, not a per-passage filter: if
        # the best passage in the corpus is not relevant, nothing is; if it is,
        # the rest of the ranked list is the supporting context.
        #
        # Filtering per hit instead measurably costs recall -- relevant
        # passages ranked 2nd or 3rd legitimately score -3 to -5 while the top
        # hit scores +7, so a single cutoff applied to every hit prunes good
        # context. Measured: per-hit gating dropped passage hit rate from 100%
        # to 83.3% while adding nothing to abstention.
        if (
            settings.rerank_min_score is not None
            and fused
            and (fused[0].rerank_score or 0.0) < settings.rerank_min_score
        ):
            log.info(
                "hybrid query=%r best rerank %.2f < %.2f -- declining",
                query, fused[0].rerank_score or 0.0, settings.rerank_min_score,
            )
            fused = []

        # Having decided the query IS answerable, drop passages far below the
        # winner. Query-level gating alone keeps the whole top-k, so a question
        # with three good sources still returns five -- and the two passengers
        # are shown to the user as sources, which costs more trust than the
        # tokens cost context.
        #
        # Relative, not absolute: a good passage's score depends on how
        # answerable the question is at all, so the best hit for one question
        # can score below the fifth hit of another.
        elif fused and settings.rerank_relative_drop is not None:
            best = fused[0].rerank_score or 0.0
            kept = [
                h for h in fused
                if best - (h.rerank_score or 0.0) <= settings.rerank_relative_drop
            ]
            if len(kept) < len(fused):
                log.info(
                    "hybrid query=%r dropped %d passage(s) more than %.1f below best",
                    query, len(fused) - len(kept), settings.rerank_relative_drop,
                )
            fused = kept

    for h in fused[:k]:
        log.info(
            "hybrid query=%r %s#%d rrf=%.4f cos=%.3f kw=%s rerank=%s",
            query, h.filename, h.ordinal, h.rrf_score or 0.0, h.similarity,
            f"{h.keyword_rank:.4f}" if h.keyword_rank is not None else "-",
            f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-",
        )
    return fused[:k]
