"""Cross-encoder reranking of retrieved passages.

Vector and keyword search both score a chunk WITHOUT ever looking at it
alongside the question: cosine compares two independently-produced vectors, and
ts_rank_cd counts term overlap. A cross-encoder concatenates query and passage
and runs them through one transformer, so attention crosses between them and
the score reflects "does this passage answer this question" rather than "are
these two texts about similar things".

That is strictly more informative and far more expensive -- it is a forward
pass per candidate, not a dot product -- which is why it runs on a shortlist
that retrieval has already narrowed, never over the corpus.

Uses `transformers` directly rather than sentence-transformers: transformers is
already installed (docling depends on it), a cross-encoder is a sequence
classifier with one output, and the wrapper would add a dependency to save
about fifteen lines.
"""

import logging
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_settings
from .retrieval import Hit

log = logging.getLogger(__name__)

_model = None
_tokenizer = None

# Guards BOTH the lazy load and inference.
#
# Module 5 paid for this lesson with a SIGSEGV: docling's converter is a shared
# singleton, `_ingest` runs in a threadpool, and two threads through one torch
# module took the whole server down with no Python traceback. The reranker is
# the second thread-unsafe model runtime in this process and it runs in the
# chat path, so it gets the same treatment. Two threads racing the lazy load
# would also each build a copy of the model.
_lock = threading.Lock()


def _load():
    global _model, _tokenizer
    if _model is None:
        settings = get_settings()
        log.info("loading reranker %s", settings.rerank_model)
        _tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model)
        _model = AutoModelForSequenceClassification.from_pretrained(
            settings.rerank_model
        )
        _model.eval()
        # Without this, torch grabs all 16 cores for a model small enough that
        # thread coordination costs more than the work. It also stops the
        # reranker starving a concurrent docling conversion.
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    return _model, _tokenizer


def rerank(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """Re-score candidates by reading each against the query, best first.

    Never raises on an empty candidate list -- an empty retrieval is a normal
    outcome, not an error, and the caller should not have to special-case it.
    """
    if not hits:
        return []

    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    with _lock:
        model, tokenizer = _load()
        # Truncation is at the model's 512-token limit while chunks target 800
        # tokens, so LONG PASSAGES ARE CUT. A fact in the tail of a chunk is
        # invisible to the reranker even though retrieval found the chunk --
        # worth remembering before blaming the model for a bad ordering.
        batch = tokenizer(
            [query] * len(hits),
            [h.content for h in hits],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**batch).logits

    # ms-marco cross-encoders emit a single relevance logit per pair. Higher is
    # more relevant; the scale is unbounded and uncalibrated, so it ranks
    # reliably but an absolute cutoff has to be measured, not guessed.
    scores = logits[:, 0].tolist()
    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)

    ranked = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
    return ranked[:k]
