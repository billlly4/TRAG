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
_device = "cpu"

# Guards the lazy LOAD only -- not inference.
#
# It used to cover both. That was inherited caution from the Module 5 SIGSEGV,
# where docling's shared converter plus two threads took the server down with no
# Python traceback. Two things make it the wrong call here:
#
#   1. The reason is stale. That converter was docling's, and docling moved out
#      of this process entirely in Module 8.
#   2. It was measured, and it was the single largest cost in the app. p95 at 10
#      concurrent users was 12,969 ms -- almost exactly 10x the 1,285 ms
#      single-call cost, the signature of a strict queue. After the Ollama fix
#      it accounted for 87% of retrieval time at ONE user.
#
# A transformers model in eval() under inference_mode() holds no mutable state
# across a forward pass, which is the case PyTorch supports concurrently. The
# claim is verified by running 10 concurrent reranks in one process, not by
# trusting that sentence -- see loadlocal.py.
#
# Double-checked: the fast path skips the lock entirely once loaded, so this
# costs nothing per call. The re-check inside matters -- two threads can both
# pass the outer test before either takes the lock.
_lock = threading.Lock()


def _pick_device(settings) -> str:
    """GPU when there is one, unless configured otherwise.

    On CPU this model was 1351 ms for 20 candidates -- 90% of retrieval time at
    a single user -- purely because the default torch wheel is `+cpu`. It is a
    22M-parameter encoder; the arithmetic is not the problem, the hardware was.
    """
    want = settings.rerank_device
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if want == "cuda":
        log.warning("RERANK_DEVICE=cuda but torch reports no CUDA; using CPU")
    return "cpu"


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer

    with _lock:
        if _model is None:
            settings = get_settings()
            device = _pick_device(settings)
            log.info("loading reranker %s on %s", settings.rerank_model, device)
            tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.rerank_model
            )
            model.eval()
            # fp32 deliberately, even on GPU. rerank_min_score (-7.0) and
            # rerank_relative_drop (8.0) were measured against fp32 logits; fp16
            # would shift them by an unknown amount and silently move the
            # abstention gate. The model is small enough that fp32 is already
            # fast on GPU, so half precision buys little and risks the one thing
            # that was carefully calibrated.
            if device == "cuda":
                model.to(device)
            _device = device
            # CPU only. On GPU the work is not on these threads, and the tuned
            # value below was measured against CPU inference.
            #
            # Calls run concurrently now, so this is a per-call share of the
            # machine rather than a cap on its one exclusive user: N threads x M
            # concurrent calls must stay near the core count. At 4 threads and
            # 10 callers that is 40 threads on 16 cores, and the thrash costs
            # more than the serialisation it replaced.
            if device == "cpu":
                torch.set_num_threads(
                    max(1, min(settings.rerank_torch_threads, torch.get_num_threads()))
                )
            # Publish last: another thread's fast path reads _model, so it must
            # not become non-None until the model is fully built.
            _tokenizer, _model = tokenizer, model
    return _model, _tokenizer


def _infer(model, batch):
    """One forward pass, falling back to CPU if the GPU is full.

    The card is shared. Ollama holds nomic-embed-text and, during ingestion,
    qwen2.5vl at ~6 GB on a 6 GB device -- so there are windows where a 90 MB
    reranker genuinely cannot find room. Without this, a chat during a
    figure-heavy ingest would die on a CUDA OOM, which is a much worse outcome
    than being briefly slow.

    The demotion is permanent for the process rather than per-call: if VRAM is
    contended once it will be again shortly, and flapping between devices would
    pay the transfer cost on every attempt. A restart re-tries the GPU.
    """
    global _device
    if _device == "cuda":
        batch = {k: v.to("cuda") for k, v in batch.items()}
    try:
        with torch.inference_mode():
            return model(**batch).logits.cpu()
    except torch.cuda.OutOfMemoryError:
        log.warning("reranker hit CUDA OOM (GPU shared with Ollama); moving to CPU")
        model.to("cpu")
        _device = "cpu"
        torch.cuda.empty_cache()
        settings = get_settings()
        torch.set_num_threads(
            max(1, min(settings.rerank_torch_threads, torch.get_num_threads()))
        )
        with torch.inference_mode():
            return model(**{k: v.to("cpu") for k, v in batch.items()}).logits


def rerank(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """Re-score candidates by reading each against the query, best first.

    Never raises on an empty candidate list -- an empty retrieval is a normal
    outcome, not an error, and the caller should not have to special-case it.
    """
    if not hits:
        return []

    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    model, tokenizer = _load()
    # Truncation is at the model's 512-token limit while chunks target 800
    # tokens, so LONG PASSAGES ARE CUT. A fact in the tail of a chunk is
    # invisible to the reranker even though retrieval found the chunk --
    # worth remembering before blaming the model for a bad ordering.
    #
    # No lock here: every tensor below is local to this call, and the model is
    # read-only in eval(). See the note on _lock.
    batch = tokenizer(
        [query] * len(hits),
        [h.content for h in hits],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    logits = _infer(model, batch)

    # ms-marco cross-encoders emit a single relevance logit per pair. Higher is
    # more relevant; the scale is unbounded and uncalibrated, so it ranks
    # reliably but an absolute cutoff has to be measured, not guessed.
    scores = logits[:, 0].tolist()
    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)

    ranked = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
    return ranked[:k]
