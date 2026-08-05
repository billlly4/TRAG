# The chat API. Not the extractor, not the ingest worker -- those stay on the
# machine with the GPU, and reach this deployment only through Supabase.
#
# Built and run on arm64 (Oracle Ampere A1). Build it ON the VM: an x86 image
# does not transfer, and several dependencies here ship native wheels.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Compile bytecode at install time rather than on first request, and copy
# packages instead of symlinking them (symlinks into the uv cache break once the
# cache layer is gone).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Under /app so the chown below reaches it. The default is ~/.cache, and the
    # reranker is cached during the build as root while the container runs as
    # `trag` -- so with the default the runtime user finds an empty cache and
    # re-downloads the model on the first question. Silent, and only visible as
    # a slow first request (or a failure, on a host with no egress).
    HF_HOME=/app/.cache/huggingface

# The runtime user is created BEFORE anything is installed, and owns /app, so
# the install lands owned correctly. The obvious alternative -- install as root,
# then `chown -R trag:trag /app` at the end -- measured 6.5 GB and 290 s: chown
# rewrites every file, and Docker's copy-on-write stores a second copy of the
# whole virtualenv in that layer.
RUN useradd --create-home --uid 10001 trag
WORKDIR /app
RUN chown trag:trag /app
USER trag

# Dependencies before source, so editing a Python file does not re-run the
# install. uv.lock is what makes this reproducible -- --frozen refuses to
# re-resolve, so the server gets the versions that were tested.
COPY --chown=trag:trag pyproject.toml uv.lock ./

# THE CUDA PIN IS SKIPPED, NOT REMOVED.
#
# pyproject.toml pins torch to the cu126 index because that is what took
# reranking from 1260 ms to 80 ms on the RTX 3060. That index publishes no
# arm64 build, so a plain `uv sync` here does not merely waste space -- it fails
# to resolve. Editing pyproject.toml would fix the image and silently
# de-optimise the machine that does the real work.
#
# So: install everything except torch from the lock, then install torch from the
# CPU index.
#
# TORCHVISION MOVES WITH IT. Skipping torch alone builds an image that installs
# cleanly and then fails at runtime: torchvision ships compiled extensions built
# against a specific torch, so the lockfile's CUDA-matched torchvision on top of
# a CPU torch gives
#
#   RuntimeError: operator torchvision::nms does not exist
#
# which surfaces as transformers being unable to import BertForSequenceClassifi-
# cation -- i.e. the reranker silently unavailable, blamed on transformers.
#
# AND SO DOES ITS ENTIRE CUDA CLOSURE. `--no-install-package torch` skips torch
# but not what torch pulled in, so the first working build shipped 3.6 GB of
# nvidia-* wheels plus 691 MB of triton -- GPU tooling, in a CPU image, for a
# machine with no GPU. Measured: 20.1 GB total.
#
# These are x86-linux wheels, so on arm64 they are absent anyway and every line
# below is a no-op there. They are listed explicitly so the image is the same
# shape on both architectures, and so a laptop build does not quietly differ
# from the server it is meant to reproduce.
RUN --mount=type=cache,target=/home/trag/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package triton \
        --no-install-package cuda-bindings \
        --no-install-package cuda-pathfinder \
        --no-install-package cuda-toolkit \
        --no-install-package nvidia-cublas-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-cufile-cu12 \
        --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 \
        --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 \
        --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 \
        --no-install-package nvidia-nvshmem-cu12 \
        --no-install-package nvidia-nvtx-cu12 \
 && uv pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=trag:trag backend/ ./backend/

# The reranker's weights, baked into the image rather than fetched on first
# request. Without this the first question after every container start waits on
# a model download. ~90 MB.
#
# The model name is read from config.py rather than repeated here, so changing
# RERANK_MODEL cannot leave the image caching the wrong weights.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
from backend.app.config import Settings; \
m = Settings.model_fields['rerank_model'].default; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m); \
print(f'cached reranker: {m}')"

EXPOSE 8000

# No --reload, and bound to 0.0.0.0 because Docker's port mapping cannot reach
# a loopback bind. Caddy is what faces the internet; this listens only on the
# compose network.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
