# syntax=docker/dockerfile:1
#
# medmcp-neuro-core — neuro-imaging tool stack as a fixed-environment MCP stdio server.
# GPU (all torch): HD-BET (skull_strip) + antspyx (registration) + FastSurfer
# (segment_brain). Launched by the core via `docker run -i --device nvidia.com/gpu=all`.
#
# segment_brain uses FastSurfer's seg-only pipeline (FastSurferVINN) — no FreeSurfer
# license required, torch-based, so it rides the same cu128 build and is arm64-capable.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`).
LABEL org.medmcp.stack='{"name": "medmcp-neuro-core", "gpu": true, "tool_timeout_sec": 7200, "skills_path": "/app/src/medmcp_neuro_core/skills"}'

# torch is pinned to the CUDA 12.8 (cu128) build (see pyproject), so it runs
# natively on any host driver >= R570 (Turing through Blackwell) — no forward-compat
# shim needed. libgomp1 is needed by antspyx/torch OpenMP paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# antspyx publishes aarch64 wheels only for 0.4.2; every release from 0.5 on is
# x86_64/macos only. We keep 0.6.3 on both architectures rather than pinning an
# older ANTs on arm64 — differing ANTs versions per architecture would mean
# registration output that depends on the machine it ran on. So on arm64 antspyx
# is compiled from its sdist, which builds ITK+ANTs via CMake and needs a full
# C++ toolchain (without it: "CMAKE_CXX_COMPILER not set, after EnableLanguage").
# This makes arm64 builds substantially slower; amd64 still installs the wheel and
# is unaffected.
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential cmake git python3-dev \
            zlib1g-dev libpng-dev libjpeg-dev libtiff-dev && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Trust extra CA certs at build time behind a TLS-intercepting (MITM) proxy so
# uv/pip/git fetch through it. Drop the proxy root CA as a *.crt into ./certs/
# (gitignored; empty = no-op — CI / non-proxied builds add nothing). UV_NATIVE_TLS
# makes uv use the system trust store. Runtime is offline, so no production impact.
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
ENV UV_NATIVE_TLS=1

WORKDIR /app

# On arm64 this compiles antspyx (ITK+ANTs) from source, which by default runs one
# compile job per core. ITK's template-heavy translation units are memory-hungry, so
# on a small runner the peak exceeds RAM and the job is killed with no diagnostic at
# all — no failing step, no log. Leave it unset for local builds on a large machine
# (full parallelism); CI passes a cap on the arm64 leg only.
ARG CMAKE_BUILD_PARALLEL_LEVEL=""

# Frozen install from the committed lock (build-time network; runtime offline).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL}" ]; then \
        export CMAKE_BUILD_PARALLEL_LEVEL MAKEFLAGS="-j${CMAKE_BUILD_PARALLEL_LEVEL}"; \
        echo "capping build parallelism at ${CMAKE_BUILD_PARALLEL_LEVEL}"; \
    fi; \
    uv sync --frozen --no-dev \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /app/.venv -name '*.a' -delete

# Python downloaders (HD-BET, FastSurfer checkpoints) use requests/urllib, which
# trust certifi's bundle — not the system store — so they also need pointing at the
# updated bundle to fetch through a MITM proxy. Harmless without a proxy CA, and the
# runtime is offline so there is no production impact.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Model weights come from third-party hosts (Zenodo for HD-BET, Deep-MI for
# FastSurfer) that intermittently answer 5xx. Unretried, a single bad response kills
# the entire image build — a Zenodo 504 on the HD-BET weights failed CI while the same
# URL served fine seconds later, and the sibling neuro-cancer stack hit the equivalent
# NVIDIA 504 three builds running. Retry with linear backoff.
RUN printf '%s\n' '#!/bin/sh' \
      'n=0' \
      'until "$@"; do' \
      '  n=$((n+1))' \
      '  [ "$n" -ge 5 ] && { echo "retry: failed after $n attempts: $*" >&2; exit 1; }' \
      '  echo "retry: attempt $n failed; sleeping $((n*15))s" >&2' \
      '  sleep $((n*15))' \
      'done' \
    > /usr/local/bin/retry && chmod +x /usr/local/bin/retry

# Bake HD-BET weights so skull_strip runs with --network none (no runtime download).
RUN retry /app/.venv/bin/python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# ── FastSurfer (segment_brain) ────────────────────────────────────────────────
# Whole-brain segmentation via FastSurferVINN, seg-only (no FreeSurfer license).
# FastSurfer pins torch==2.7.1 — the SAME version /app/.venv uses (pinned in
# pyproject) — so its seg deps install into that ONE venv instead of a second one,
# sharing a single ~8 GB torch + CUDA stack. run_fastsurfer.sh runs with
# --py /app/.venv/bin/python (FASTSURFER_PYTHON below); subprocess-isolated either way.
ARG FASTSURFER_VERSION=v2.5.4
# Everything below in one layer so the cleanup (.git, bytecode, static libs) actually
# shrinks the image instead of just shadowing files in an earlier layer.
# seg-only: drop scikit-sparse — a recon_surf/surface-pipeline dep (needs SuiteSparse,
# and the pinned version isn't published) that is never imported on the seg path.
# Bake only the FastSurferVINN (asegdkt) checkpoint with --vinn: the seg-only path never
# runs CerebNet/HypVINN/CC, so --all would download those for nothing. Offline at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 --branch ${FASTSURFER_VERSION} \
        https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer \
 && grep -viE '^\s*scikit-sparse' /opt/FastSurfer/requirements.txt > /tmp/fs-seg-req.txt \
 && uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match -r /tmp/fs-seg-req.txt \
 && retry env PYTHONPATH=/opt/FastSurfer /app/.venv/bin/python \
        /opt/FastSurfer/FastSurferCNN/download_checkpoints.py --vinn \
 && rm -rf /opt/FastSurfer/.git \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /app/.venv -name '*.a' -delete

ENV PATH=/opt/FastSurfer:/app/.venv/bin:$PATH \
    FASTSURFER_HOME=/opt/FastSurfer \
    FASTSURFER_PYTHON=/app/.venv/bin/python \
    PYTHONPATH=/opt/FastSurfer \
    UV_NO_SYNC=1

ENTRYPOINT ["tini", "--", "medmcp-neuro-core"]
