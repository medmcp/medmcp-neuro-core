# syntax=docker/dockerfile:1
#
# medmcp-neuro — neuro-imaging tool stack as a fixed-environment MCP stdio server.
# GPU (all torch): HD-BET (skull_strip) + antspyx (registration) + FastSurfer
# (segment_brain). Launched by the core via `docker run -i --device nvidia.com/gpu=all`.
#
# segment_brain uses FastSurfer's seg-only pipeline (FastSurferVINN) — no FreeSurfer
# license required, torch-based, so it rides the same cu128 build and is arm64-capable.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`).
LABEL org.medmcp.stack='{"name": "medmcp-neuro", "gpu": true, "tool_timeout_sec": 7200, "skills_path": "/app/src/medmcp_neuro/skills"}'

# torch is pinned to the CUDA 12.8 (cu128) build (see pyproject), so it runs
# natively on any host driver >= R570 (Turing through Blackwell) — no forward-compat
# shim needed. libgomp1 is needed by antspyx/torch OpenMP paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Trust extra CA certs at build time behind a TLS-intercepting (MITM) proxy so
# uv/pip/git fetch through it. Drop the proxy root CA as a *.crt into ./certs/
# (gitignored; empty = no-op — CI / non-proxied builds add nothing). UV_NATIVE_TLS
# makes uv use the system trust store. Runtime is offline, so no production impact.
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
ENV UV_NATIVE_TLS=1

WORKDIR /app

# Frozen install from the committed lock (build-time network; runtime offline).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Python downloaders (HD-BET, FastSurfer checkpoints) use requests/urllib, which
# trust certifi's bundle — not the system store — so they also need pointing at the
# updated bundle to fetch through a MITM proxy. Harmless without a proxy CA, and the
# runtime is offline so there is no production impact.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Bake HD-BET weights so skull_strip runs with --network none (no runtime download).
RUN /app/.venv/bin/python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# ── FastSurfer (segment_brain) ────────────────────────────────────────────────
# Whole-brain segmentation via FastSurferVINN, seg-only (no FreeSurfer license).
# FastSurfer pins its own torch (2.7.1) which conflicts with the cu128 torch +
# HD-BET in /app/.venv, so it gets an ISOLATED venv (still GPU/cu128 via
# --torch-backend). segment_brain runs run_fastsurfer.sh with --py pointing here,
# so the two torch stacks never share a process. Subprocess-isolated either way.
ARG FASTSURFER_VERSION=v2.5.4
# seg-only: drop scikit-sparse — a recon_surf/surface-pipeline dep (needs SuiteSparse,
# and the pinned version isn't published) that is never imported on the seg path.
RUN git clone --depth 1 --branch ${FASTSURFER_VERSION} \
        https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer \
 && uv venv --python 3.12 /opt/fastsurfer-venv \
 && grep -viE '^\s*scikit-sparse' /opt/FastSurfer/requirements.txt > /tmp/fs-seg-req.txt \
 && uv pip install --python /opt/fastsurfer-venv/bin/python \
        --torch-backend=cu128 --index-strategy unsafe-best-match \
        -r /tmp/fs-seg-req.txt
# Bake the inference checkpoints so segment_brain runs offline (like HD-BET).
# PYTHONPATH so the script can import the FastSurferCNN package (ENV is set below).
RUN PYTHONPATH=/opt/FastSurfer /opt/fastsurfer-venv/bin/python \
        /opt/FastSurfer/FastSurferCNN/download_checkpoints.py --all

ENV PATH=/opt/FastSurfer:/app/.venv/bin:$PATH \
    FASTSURFER_HOME=/opt/FastSurfer \
    PYTHONPATH=/opt/FastSurfer \
    UV_NO_SYNC=1

ENTRYPOINT ["tini", "--", "medmcp-neuro"]
