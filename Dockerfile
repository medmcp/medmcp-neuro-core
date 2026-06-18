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

WORKDIR /app

# Frozen install from the committed lock (build-time network; runtime offline).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bake HD-BET weights so skull_strip runs with --network none (no runtime download).
RUN /app/.venv/bin/python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# ── FastSurfer (segment_brain) ────────────────────────────────────────────────
# Whole-brain segmentation via FastSurferVINN, seg-only (no FreeSurfer license).
# BUILD-VALIDATION TODOs (verify on first build):
#   1. Pin FASTSURFER_VERSION to a verified release tag.
#   2. FastSurfer's requirements.txt pins torch; if it conflicts with HD-BET/antspyx
#      in /app/.venv, move FastSurfer to its own venv and pass run_fastsurfer.sh
#      --python /opt/fastsurfer-venv/bin/python instead. (Subprocess-isolated either way.)
ARG FASTSURFER_VERSION=v2.3.3
RUN git clone --depth 1 --branch ${FASTSURFER_VERSION} \
        https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer \
 && uv pip install --python /app/.venv -r /opt/FastSurfer/requirements.txt
# Bake the inference checkpoints so segment_brain runs offline (like HD-BET).
RUN /app/.venv/bin/python /opt/FastSurfer/FastSurferCNN/download_checkpoints.py --all

ENV PATH=/opt/FastSurfer:/app/.venv/bin:$PATH \
    FASTSURFER_HOME=/opt/FastSurfer \
    PYTHONPATH=/opt/FastSurfer \
    UV_NO_SYNC=1

ENTRYPOINT ["tini", "--", "medmcp-neuro"]
