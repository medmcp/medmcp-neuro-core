# syntax=docker/dockerfile:1
#
# medmcp-neuro — neuro-imaging tool stack as a fixed-environment MCP stdio server.
# GPU: torch/HD-BET (skull_strip) + antspyx (registration). Launched by the core
# via `docker run -i --device nvidia.com/gpu=all`.
#
# NOTE (this pass): FreeSurfer is NOT installed, so `segment_brain` is unavailable;
# add it as a follow-up layer (x86-only, license mounted at runtime).
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE} AS runtime

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

ENV PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1

ENTRYPOINT ["tini", "--", "medmcp-neuro"]
