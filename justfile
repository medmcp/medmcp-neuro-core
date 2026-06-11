set quiet := true

default:
    @just --list

# Remove caches and build artifacts
clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf build
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*

@install_uv:
    if ! command -v uv >/dev/null 2>&1; then \
        echo "uv is not installed. Installing..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    else \
        echo "uv is available and ready to use..."; \
    fi

# Install uv and sync dev environment, register pre-commit hooks
setup: install_uv
    uv sync
    uv run pre-commit install

# Build the dedicated LST-AI sidecar venv used by segment_ms_lesions
lst-ai-setup: install_uv
    #!/usr/bin/env bash
    set -euo pipefail
    VENV="${MEDMCP_LST_AI_VENV:-$HOME/.medmcp_neuro/lst-ai-venv}"
    echo "Creating LST-AI sidecar venv at $VENV (Python 3.11)…"
    uv venv --python 3.11 "$VENV"
    echo "Installing pinned LST-AI + HD-BET from lst-ai/requirements.txt…"
    uv pip install --python "$VENV/bin/python" -r lst-ai/requirements.txt
    echo
    echo "Done. Point the tool at the sidecar 'lst' by exporting:"
    echo "  export MEDMCP_LST_AI_BIN=$VENV/bin/lst"
    echo "(the 'greedy' binary is fetched automatically on first run)"

# Run every CI check locally (lint, format, typecheck, tests)
check: lint format-check typecheck test

# Lint with ruff
lint:
    uv run ruff check

# Format code with ruff
format:
    uv run ruff format

# Check formatting without writing changes
format-check:
    uv run ruff format --check

# Strict type-checking with pyright
typecheck:
    uv run pyright

# Run the pytest suite
test *ARGS:
    uv run pytest {{ARGS}}

# Auto-fix lint findings and format
fix:
    uv run ruff check --fix
    uv run ruff format
