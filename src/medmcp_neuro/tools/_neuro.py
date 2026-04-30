"""Shared internal utilities for medmcp-neuro tools."""

import os
import shutil
import sys
from pathlib import Path


def detect_devices() -> list[str]:
    """Return available compute devices, always including 'cpu'.

    Returns:
        List containing 'cpu' plus any of 'cuda' or 'mps' that are available.
    """
    import torch

    devices: list[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def cuda_unavailable_note() -> str:
    """Return an actionable note if a CUDA-capable torch is installed but CUDA is unavailable.

    Returns:
        Human-readable note with install guidance, or empty string if CUDA works or is absent.
    """
    import torch

    if torch.cuda.is_available():
        return ""
    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda_ver is None:
        return (
            " Note: CPU-only torch is installed — GPU inference is not possible."
            " See https://pytorch.org/get-started/locally/ to install a CUDA-enabled build."
        )
    return (
        f" Note: torch was compiled for CUDA {cuda_ver} but CUDA is currently unavailable."
        " Check your driver or see https://pytorch.org/get-started/locally/ for install options."
    )


def find_binary(name: str, env_var: str) -> str | None:
    """Locate a neuroimaging tool binary.

    Checks the environment variable first, then PATH, then the bin directory
    of the running Python interpreter (covers uv tool / venv installs where
    the bin dir is not on PATH).

    Args:
        name: Binary name to search for on PATH.
        env_var: Environment variable that may hold an absolute path override.

    Returns:
        Absolute path to the binary, or None if not found.
    """
    override = os.environ.get(env_var)
    if override:
        return override
    on_path = shutil.which(name)
    if on_path:
        return on_path
    # MCP servers are often invoked without the venv on PATH; check the same
    # bin directory as the running interpreter before giving up.
    sibling = Path(sys.executable).parent / name
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None
