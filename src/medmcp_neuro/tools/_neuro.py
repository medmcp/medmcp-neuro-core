"""Shared internal utilities for medmcp-neuro tools."""

import os
import shutil


def find_binary(name: str, env_var: str) -> str | None:
    """Locate a neuroimaging tool binary.

    Checks the environment variable first (allows path override without
    modifying PATH), then falls back to shutil.which.

    Args:
        name: Binary name to search for on PATH.
        env_var: Environment variable that may hold an absolute path override.

    Returns:
        Absolute path to the binary, or None if not found.
    """
    override = os.environ.get(env_var)
    if override:
        return override
    return shutil.which(name)
