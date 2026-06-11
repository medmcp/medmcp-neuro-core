"""Helpers for invoking LST-AI lesion segmentation.

LST-AI is heavyweight: it pins a specific (old) HD-BET commit and depends on the
native ``greedy`` registration binary. Installing it into this package's
environment would clash with our own torch / HD-BET / ANTsPy versions, so it is
**never imported in-process**. Instead it lives in its own dedicated virtualenv
and is invoked as an external program through its ``lst`` console script — the
same subprocess pattern the other tools use, just pointed at a sibling venv.

This module locates the ``lst`` and ``greedy`` binaries and builds the command
line. The side-effecting ``subprocess.run`` and output discovery live in
``lesion_segmentation.py`` so they remain easy to mock in tests.
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

from medmcp_neuro.tools._neuro import find_binary

# Prebuilt greedy binary shipped with the LST-AI release. Overridable via
# $MEDMCP_GREEDY_BIN.
_GREEDY_RELEASE_URL = "https://github.com/CompImg/LST-AI/releases/download/v1.0.0/greedy"
_BIN_CACHE_DIR = Path.home() / ".medmcp_neuro" / "bin"


def native_lst_bin() -> str | None:
    """Return the ``lst`` console-script path, or None if not installed.

    Looks at ``$MEDMCP_LST_AI_BIN`` first, then PATH, then the running
    interpreter's bin directory.
    """
    return find_binary("lst", "MEDMCP_LST_AI_BIN")


def require_lst_bin() -> str:
    """Return the ``lst`` console-script path or raise with install guidance."""
    found = native_lst_bin()
    if found is None:
        raise RuntimeError(_install_msg())
    return found


def _install_msg() -> str:
    return (
        "LST-AI is not installed. Install it into a dedicated virtualenv (kept\n"
        "separate from this package to avoid HD-BET / torch version conflicts):\n"
        "  python3 -m venv ~/.medmcp_neuro/lst-ai-venv\n"
        "  source ~/.medmcp_neuro/lst-ai-venv/bin/activate\n"
        "  pip install git+https://github.com/CompImg/LST-AI\n"
        "then point this tool at it:\n"
        "  export MEDMCP_LST_AI_BIN=~/.medmcp_neuro/lst-ai-venv/bin/lst"
    )


def device_flag(device: str, gpu_id: int) -> str:
    """Translate this package's device convention to LST-AI's ``--device`` value.

    LST-AI accepts ``cpu`` or an integer GPU id (it has no MPS path).

    Args:
        device: ``"cpu"`` or ``"cuda"``.
        gpu_id: GPU index used when ``device == "cuda"``.

    Returns:
        ``"cpu"`` or the GPU id as a string.

    Raises:
        ValueError: If ``device`` is not ``"cpu"`` or ``"cuda"``.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return str(gpu_id)
    raise ValueError(
        f"LST-AI supports device 'cpu' or 'cuda', got {device!r} (MPS is unsupported)."
    )


def ensure_greedy() -> str:
    """Locate the ``greedy`` binary, downloading the release build on first use.

    Resolution order: ``$MEDMCP_GREEDY_BIN`` / PATH, then the package bin cache,
    then download from the LST-AI release into the cache.

    Returns:
        Absolute path to an executable ``greedy``.
    """
    found = find_binary("greedy", "MEDMCP_GREEDY_BIN")
    if found:
        return found

    cached = _BIN_CACHE_DIR / "greedy"
    if cached.is_file():
        return str(cached)

    _BIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[medmcp-neuro] Downloading greedy to {cached} …",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.NamedTemporaryFile(dir=_BIN_CACHE_DIR, suffix=".tmp", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        urllib.request.urlretrieve(_GREEDY_RELEASE_URL, tmp)
        tmp.chmod(0o755)
        tmp.rename(cached)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return str(cached)


def build_command(
    *,
    lst_bin: str,
    t1_path: Path,
    flair_path: Path,
    output_dir: Path,
    temp_dir: Path,
    device: str,
    skull_stripped: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the ``lst`` command line.

    Note: LST-AI's already-skull-stripped flag is ``--stripped`` (not
    ``--skull-stripped``, which appears in some older docs).
    """
    cmd = [
        lst_bin,
        "--t1",
        str(t1_path),
        "--flair",
        str(flair_path),
        "--output",
        str(output_dir),
        "--temp",
        str(temp_dir),
        "--device",
        device,
    ]
    if skull_stripped:
        cmd.append("--stripped")
    cmd.extend(extra_args)
    return cmd
