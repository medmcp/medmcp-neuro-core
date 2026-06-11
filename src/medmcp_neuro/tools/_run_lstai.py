"""Backend dispatch for LST-AI lesion segmentation.

LST-AI is heavyweight: it pins a specific HD-BET commit and depends on the
native ``greedy`` registration binary. Installing it into this package's
environment would clash with our own torch / HD-BET / ANTsPy versions, so it is
**never imported in-process**. Instead it is invoked as an external program
through one of two backends:

* ``native`` — the ``lst`` console script from a dedicated LST-AI virtualenv,
  located via ``$MEDMCP_LST_AI_BIN`` or PATH. The ``greedy`` binary is located
  via ``$MEDMCP_GREEDY_BIN`` or PATH, or downloaded once to the cache.
* ``docker`` — the ``jqmcginnis/lst-ai`` image, which bundles greedy + HD-BET.

This module only resolves the backend and builds the command line. The
side-effecting ``subprocess.run`` and output discovery live in
``lesion_segmentation.py`` so they remain easy to mock in tests.
"""

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

from medmcp_neuro.tools._neuro import find_binary

Backend = Literal["native", "docker"]

DEFAULT_IMAGE = "jqmcginnis/lst-ai:v1.2.0"
# Prebuilt greedy binary shipped with the LST-AI release; matches the version
# the native pipeline expects. Overridable via $MEDMCP_GREEDY_BIN.
_GREEDY_RELEASE_URL = "https://github.com/CompImg/LST-AI/releases/download/v1.0.0/greedy"
_BIN_CACHE_DIR = Path.home() / ".medmcp_neuro" / "bin"


def native_lst_bin() -> str | None:
    """Return the ``lst`` console-script path, or None if not installed."""
    return find_binary("lst", "MEDMCP_LST_AI_BIN")


def docker_image() -> str:
    """Return the LST-AI docker image, overridable via ``$MEDMCP_LST_AI_IMAGE``."""
    import os

    return os.environ.get("MEDMCP_LST_AI_IMAGE", DEFAULT_IMAGE)


def docker_available() -> bool:
    """Return True if a ``docker`` executable is on PATH."""
    return shutil.which("docker") is not None


def resolve_backend(requested: Backend | None) -> Backend:
    """Choose the LST-AI backend.

    Args:
        requested: Explicit backend, or None to auto-select. When None, the
            ``$MEDMCP_LST_AI_BACKEND`` env var (``native``/``docker``) is honoured
            if set; otherwise native is preferred with docker as fallback.

    Returns:
        The selected backend.

    Raises:
        RuntimeError: If the requested backend is unavailable, or if neither
            backend can be found during auto-selection. The message includes
            install guidance.
        ValueError: If ``$MEDMCP_LST_AI_BACKEND`` is set to an unknown value.
    """
    import os

    if requested is None:
        env_backend = os.environ.get("MEDMCP_LST_AI_BACKEND")
        if env_backend:
            if env_backend not in ("native", "docker"):
                raise ValueError(
                    f"MEDMCP_LST_AI_BACKEND must be 'native' or 'docker', got {env_backend!r}."
                )
            requested = env_backend  # type: ignore[assignment]

    if requested == "native":
        if native_lst_bin() is None:
            raise RuntimeError(_native_missing_msg())
        return "native"
    if requested == "docker":
        if not docker_available():
            raise RuntimeError(
                "Docker backend requested but the 'docker' executable is not on PATH."
            )
        return "docker"

    # auto: prefer native, fall back to docker
    if native_lst_bin() is not None:
        return "native"
    if docker_available():
        return "docker"
    raise RuntimeError(
        "LST-AI is not available via either backend.\n" + _native_missing_msg() + "\n"
        "Alternatively install Docker so the 'jqmcginnis/lst-ai' image can be used."
    )


def _native_missing_msg() -> str:
    return (
        "The 'lst' console script was not found. Install LST-AI into a dedicated\n"
        "virtualenv (kept separate from this package to avoid dependency conflicts):\n"
        "  python3 -m venv ~/.medmcp_neuro/lst-ai-venv\n"
        "  source ~/.medmcp_neuro/lst-ai-venv/bin/activate\n"
        "  pip install git+https://github.com/CompImg/LST-AI\n"
        "then point this tool at it with:\n"
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


def build_native_command(
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
    """Build the ``lst`` command line for the native backend."""
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
        cmd.append("--skull-stripped")
    cmd.extend(extra_args)
    return cmd


# Container mount targets used by the docker backend.
_C_T1_DIR = "/data/t1in"
_C_FLAIR_DIR = "/data/flairin"
_C_OUT_DIR = "/data/out"


def build_docker_command(
    *,
    image: str,
    t1_path: Path,
    flair_path: Path,
    output_dir: Path,
    device: str,
    skull_stripped: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the ``docker run`` command line for the docker backend.

    Each input's parent directory is bind-mounted read-only; the output
    directory is mounted read-write. Paths passed to LST-AI are rewritten to
    their in-container locations.
    """
    cmd = ["docker", "run", "--rm"]
    if device != "cpu":
        cmd += ["--gpus", "all"]
    cmd += [
        "-v",
        f"{t1_path.parent}:{_C_T1_DIR}:ro",
        "-v",
        f"{flair_path.parent}:{_C_FLAIR_DIR}:ro",
        "-v",
        f"{output_dir}:{_C_OUT_DIR}",
        image,
        "--t1",
        f"{_C_T1_DIR}/{t1_path.name}",
        "--flair",
        f"{_C_FLAIR_DIR}/{flair_path.name}",
        "--output",
        _C_OUT_DIR,
        "--device",
        device,
    ]
    if skull_stripped:
        cmd.append("--skull-stripped")
    cmd.extend(extra_args)
    return cmd
