"""Shared internal utilities for medmcp-neuro-core tools."""

from __future__ import annotations

import os
import shutil
import sys
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Shared device convention: every GPU-capable tool accepts one of these (default "auto").
Device = Literal["auto", "cuda", "mps", "cpu"]


def nii_stem(path: Path) -> str:
    """Return the NIfTI stem, stripping .nii.gz or .nii suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def require_image(path: Path, role: str = "Input") -> None:
    """Fail early on an image that is missing, truncated or no readable volume.

    Reads the voxel data once: a truncated ``.nii.gz`` keeps a valid header, so only
    the data shows the damage, and HD-BET would return an empty brain from it.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be read as an image (truncated or corrupt).
    """
    if not path.exists():
        raise FileNotFoundError(f"{role} not found: {path}")
    import nibabel as nib
    import numpy as np
    from nibabel.filebasedimages import ImageFileError
    from nibabel.spatialimages import SpatialImage

    try:
        img = nib.load(str(path))  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(img, SpatialImage):
            raise ValueError("no voxel geometry")
        np.asanyarray(img.dataobj)
    except (OSError, EOFError, ValueError, zlib.error, ImageFileError) as exc:
        msg = f"{role} is unreadable: the file is truncated or corrupt ({exc}): {path}"
        raise ValueError(msg) from exc


def nifti_load(
    path: Path,
    role: str = "Input",
    *,
    dtype: Any | None = None,
    require_3d: bool = True,
) -> tuple[NDArray[Any], Any, tuple[float, float, float]]:
    """Load a NIfTI or MGZ spatial image and return ``(data, affine, zooms)``.

    Validates readability with :func:`require_image`. When ``require_3d`` is True
    (default), squeezes a trailing singleton 4th dimension ``(X, Y, Z, 1)`` and rejects
    other non-3-D shapes. When ``dtype`` is an integer numpy dtype, rounds floating-point
    voxel values via ``np.rint`` before casting.

    Args:
        path: Path to the NIfTI (``.nii``, ``.nii.gz``) or MGZ (``.mgz``) image.
        role: Human-readable role label used in error messages.
        dtype: Optional numpy dtype to cast the voxel array to.
        require_3d: Enforce a 3-D spatial volume (squeezing ``(X, Y, Z, 1)``).

    Returns:
        Tuple of ``(voxel_array, affine_4x4, (zoom_x, zoom_y, zoom_z))``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the image is corrupt, has no spatial geometry, or fails ``require_3d``.
    """
    import nibabel as nib
    import numpy as np
    from nibabel.spatialimages import SpatialImage

    require_image(path, role=role)
    img = nib.load(str(path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(img, SpatialImage):
        raise ValueError(f"{role} is not a spatial image: {path}")

    raw: Any = np.asarray(img.dataobj)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    if require_3d:
        if raw.ndim == 4 and raw.shape[3] == 1:
            raw = raw[..., 0]
        if raw.ndim != 3:
            raise ValueError(f"{role} must be a 3-D volume, got shape {raw.shape}: {path}")

    if dtype is not None:
        if np.issubdtype(dtype, np.integer):
            arr: NDArray[Any] = np.rint(np.asarray(raw, dtype=np.float64)).astype(dtype)
        else:
            arr = np.asarray(raw, dtype=dtype)
    else:
        arr = np.asarray(raw)

    raw_zooms = img.header.get_zooms()[:3]  # pyright: ignore[reportUnknownMemberType]
    zooms = (float(raw_zooms[0]), float(raw_zooms[1]), float(raw_zooms[2]))
    return arr, img.affine, zooms


def nifti_save(
    arr: NDArray[Any],
    affine: Any,
    path: Path,
    zooms: tuple[float, float, float] | None = None,
) -> Path:
    """Save a numpy array as a NIfTI-1 file preserving affine and optional voxel spacing.

    Creates parent directories if absent and writes ``path``.

    Args:
        arr: Voxel array to write.
        affine: 4x4 affine matrix.
        path: Destination NIfTI file path (``.nii`` or ``.nii.gz``).
        zooms: Optional 3-element voxel spacing tuple ``(dx, dy, dz)`` in mm.

    Returns:
        The written ``path``.
    """
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(arr, affine)
    if zooms is not None:
        out_img.header.set_zooms(zooms)
    nib.save(out_img, str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


def is_same_grid(
    shape_a: tuple[int, ...], aff_a: Any, shape_b: tuple[int, ...], aff_b: Any
) -> bool:
    """Return True if two volumes share the same 3-D spatial shape and affine."""
    import numpy as np

    return tuple(shape_a[:3]) == tuple(shape_b[:3]) and bool(np.allclose(aff_a, aff_b, atol=1e-3))


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


def resolve_device(device: Device) -> str:
    """Resolve a requested compute device to a concrete one (the shared device convention).

    Every GPU-capable tool accepts ``device`` as ``"auto"`` (the default), ``"cuda"``,
    ``"mps"``, or ``"cpu"`` and resolves it through this helper. ``"auto"`` selects the
    best available accelerator — CUDA (NVIDIA), then MPS (Apple Silicon), then CPU — from
    what torch reports; an explicit device is returned unchanged. Tools should run on, and
    report, the *resolved* device so an ``"auto"`` → CPU fallback is never silent.

    Args:
        device: ``"auto"``, or an explicit ``"cuda"`` / ``"mps"`` / ``"cpu"``.

    Returns:
        A concrete device string ('cuda', 'mps', or 'cpu').
    """
    if device != "auto":
        return device
    available = detect_devices()
    if "cuda" in available:
        return "cuda"
    if "mps" in available:
        return "mps"
    return "cpu"


def cuda_unavailable_note() -> str:
    """Return an actionable note if a CUDA-capable torch is installed but CUDA is unavailable.

    Returns:
        Human-readable note with install guidance, or empty string if CUDA works or is absent.
    """
    import torch

    if torch.cuda.is_available():
        return ""
    # torch.version isn't exposed in torch's type stubs; reach it via getattr so strict
    # pyright stays happy (the attribute is the documented way to read the CUDA build).
    cuda_ver: str | None = getattr(getattr(torch, "version", None), "cuda", None)
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
