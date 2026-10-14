"""Through-plane super-resolution of anisotropic MRI with ECLARE or SMORE.

Both methods are *self-supervised*: there are no pretrained weights. Each call trains
a fresh network on the input volume itself — learning from its high-resolution
in-plane data what a high-resolution through-plane would look like — and then applies
it. That makes them slow (minutes to an hour per image on a GPU) but contrast- and
scanner-agnostic. They target 2D-acquired scans whose slices are much thicker than
the in-plane resolution (e.g. 0.9 x 0.9 x 5 mm); the output is resampled to the
in-plane resolution in all three axes.

* **ECLARE** (default) — Remedios et al., J. Med. Imaging 2026. Estimates the slice
  profile from the image (ESPRESO) unless a slice thickness is given. Upstream:
  https://github.com/sremedios/ECLARE (GPL-3.0), installed from PyPI (``eclare``).
* **SMORE** v4 — Zhao et al., IEEE TMI 2021 / Remedios et al., SASHIMI 2023. The
  predecessor of ECLARE from the same group; models the slice profile as an RF pulse
  of thickness = slice spacing unless told otherwise, and requires isotropic in-plane
  resolution. Upstream: https://gitlab.com/iacl/smore (Apache-2.0), tag v4.0.5.

The upstream code runs as its own console scripts (``run-eclare``; ``smore-train`` +
``smore-test``) in a subprocess. That keeps its DataLoader worker processes off the
MCP stdio pipes and keeps the GPL-licensed ECLARE a separate program rather than a
library linked into this package. ``run-smore`` is deliberately not used: it ignores
the exit codes of its train/test steps, so a failed training would go unnoticed.
"""

import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Literal, TypedDict

from medmcp_neuro_core.tools._neuro import (
    cuda_unavailable_note,
    detect_devices,
    find_binary,
    nii_stem,
)

SRMethod = Literal["eclare", "smore"]
# The shared Device convention minus "mps": upstream selects a CUDA GPU by index or
# runs on CPU, and has no Apple MPS path, so "mps" is not offered at all.
SRDevice = Literal["auto", "cuda", "cpu"]

# Upstream defaults for the number of training patches (run-eclare --n-patches /
# smore-train --n-patches). Reported back so the caller knows what actually ran.
_DEFAULT_N_PATCHES: dict[str, int] = {"eclare": 1_000_000, "smore": 832_000}

# Coarsest/finest voxel ratio below which there is nothing to super-resolve.
_ANISO_SKIP = 1.1
# ...and below which the expected gain is small enough to say so.
_ANISO_MILD = 1.5
# In-plane tolerance (mm) SMORE uses for "isotropic in-plane" (smore parse_image_file).
_SMORE_INPLANE_ATOL_MM = 1e-2
# Whole-call budget, matching the stack's tool_timeout_sec (7200 s).
_TIMEOUT_SEC = 7200
# Lines of upstream log surfaced on success (the full log is long: tqdm bars).
_LOG_TAIL_LINES = 15


class SuperResolveResult(TypedDict):
    """Super-resolution result (``output_path`` is ``None`` when skipped)."""

    output_path: str | None
    input_path: str
    method: str
    device: str
    skipped: bool
    input_voxel_size_mm: list[float]
    output_voxel_size_mm: list[float] | None
    input_shape: list[int]
    output_shape: list[int] | None
    through_plane_axis: int
    n_patches: int
    runtime_sec: float
    warnings: list[str]
    _render: str


def _read_geometry(path: Path) -> tuple[list[int], list[float]]:
    """Return (shape, voxel sizes in mm) from a NIfTI header without loading the data.

    Narrows to ``SpatialImage`` (which all NIfTIs are) so the rest of the module stays
    strictly typed — ``nib.load`` is untyped and the base class has no ``get_zooms``.

    Raises:
        ValueError: If the file is not a spatial image.
    """
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage

    img = nib.load(str(path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(img, SpatialImage):
        raise ValueError(f"{path} is not a spatial image (no voxel geometry)")
    shape = [int(s) for s in img.shape]
    zooms = [float(z) for z in img.header.get_zooms()]
    return shape, zooms


def _nii_ext(path: Path) -> str:
    """Return ``.nii.gz`` or ``.nii`` — upstream keeps the input's extension."""
    return ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"


def _resolve_sr_device(device: SRDevice) -> tuple[str, list[str]]:
    """Resolve the device for ECLARE/SMORE, which support CUDA or CPU only.

    Upstream selects a GPU by index and silently drops to CPU when CUDA is
    unavailable, so CUDA availability is checked here instead. ``"auto"`` never
    falls back to CPU: training a network per image on CPU takes many hours and would
    outlive the tool timeout, so CPU has to be asked for explicitly.

    Returns:
        (resolved device, warnings).

    Raises:
        RuntimeError: When CUDA is requested (or ``"auto"``) but unavailable.
    """
    if device == "cpu":
        return "cpu", [
            "Running on CPU: ECLARE/SMORE train a network on every image, which takes "
            "many hours on CPU and can exceed the 2-hour tool timeout. Prefer a CUDA GPU."
        ]
    if "cuda" not in detect_devices():
        raise RuntimeError(
            "No CUDA GPU available. Super-resolution trains a network on the image "
            "itself and is impractical on CPU (many hours); pass device='cpu' to run "
            f"anyway.{cuda_unavailable_note()}"
        )
    return "cuda", []


def _check_geometry(
    shape: list[int], zooms: list[float], method: SRMethod
) -> tuple[int, float, list[str]]:
    """Validate the input geometry and find the through-plane axis.

    Returns:
        (through-plane axis index, coarsest/finest voxel ratio, warnings).

    Raises:
        ValueError: If the image is not a 3-D, single-axis-anisotropic volume that the
            chosen method supports.
    """
    if len(shape) != 3:
        raise ValueError(
            f"Expected a 3-D volume, got shape {tuple(shape)}. Split 4-D series into "
            "3-D volumes first (ECLARE/SMORE super-resolve one volume at a time)."
        )
    zooms = zooms[:3]
    if any(z <= 0 for z in zooms):
        raise ValueError(f"Invalid voxel sizes {zooms} in the header.")

    axis = max(range(3), key=lambda i: zooms[i])
    inplane = [zooms[i] for i in range(3) if i != axis]
    ratio = zooms[axis] / min(inplane)
    warnings: list[str] = []
    if ratio < _ANISO_SKIP:
        return axis, ratio, warnings

    # Upstream asserts the coarsest spacing is unique: two equally thick axes are not
    # a 2D (single through-plane) acquisition.
    if max(inplane) / zooms[axis] > 0.99:
        raise ValueError(
            f"Voxel sizes {zooms} have two equally coarse axes; ECLARE/SMORE need one "
            "through-plane axis that is coarser than both in-plane axes."
        )
    inplane_diff = abs(inplane[0] - inplane[1])
    if method == "smore" and inplane_diff > _SMORE_INPLANE_ATOL_MM:
        raise ValueError(
            f"SMORE requires isotropic in-plane resolution, got {inplane[0]:g} x "
            f"{inplane[1]:g} mm. Use method='eclare', which supports this."
        )
    if method == "eclare" and max(inplane) / min(inplane) > _ANISO_SKIP:
        warnings.append(
            f"In-plane resolution is anisotropic ({inplane[0]:g} x {inplane[1]:g} mm); "
            "ECLARE super-resolves the through-plane axis to the finer in-plane spacing "
            f"({min(inplane):g} mm) and leaves the in-plane axes unchanged."
        )
    if ratio < _ANISO_MILD:
        warnings.append(
            f"Only mild anisotropy ({ratio:.2f}x); the gain from super-resolution will be small."
        )
    return axis, ratio, warnings


def _find_cli(name: str, env_var: str) -> str:
    """Locate an upstream console script (installed into the app venv)."""
    binary = find_binary(name, env_var)
    if binary is None:
        raise RuntimeError(
            f"{name} not found. It is installed with the medmcp-neuro-core package "
            f"(see pyproject.toml); set ${env_var} to override its location."
        )
    return binary


def _run_cli(cmd: list[str], log_path: Path, cwd: Path, timeout: float, label: str) -> None:
    """Run one upstream CLI step, logging stdout+stderr to ``log_path``.

    The log goes to a file rather than a pipe: training emits a long tqdm stream, and
    a file needs no draining. ``TQDM_MININTERVAL`` thins the progress bars to one
    update every 30 s. On failure the tail of the log is raised.

    Raises:
        RuntimeError: On a non-zero exit or timeout.
    """
    env = {**os.environ, "TQDM_MININTERVAL": "30", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    print(f"[medmcp-neuro-core] super_resolve: {label} ...", file=sys.stderr, flush=True)
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{label} timed out after {timeout:.0f} s. Try a smaller n_patches."
            ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}):\n{_log_tail(log_path, 40)}")


def _log_tail(log_path: Path, n: int) -> str:
    """Return the last ``n`` non-empty lines of the log, splitting tqdm's CR updates."""
    with open(log_path, errors="replace") as fh:
        lines = (seg.rstrip() for line in fh for seg in line.split("\r"))
        return "\n".join(deque((ln for ln in lines if ln), maxlen=n))


def _eclare_cmd(
    input_path: Path,
    work_dir: Path,
    gpu_id: str,
    relative_thickness: float | None,
    n_patches: int | None,
) -> list[str]:
    cmd = [
        _find_cli("run-eclare", "RUN_ECLARE"),
        "--in-fpath",
        str(input_path),
        "--out-dir",
        str(work_dir),
        "--gpu-id",
        gpu_id,
        "--suffix",
        "_eclare",
    ]
    if relative_thickness is not None:
        cmd += ["--relative-slice-thickness", f"{relative_thickness:.6g}"]
    if n_patches is not None:
        cmd += ["--n-patches", str(n_patches)]
    return cmd


def _smore_cmds(
    input_path: Path,
    work_dir: Path,
    sr_path: Path,
    gpu_id: str,
    slice_thickness_mm: float | None,
    n_patches: int | None,
) -> tuple[list[str], list[str]]:
    weight_dir = work_dir / "weights"
    train = [
        _find_cli("smore-train", "SMORE_TRAIN"),
        "--in-fpath",
        str(input_path),
        "--weight-dir",
        str(weight_dir),
        "--gpu-id",
        gpu_id,
        "--verbose",
    ]
    if slice_thickness_mm is not None:
        train += ["--slice-thickness", f"{slice_thickness_mm:.6g}"]
    if n_patches is not None:
        train += ["--n-patches", str(n_patches)]
    test = [
        _find_cli("smore-test", "SMORE_TEST"),
        "--in-fpath",
        str(input_path),
        "--out-fpath",
        str(sr_path),
        "--weight-dir",
        str(weight_dir),
        "--gpu-id",
        gpu_id,
    ]
    return train, test


def _fmt_mm(zooms: list[float]) -> str:
    return " x ".join(f"{z:.3g}" for z in zooms) + " mm"


def _render(result_skipped: bool, warnings: list[str]) -> str:
    warn = "Surface every entry in 'warnings' to the user FIRST.\n" if warnings else ""
    if result_skipped:
        return (
            "DISPLAY RULES — follow exactly:\n"
            f"{warn}"
            "Tell the user the input is already (near-)isotropic, so super-resolution was "
            "skipped and no file was written. Report input_voxel_size_mm.\n"
            "NEXT ACTION: Ask what to do next; do not retry super_resolve on this image."
        )
    return (
        "DISPLAY RULES — follow exactly:\n"
        f"{warn}"
        "Report the result as a compact key-value list:\n"
        "  Input:   <input_path>  (<input_voxel_size_mm> mm, shape <input_shape>)\n"
        "  Output:  <output_path>  (<output_voxel_size_mm> mm, shape <output_shape>)\n"
        "  Method:  <method>\n"
        "  Device:  <device>\n"
        "  Runtime: <runtime_sec> s\n"
        "Substitute values from the result dict. Omit internal keys.\n"
        "NEXT ACTION: Tell the user the output path. Remind them the through-plane detail "
        "is synthesized by a network trained on this image (good for viewing and "
        "isotropic downstream processing, not a substitute for a true 3D acquisition). "
        "Offer a next step such as segment_brain or registration on the super-resolved "
        "image. The tool already verified the file exists — do not recheck it."
    )


def super_resolve(
    input_path: Path,
    method: SRMethod = "eclare",
    output_dir: Path | None = None,
    device: SRDevice = "auto",
    slice_thickness_mm: float | None = None,
    n_patches: int | None = None,
) -> SuperResolveResult:
    """Super-resolve an anisotropic (thick-slice, 2D-acquired) MRI volume to isotropic.

    Load the ``super-resolution`` skill before calling this tool: it explains which
    method to pick and how long the run takes, which the user should hear first.

    Self-supervised — no pretrained weights and no training data: a network is
    trained on the input image itself (from its high-resolution in-plane slices) and
    then applied to the through-plane axis, so every call takes minutes to about an
    hour on a GPU. The through-plane axis (the coarsest voxel spacing) is resampled to
    the finest in-plane spacing; in-plane is unchanged. Inputs that are already
    (near-)isotropic are skipped with a warning instead of run.

    Output is written to ``output_dir`` (defaults to the input's directory) as
    ``{stem}_eclare.nii.gz`` or ``{stem}_smore.nii.gz`` (``.nii`` for ``.nii`` input),
    e.g. ``sub-01_FLAIR.nii.gz`` -> ``sub-01_FLAIR_eclare.nii.gz``.

    Args:
        input_path: Absolute path to a 3-D NIfTI (.nii or .nii.gz) with one coarse
            through-plane axis, e.g. 1 x 1 x 5 mm.
        method: ``"eclare"`` (default; newer, estimates the slice profile from the
            image, tolerates anisotropic in-plane resolution) or ``"smore"`` (SMORE v4,
            the predecessor; requires isotropic in-plane resolution).
        output_dir: Directory for the output. Defaults to ``input_path.parent``.
        device: ``"auto"`` (default; the CUDA GPU), ``"cuda"``, or ``"cpu"`` (many
            hours — must be requested explicitly). Apple MPS is not supported upstream.
        slice_thickness_mm: Acquired slice thickness in mm, if known (often differs
            from the slice spacing in the header, e.g. 5 mm slices with a 1 mm gap
            have 6 mm spacing). Omit to let ECLARE estimate the slice profile from the
            image, or for SMORE to assume thickness = spacing.
        n_patches: Number of training patches. Omit for the upstream default
            (ECLARE 1,000,000; SMORE 832,000). Lower is faster but lower quality —
            only reduce it for a quick preview.

    Returns:
        ``SuperResolveResult`` with the output path, the voxel sizes and shapes before
        and after, the resolved device, runtime, and any warnings.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If the geometry is unsupported (not 3-D, two coarse axes, or
            anisotropic in-plane for SMORE), or a parameter is out of range.
        RuntimeError: If no GPU is available for ``"auto"``/``"cuda"``, or the
            upstream run fails or times out.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if slice_thickness_mm is not None and slice_thickness_mm <= 0:
        raise ValueError(f"slice_thickness_mm must be positive, got {slice_thickness_mm}")
    if n_patches is not None and n_patches < 1000:
        raise ValueError(f"n_patches must be at least 1000, got {n_patches}")

    start = time.monotonic()
    shape, zooms = _read_geometry(input_path)
    axis, ratio, warnings = _check_geometry(shape, zooms, method)
    zooms = zooms[:3]
    effective_patches = n_patches if n_patches is not None else _DEFAULT_N_PATCHES[method]

    if ratio < _ANISO_SKIP:
        warnings.append(
            f"Input voxel size {_fmt_mm(zooms)} is already (near-)isotropic "
            f"(anisotropy {ratio:.2f}x); nothing to super-resolve, skipped."
        )
        return {
            "output_path": None,
            "input_path": str(input_path),
            "method": method,
            "device": "none",
            "skipped": True,
            "input_voxel_size_mm": zooms,
            "output_voxel_size_mm": None,
            "input_shape": shape,
            "output_shape": None,
            "through_plane_axis": axis,
            "n_patches": 0,
            "runtime_sec": round(time.monotonic() - start, 1),
            "warnings": warnings,
            "_render": _render(True, warnings),
        }

    resolved, device_warnings = _resolve_sr_device(device)
    warnings += device_warnings
    for warning in warnings:
        print(f"[medmcp-neuro-core] super_resolve: WARNING: {warning}", file=sys.stderr)
    gpu_id = "0" if resolved == "cuda" else "-1"  # upstream: -1 = CPU

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _nii_ext(input_path)
    final_path = out_dir / f"{nii_stem(input_path)}_{method}{ext}"

    # Work beside the output (same filesystem, so the final move is a rename, and the
    # weights/intermediates of a large volume never fill /tmp).
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".super_resolve-") as tmp:
        work_dir = Path(tmp)
        log_path = work_dir / "upstream.log"
        if method == "eclare":
            relative = None
            if slice_thickness_mm is not None:
                inplane = min(zooms[i] for i in range(3) if i != axis)
                relative = slice_thickness_mm / inplane  # ECLARE wants it per in-plane voxel
            sr_path = work_dir / input_path.name.replace(".nii", "_eclare.nii")
            cmd = _eclare_cmd(input_path, work_dir, gpu_id, relative, n_patches)
            _run_cli(cmd, log_path, work_dir, _TIMEOUT_SEC, f"ECLARE on {resolved}")
        else:
            sr_path = work_dir / f"sr{ext}"
            train, test = _smore_cmds(
                input_path, work_dir, sr_path, gpu_id, slice_thickness_mm, n_patches
            )
            _run_cli(train, log_path, work_dir, _TIMEOUT_SEC, f"SMORE training on {resolved}")
            remaining = max(60.0, _TIMEOUT_SEC - (time.monotonic() - start))
            _run_cli(test, log_path, work_dir, remaining, f"SMORE inference on {resolved}")

        if not sr_path.exists():
            raise RuntimeError(f"{method} finished but wrote no output:\n{_log_tail(log_path, 40)}")
        sys.stderr.write(_log_tail(log_path, _LOG_TAIL_LINES) + "\n")
        sys.stderr.flush()
        os.replace(sr_path, final_path)

    out_shape, out_zooms = _read_geometry(final_path)
    runtime = round(time.monotonic() - start, 1)
    return {
        "output_path": str(final_path),
        "input_path": str(input_path),
        "method": method,
        "device": resolved,
        "skipped": False,
        "input_voxel_size_mm": zooms,
        "output_voxel_size_mm": out_zooms[:3],
        "input_shape": shape,
        "output_shape": out_shape,
        "through_plane_axis": axis,
        "n_patches": effective_patches,
        "runtime_sec": runtime,
        "warnings": warnings,
        "_render": _render(False, warnings),
    }
