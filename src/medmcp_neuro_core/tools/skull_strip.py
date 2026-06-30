"""Skull stripping tool using HD-BET."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro_core.tools._hdbet_worker import WorkerError, get_worker, persist_enabled
from medmcp_neuro_core.tools._neuro import Device, cuda_unavailable_note, nii_stem, resolve_device


class SkullStripResult(TypedDict):
    """Successful skull stripping result."""

    brain_path: str
    input_path: str
    device: str
    _render: str


class WarmupResult(TypedDict):
    """Result of pre-loading the HD-BET model into a persistent worker."""

    ok: bool
    device: str
    warmed: bool


def skull_strip(
    input_path: Path,
    output_dir: Path | None = None,
    device: Device = "auto",
) -> SkullStripResult:
    """Extract brain from a structural NIfTI image using HD-BET.

    Runs HD-BET brain extraction on a 3-D NIfTI volume and writes the
    skull-stripped brain to ``output_dir`` with a ``_skullstripped`` suffix
    (e.g. ``sub-01_T1w.nii.gz`` → ``sub-01_T1w_skullstripped.nii.gz``).

    The compute device follows the shared device convention: ``"auto"`` (default)
    selects the best available — CUDA (NVIDIA), then MPS (Apple Silicon), then CPU —
    or pass an explicit ``"cuda"`` / ``"mps"`` / ``"cpu"`` to override. The resolved
    device is reported in the result. On CPU, test-time augmentation is disabled
    (faster, negligible quality loss), so the mask can differ slightly by device.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
            Must be a 3-D volume; 4-D images are not supported by HD-BET.
            Use fslsplit to extract individual volumes from a 4-D series first.
        output_dir: Directory where the skull-stripped image is written.
            Defaults to the same directory as ``input_path``.
        device: Compute device — ``"auto"`` (default; cuda > mps > cpu), or force
            ``"cuda"`` / ``"mps"`` / ``"cpu"``.

    Returns:
        ``SkullStripResult`` with paths to the skull-stripped image.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If HD-BET inference fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    resolved = resolve_device(device)  # 'auto' -> cuda > mps > cpu; report the resolved one
    out_dir = output_dir if output_dir is not None else input_path.parent
    stem = nii_stem(input_path)
    brain_path = out_dir / f"{stem}_skullstripped.nii.gz"
    use_tta = resolved != "cpu"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[medmcp-neuro-core] skull_strip: starting HD-BET on {resolved}...",
        file=sys.stderr,
        flush=True,
    )

    # Fast path: a persistent worker holding the model warm (reused across calls).
    # Used when one is already warm (via the `warmup` hook) or when auto-start is
    # enabled. A dead/unavailable worker falls back to the per-call subprocess so
    # skull_strip never breaks; a genuine inference failure (RuntimeError) is not
    # retried.
    worker = get_worker(resolved, use_tta, start=persist_enabled())
    if worker is not None:
        try:
            worker.run(str(input_path), stem, str(brain_path))
        except WorkerError:
            _run_via_subprocess(resolved, use_tta, input_path, stem, brain_path)
    else:
        _run_via_subprocess(resolved, use_tta, input_path, stem, brain_path)

    if not brain_path.exists():
        raise RuntimeError(f"Skull stripping completed but output not found: {brain_path}")

    cuda_note = cuda_unavailable_note()
    device_note = (f"\n{cuda_note.strip()}") if resolved == "cpu" and cuda_note else ""
    result: SkullStripResult = {
        "brain_path": str(brain_path),
        "input_path": str(input_path),
        "device": resolved,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the skull stripping result as a compact key-value list:\n"
            "  Input:  <input_path>\n"
            "  Output: <brain_path>\n"
            "  Device: <device>\n"
            f"Substitute values from the result dict. Omit internal keys.{device_note}\n"
            "NEXT ACTION: Tell the user the output path and ask what processing "
            "step to run next (e.g. registration to MNI, tissue segmentation). "
            "The tool already verified the file exists — do not attempt to recheck it."
        ),
    }
    return result


def warmup(device: Device = "auto") -> WarmupResult:
    """Pre-load the HD-BET model so the first skull_strip is already warm.

    Called automatically when the stack is activated (the workspace pre-warm hook).
    Starts a persistent inference worker holding the model in memory; subsequent
    ``skull_strip`` calls on the same device reuse it instead of reloading, saving
    the model-load cost per call. Best-effort: returns ``warmed: false`` if the model
    can't load (e.g. no GPU), in which case ``skull_strip`` just loads lazily as before.

    ``device`` follows the shared convention and defaults to ``"auto"`` — the same
    default as ``skull_strip``, so the device warmed here matches the one a later
    ``skull_strip`` resolves to and the warm worker is reused.

    Args:
        device: Device to warm — ``"auto"`` (default; cuda > mps > cpu), or force
            ``"cuda"`` / ``"mps"`` / ``"cpu"``.

    Returns:
        ``WarmupResult`` indicating whether a warm worker is now ready.
    """
    resolved = resolve_device(device)
    worker = get_worker(resolved, resolved != "cpu", start=True)
    return {"ok": worker is not None, "device": resolved, "warmed": worker is not None}


def _run_via_subprocess(
    device: str, use_tta: bool, input_path: Path, stem: str, brain_path: Path
) -> None:
    """Run HD-BET in a throwaway subprocess (the fallback / default path).

    Isolates HD-BET from the MCP file descriptors: nnU-Net's multiprocessing
    workers would otherwise inherit and deadlock on the MCP pipe FDs. The result is
    written to a tempfile rather than stdout to avoid contamination from nnU-Net's
    own output bypassing the Python-level redirect.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        result_path = tf.name

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "medmcp_neuro_core.tools._run_hdbet"],
            input=json.dumps(
                {
                    "device": device,
                    "use_tta": use_tta,
                    "input_path": str(input_path),
                    "stem": stem,
                    "brain_path": str(brain_path),
                    "result_path": result_path,
                }
            ),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        if proc.returncode != 0:
            raise RuntimeError(f"HD-BET failed (exit {proc.returncode}): {proc.stderr.strip()}")

        try:
            with open(result_path) as f:
                run_result: dict[str, object] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            run_result = {}

        if not run_result.get("ok"):
            raise RuntimeError(f"HD-BET failed: {run_result.get('error', 'unknown')}")

    finally:
        Path(result_path).unlink(missing_ok=True)
