"""Skull stripping tool using HD-BET."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools._neuro import cuda_unavailable_note, nii_stem


class SkullStripResult(TypedDict):
    """Successful skull stripping result."""

    brain_path: str
    input_path: str
    device: str
    _render: str


def skull_strip(
    input_path: Path,
    output_dir: Path | None = None,
    device: str = "cpu",
) -> SkullStripResult:
    """Extract brain from a structural NIfTI image using HD-BET.

    Runs HD-BET brain extraction on a 3-D NIfTI volume and writes the
    skull-stripped brain to ``output_dir`` with a ``_skullstripped`` suffix
    (e.g. ``sub-01_T1w.nii.gz`` → ``sub-01_T1w_skullstripped.nii.gz``).
    Test-time augmentation is automatically disabled on CPU (faster with
    negligible quality loss).

    Before calling, ask the user which compute device to use:
    - ``"cpu"`` — always available, ~2 minutes (TTA disabled).
    - ``"cuda"`` — NVIDIA GPU, fastest; only available if a CUDA-enabled torch is installed.
    - ``"mps"`` — Apple Silicon GPU; only available on macOS with Apple Silicon.
    Default to ``"cpu"`` if the user does not specify.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
            Must be a 3-D volume; 4-D images are not supported by HD-BET.
            Use fslsplit to extract individual volumes from a 4-D series first.
        output_dir: Directory where the skull-stripped image is written.
            Defaults to the same directory as ``input_path``.
        device: Compute device — ``"cpu"`` (default), ``"cuda"``, or ``"mps"``.

    Returns:
        ``SkullStripResult`` with paths to the skull-stripped image.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If HD-BET inference fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_dir = output_dir if output_dir is not None else input_path.parent
    stem = nii_stem(input_path)
    brain_path = out_dir / f"{stem}_skullstripped.nii.gz"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[medmcp-neuro] skull_strip: starting HD-BET on {device}...",
        file=sys.stderr,
        flush=True,
    )

    # Run HD-BET in a subprocess with its own stdin/stdout pipes (via input= and
    # capture_output=True) so the MCP file descriptors are not inherited by
    # nnU-Net's multiprocessing workers.  Without isolation, the Manager().Queue
    # workers deadlock because they inherit the MCP pipe FDs.
    # The result is written to a tempfile rather than stdout to avoid
    # contamination from nnU-Net's own output bypassing the Python-level redirect.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        result_path = tf.name

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "medmcp_neuro.tools._run_hdbet"],
            input=json.dumps(
                {
                    "device": device,
                    "use_tta": device != "cpu",
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

    if not brain_path.exists():
        raise RuntimeError(f"Skull stripping completed but output not found: {brain_path}")

    cuda_note = cuda_unavailable_note()
    device_note = (f"\n{cuda_note.strip()}") if device == "cpu" and cuda_note else ""
    result: SkullStripResult = {
        "brain_path": str(brain_path),
        "input_path": str(input_path),
        "device": device,
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
