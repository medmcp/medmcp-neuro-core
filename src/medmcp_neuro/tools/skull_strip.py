"""Skull stripping tool using HD-BET."""

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools._neuro import cuda_unavailable_note, detect_devices, nii_stem


class DeviceChoiceResult(TypedDict):
    """Returned when the caller must confirm device and output path with the user before running."""

    available_devices: list[str]
    recommended_device: str
    brain_path: str
    _render: str


class SkullStripResult(TypedDict):
    """Successful skull stripping result."""

    brain_path: str
    input_path: str
    device: str
    _render: str


def skull_strip(
    input_path: Path,
    output_dir: Path | None = None,
    device: str | None = None,
) -> DeviceChoiceResult | SkullStripResult:
    """Extract brain from a structural NIfTI image using HD-BET.

    Runs HD-BET brain extraction on a 3-D NIfTI volume and writes the
    skull-stripped brain to ``output_dir`` with a ``_skullstripped`` suffix
    (e.g. ``sub-01_T1w.nii.gz`` → ``sub-01_T1w_skullstripped.nii.gz``).
    Test-time augmentation is automatically disabled on CPU (faster with
    negligible quality loss).

    When ``device`` is omitted, available devices are detected and the tool
    returns a ``DeviceChoiceResult`` so the caller can confirm the device and
    output path with the user before re-invoking with an explicit device.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
            Must be a 3-D volume; 4-D images are not supported by HD-BET.
            Use fslsplit to extract individual volumes from a 4-D series first.
        output_dir: Directory where the skull-stripped image is written.
            Defaults to the same directory as ``input_path``.
        device: Compute device — ``"cpu"``, ``"cuda"`` (any NVIDIA GPU), or
            ``"mps"`` (Apple Silicon). Omit to trigger device detection.

    Returns:
        ``SkullStripResult`` on success, or ``DeviceChoiceResult`` when the
        caller must confirm device selection with the user and re-invoke.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If HD-BET inference fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_dir = output_dir if output_dir is not None else input_path.parent
    stem = nii_stem(input_path)
    brain_path = out_dir / f"{stem}_skullstripped.nii.gz"

    if device is None:
        available = detect_devices()
        recommended = next(d for d in ("cuda", "mps", "cpu") if d in available)
        options = ", ".join(f'"{d}"' for d in available)
        duration_note = (
            " CPU inference takes ~2 minutes (TTA disabled)." + cuda_unavailable_note()
            if recommended == "cpu"
            else ""
        )
        choice: DeviceChoiceResult = {
            "available_devices": available,
            "recommended_device": recommended,
            "brain_path": str(brain_path),
            "_render": (
                f'Available devices: {options}. Recommended: "{recommended}".\n'
                f"Output will be written to: {brain_path}\n"
                f"NEXT ACTION: Tell the user the recommended device and output path."
                f"{duration_note} Ask them to confirm or choose a different device, "
                f"then call skull_strip again with the chosen device."
            ),
        }
        return choice

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
    proc = subprocess.run(
        [sys.executable, "-m", "medmcp_neuro.tools._run_hdbet"],
        input=json.dumps(
            {
                "device": device,
                "use_tta": device != "cpu",
                "input_path": str(input_path),
                "stem": stem,
                "brain_path": str(brain_path),
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

    stdout = proc.stdout.strip() if proc.stdout else ""
    run_result: dict[str, object] = json.loads(stdout) if stdout else {}
    if not run_result.get("ok"):
        raise RuntimeError(f"HD-BET failed: {run_result.get('error', 'unknown')}")

    if not brain_path.exists():
        raise RuntimeError(f"Skull stripping completed but output not found: {brain_path}")

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
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Tell the user the output path and ask what processing "
            "step to run next (e.g. registration to MNI, tissue segmentation). "
            "The tool already verified the file exists — do not attempt to recheck it."
        ),
    }
    return result
