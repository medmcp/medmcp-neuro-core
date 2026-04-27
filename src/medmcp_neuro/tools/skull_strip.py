"""Skull stripping tool using HD-BET."""

import subprocess
from pathlib import Path
from typing import Any

from medmcp_neuro.tools._neuro import find_binary


def _nii_stem(path: Path) -> str:
    """Return the NIfTI stem, stripping .nii.gz or .nii suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def skull_strip(
    input_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract brain from a structural NIfTI image using HD-BET.

    Runs HD-BET brain extraction on a 3-D NIfTI volume and writes the
    skull-stripped brain and a binary brain mask to ``output_dir``.
    Test-time augmentation is automatically disabled when ``device`` is
    ``"cpu"`` (faster with negligible quality loss on CPU).

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
            Must be a 3-D volume; 4-D images are not supported by HD-BET.
            Use fslsplit to extract individual volumes from a 4-D series first.
        output_dir: Directory where outputs are written. Created if absent.
        device: Compute device — ``"cpu"``, ``"cuda"`` (any NVIDIA GPU), or
            ``"mps"`` (Apple Silicon). Defaults to ``"cpu"``.

    Returns:
        Dict with keys ``brain_path``, ``mask_path``, ``input_path``,
        ``device``, and ``_render``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If the ``hd-bet`` binary is not found on PATH.
        subprocess.CalledProcessError: If HD-BET exits with a non-zero code.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    hd_bet = find_binary("hd-bet", "HD_BET_BINARY")
    if hd_bet is None:
        raise RuntimeError("hd-bet binary not found. Install with: pip install hd-bet")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _nii_stem(input_path)
    output_prefix = output_dir / f"{stem}_brain"
    brain_path = output_dir / f"{stem}_brain.nii.gz"
    mask_path = output_dir / f"{stem}_brain_mask.nii.gz"

    cmd: list[str] = [
        hd_bet,
        "-i",
        str(input_path),
        "-o",
        str(output_prefix),
        "-device",
        device,
        "--save_bet_mask",
    ]
    if device == "cpu":
        cmd.append("--disable_tta")

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return {
        "brain_path": str(brain_path),
        "mask_path": str(mask_path),
        "input_path": str(input_path),
        "device": device,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the skull stripping result as a compact key-value list:\n"
            "  Input:  <input_path>\n"
            "  Brain (skull-stripped): <brain_path>\n"
            "  Mask:   <mask_path>\n"
            "  Device: <device>\n"
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Confirm success with the user, then ask what "
            "processing step to run next (e.g. registration to MNI, "
            "tissue segmentation)."
        ),
    }
