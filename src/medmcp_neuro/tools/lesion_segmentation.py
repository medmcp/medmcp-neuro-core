"""White-matter lesion segmentation tool using LST-AI."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools import _run_lstai

_TIMEOUT = 7200  # LST-AI on CPU can run for well over an hour


class SegmentMSLesionsResult(TypedDict):
    """Successful lesion-segmentation result."""

    lesion_mask_path: str
    annotated_path: str | None
    output_files: list[str]
    stats_files: list[str]
    t1_path: str
    flair_path: str
    device: str
    skull_stripped: bool
    _render: str


def _nii_set(directory: Path) -> set[Path]:
    """Return the set of NIfTI files currently in ``directory`` (non-recursive)."""
    if not directory.exists():
        return set()
    return {p for p in directory.iterdir() if p.name.endswith((".nii", ".nii.gz"))}


def _csv_set(directory: Path) -> set[Path]:
    """Return the set of CSV files currently in ``directory`` (non-recursive)."""
    if not directory.exists():
        return set()
    return {p for p in directory.iterdir() if p.name.endswith(".csv")}


def _is_annotated(name: str) -> bool:
    """True if the filename is LST-AI's region-annotated segmentation."""
    low = name.lower()
    return "annotated" in low or "annot" in low or "label" in low


def _pick_segmentation(files: list[Path]) -> Path | None:
    """Pick the primary binary lesion mask from LST-AI's outputs.

    LST-AI writes ``space-flair_seg-lst.nii.gz`` (binary mask) and
    ``space-flair_desc-annotated_seg-lst.nii.gz`` (annotated) — both contain
    ``seg``, so the mask is the ``seg-lst`` file that is *not* annotated and not
    a probability map (``_prob``).
    """
    segs = [
        p
        for p in files
        if "seg-lst" in p.name.lower()
        and not _is_annotated(p.name)
        and "_prob" not in p.name.lower()
    ]
    if segs:
        return segs[0]
    # Fallback for unexpected naming: any non-annotated NIfTI.
    plain = [p for p in files if not _is_annotated(p.name)]
    return plain[0] if plain else (files[0] if files else None)


def _pick_annotated(files: list[Path]) -> Path | None:
    """Pick the region-annotated segmentation, if LST-AI produced one."""
    annotated = [p for p in files if _is_annotated(p.name)]
    return annotated[0] if annotated else None


def segment_ms_lesions(
    t1_path: Path,
    flair_path: Path,
    output_dir: Path | None = None,
    device: str = "cpu",
    gpu_id: int = 0,
    skull_stripped: bool = False,
) -> SegmentMSLesionsResult:
    """Segment white-matter (MS) lesions from paired T1w + FLAIR images using LST-AI.

    Requires **both** a T1w **and** a FLAIR of the same subject — it is not a
    single-contrast tool. If only one contrast is available, do not call this;
    tell the user LST-AI cannot run.

    LST-AI registers the FLAIR to the T1w and into MNI space, runs an ensemble
    segmentation network, and (unless restricted) annotates lesions by region.
    It is run out of process via its ``lst`` console script, which must be
    installed in its own virtualenv — LST-AI pins an old HD-BET that conflicts
    with this package's versions, so it is intentionally not a dependency here.
    Point ``$MEDMCP_LST_AI_BIN`` at that ``lst`` (or have it on PATH). The
    ``greedy`` registration binary is downloaded once on first use if absent.

    Skull stripping: LST-AI normally strips skulls itself with its own pinned
    HD-BET. If you have already run this package's ``skull_strip`` tool on **both**
    the T1w and FLAIR, pass ``skull_stripped=True`` so LST-AI skips that step
    (it passes ``--stripped``) — this avoids the bundled HD-BET and is the
    recommended path in this ecosystem.

    Before calling, ask the user which compute device to use:
    - ``"cpu"`` — always available, slow (often >1 h).
    - ``"cuda"`` — NVIDIA GPU (uses ``gpu_id``), much faster.
    LST-AI has no Apple-Silicon (MPS) path. Default to ``"cpu"`` if unspecified.

    Args:
        t1_path: Absolute path to the T1-weighted NIfTI (.nii or .nii.gz), 3-D.
        flair_path: Absolute path to the FLAIR NIfTI (.nii or .nii.gz), 3-D.
        output_dir: Directory for LST-AI outputs. Defaults to the T1w's directory.
        device: ``"cpu"`` (default) or ``"cuda"``.
        gpu_id: GPU index used when ``device="cuda"``. Defaults to ``0``.
        skull_stripped: Set True only when both inputs are already brain-extracted;
            passes ``--stripped`` to LST-AI.

    Returns:
        ``SegmentMSLesionsResult`` with the lesion mask path, the region-annotated
        map (if produced), and the full lists of NIfTI and CSV files LST-AI wrote.

    Raises:
        FileNotFoundError: If ``t1_path`` or ``flair_path`` does not exist.
        RuntimeError: If LST-AI is not installed or the run fails.
        ValueError: If ``device`` is not ``"cpu"`` or ``"cuda"``.
    """
    if not t1_path.exists():
        raise FileNotFoundError(f"T1w input not found: {t1_path}")
    if not flair_path.exists():
        raise FileNotFoundError(f"FLAIR input not found: {flair_path}")

    out_dir = output_dir if output_dir is not None else t1_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lst_bin = _run_lstai.require_lst_bin()
    dev_flag = _run_lstai.device_flag(device, gpu_id)
    greedy = _run_lstai.ensure_greedy()

    nii_before = _nii_set(out_dir)
    csv_before = _csv_set(out_dir)

    # LST-AI shells out to bare `hd-bet` and `greedy`, resolved via PATH. Put the
    # sidecar venv's bin FIRST so its (classic, pinned) HD-BET is used rather than
    # any host HD-BET, then the greedy cache dir.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(Path(lst_bin).parent), str(Path(greedy).parent), env.get("PATH", "")]
    )

    with tempfile.TemporaryDirectory(prefix="lst_temp_") as temp_dir:
        cmd = _run_lstai.build_command(
            lst_bin=lst_bin,
            t1_path=t1_path,
            flair_path=flair_path,
            output_dir=out_dir,
            temp_dir=Path(temp_dir),
            device=dev_flag,
            skull_stripped=skull_stripped,
            extra_args=[],
        )
        _invoke(cmd, env=env)

    produced = sorted(_nii_set(out_dir) - nii_before)
    if not produced:
        raise RuntimeError(f"LST-AI completed but no new NIfTI outputs were found in {out_dir}.")

    seg = _pick_segmentation(produced)
    annotated = _pick_annotated(produced)
    stats = sorted(_csv_set(out_dir) - csv_before)
    if seg is None:
        raise RuntimeError(f"Could not identify a lesion segmentation among: {produced}")

    return {
        "lesion_mask_path": str(seg),
        "annotated_path": str(annotated) if annotated else None,
        "output_files": [str(p) for p in produced],
        "stats_files": [str(p) for p in stats],
        "t1_path": str(t1_path),
        "flair_path": str(flair_path),
        "device": device,
        "skull_stripped": skull_stripped,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the lesion segmentation result as a compact key-value list:\n"
            "  Lesion mask: <lesion_mask_path>\n"
            "  Annotated:   <annotated_path>   (omit this line if null)\n"
            "  Stats:       <stats_files joined by ', '>   (omit if empty)\n"
            "  Device:      <device>\n"
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Tell the user the lesion mask path. If a T1w→template "
            "registration exists, offer to warp the lesion mask into MNI space with "
            "apply_transform using interpolation='NearestNeighbor'. The tool already "
            "verified the outputs exist — do not attempt to recheck them."
        ),
    }


def _invoke(cmd: list[str], *, env: dict[str, str]) -> None:
    """Run the LST-AI command, streaming stderr and raising on failure."""
    print("[medmcp-neuro] segment_ms_lesions: starting LST-AI …", file=sys.stderr, flush=True)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=_TIMEOUT,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError(
            f"LST-AI failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )


__all__ = ["SegmentMSLesionsResult", "segment_ms_lesions"]
