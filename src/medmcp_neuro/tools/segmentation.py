"""Brain segmentation tool using FastSurfer (FastSurferVINN, segmentation-only).

Runs FastSurfer's deep-learning whole-brain segmentation (``aparc.DKTatlas+aseg``)
*without* the surface pipeline, so **no FreeSurfer license is required** (the
``--fs_license`` flag only gates surface reconstruction, which we do not run).

Produces a discrete label map (FreeSurfer ``.mgz``, rendered natively by the
workspace viewer) plus a per-structure volume CSV derived from FastSurfer's own
``segstats`` (no ``mri_segstats``/FreeSurfer dependency). GPU-accelerated through
torch (CUDA, or Apple MPS); falls back to CPU when no accelerator is present.

FastSurfer shares FreeSurfer's label conventions, so the structure names below
match the aseg / DKT-atlas names that appear in the stats output.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools._neuro import cuda_unavailable_note, detect_devices, find_binary, nii_stem

# Subcortical + global structures always present in the aseg/DKT stats output.
_SUBCORTICAL_LABELS: list[str] = [
    "total intracranial",
    "csf",
    "brain-stem",
    "left cerebral white matter",
    "left cerebral cortex",
    "right cerebral white matter",
    "right cerebral cortex",
    "left cerebellum white matter",
    "left cerebellum cortex",
    "right cerebellum white matter",
    "right cerebellum cortex",
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "3rd ventricle",
    "4th ventricle",
    "left thalamus",
    "right thalamus",
    "left caudate",
    "right caudate",
    "left putamen",
    "right putamen",
    "left pallidum",
    "right pallidum",
    "left hippocampus",
    "right hippocampus",
    "left amygdala",
    "right amygdala",
    "left accumbens area",
    "right accumbens area",
    "left ventral DC",
    "right ventral DC",
]

# Cortical parcels (Desikan-Killiany-Tourville), reported per hemisphere as
# ctx-lh-* / ctx-rh-* in the stats output.
_CORTICAL_PARCEL_NAMES: list[str] = [
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]


class SegmentResult(TypedDict):
    """Brain segmentation result."""

    seg_path: str
    volumes_path: str
    input_path: str
    device: str
    _render: str


class LabelListResult(TypedDict):
    """Available segmentation labels for a given FastSurfer configuration."""

    parc: bool
    total_structures: int
    subcortical_and_global: list[str]
    cortical_parcels: list[str]
    _render: str


def _find_fastsurfer() -> str:
    """Return the absolute path to FastSurfer's ``run_fastsurfer.sh``.

    Checks, in order: $RUN_FASTSURFER override, $FASTSURFER_HOME/run_fastsurfer.sh,
    PATH, and common install locations. The Docker image installs FastSurfer at
    /opt/FastSurfer with FASTSURFER_HOME set.

    Raises:
        RuntimeError: If run_fastsurfer.sh cannot be located.
    """
    binary = find_binary("run_fastsurfer.sh", "RUN_FASTSURFER")
    if binary is None:
        for home in [
            os.environ.get("FASTSURFER_HOME"),
            "/opt/FastSurfer",
            str(Path.home() / "FastSurfer"),
        ]:
            if not home:
                continue
            candidate = Path(home) / "run_fastsurfer.sh"
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return str(candidate)
        raise RuntimeError(
            "run_fastsurfer.sh not found. The medmcp-neuro image installs FastSurfer "
            "at /opt/FastSurfer; set $FASTSURFER_HOME or $RUN_FASTSURFER otherwise."
        )
    return binary


def _resolve_device(device: str) -> str:
    """Map a requested device to one FastSurfer accepts ('cuda' | 'mps' | 'cpu').

    'auto' picks cuda, then mps, then cpu based on what torch reports available.
    """
    if device != "auto":
        return device
    available = detect_devices()
    if "cuda" in available:
        return "cuda"
    if "mps" in available:
        return "mps"
    return "cpu"


def _parse_aseg_stats(stats_path: Path) -> list[tuple[str, float]]:
    """Parse a FreeSurfer/FastSurfer .stats file into (structure, volume_mm3) rows.

    Stats data lines are whitespace-separated with the columns described in the
    ``# ColHeaders`` line; volume is ``Volume_mm3`` and the name is ``StructName``.
    Comment lines start with '#'.
    """
    rows: list[tuple[str, float]] = []
    with open(stats_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            # ColHeaders: Index SegId NVoxels Volume_mm3 StructName ...
            if len(cols) < 5:
                continue
            try:
                volume = float(cols[3])
            except ValueError:
                continue
            rows.append((cols[4], volume))
    return rows


def segment_brain(
    input_path: Path,
    output_dir: Path | None = None,
    device: str = "auto",
    threads: int = 4,
) -> SegmentResult:
    """Segment brain structures from a NIfTI image using FastSurfer.

    Runs FastSurfer's deep-learning segmentation (FastSurferVINN, ``--seg_only``)
    to label 95 cortical and subcortical structures, including left/right thalamus
    and the other subcortical volumes. No FreeSurfer license is required. Accepts
    full-head or skull-stripped T1w images; skull stripping is not required first.

    Outputs a discrete label map (``.mgz``) and a per-structure volume CSV. Call
    list_brain_segmentation_labels() to see the structures and CSV column names.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
        output_dir: Directory for outputs. Defaults to input_path's directory.
        device: 'auto' (default; cuda > mps > cpu), or force 'cuda' / 'mps' / 'cpu'.
        threads: CPU threads for pre/post-processing. Default 4.

    Returns:
        SegmentResult with paths to the segmentation label map and volume CSV.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError: If FastSurfer is not installed or segmentation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    binary = _find_fastsurfer()
    resolved_device = _resolve_device(device)

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nii_stem(input_path)
    seg_path = out_dir / f"{stem}_dseg.mgz"
    volumes_path = out_dir / f"{stem}_volumes.csv"

    # FastSurfer writes into <sd>/<sid>/; use a scratch SUBJECTS_DIR and point the
    # seg + stats outputs at absolute paths, then surface them to out_dir.
    with tempfile.TemporaryDirectory() as sd:
        stats_path = Path(sd) / f"{stem}_aseg.stats"
        cmd = [
            binary,
            "--t1", str(input_path),
            "--sid", stem,
            "--sd", sd,
            "--seg_only",
            "--asegdkt_segfile", str(seg_path),
            "--asegdkt_statsfile", str(stats_path),
            "--device", resolved_device,
            "--threads", str(threads),
        ]
        print(
            f"[medmcp-neuro] segment: running FastSurfer on {input_path.name} "
            f"(device={resolved_device})...",
            file=sys.stderr,
            flush=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        if proc.returncode != 0:
            note = cuda_unavailable_note() if resolved_device == "cuda" else ""
            raise RuntimeError(
                f"FastSurfer failed (exit {proc.returncode}): {proc.stderr.strip()}.{note}"
            )
        if not seg_path.exists():
            raise RuntimeError(f"Segmentation completed but output not found: {seg_path}")

        # Volume CSV from FastSurfer's own stats (no FreeSurfer mri_segstats).
        with open(volumes_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["structure", "volume_mm3"])
            writer.writerows(_parse_aseg_stats(stats_path))

    result: SegmentResult = {
        "seg_path": str(seg_path),
        "volumes_path": str(volumes_path),
        "input_path": str(input_path),
        "device": resolved_device,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the segmentation result as a compact key-value list:\n"
            "  Input:      <input_path>\n"
            "  Seg labels: <seg_path>\n"
            "  Device:     <device>\n"
            "  Volumes:    <volumes_path>\n"
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Tell the user the output paths. If they want a specific "
            "structure (e.g. thalamus), read the CSV at volumes_path and report the "
            "matching row(s). If the image was registered to a template, offer to warp "
            "the label map using apply_transform with interpolation=NearestNeighbor."
        ),
    }
    return result


def list_brain_segmentation_labels(parc: bool = True) -> LabelListResult:
    """List the structures FastSurfer segments and their volume-CSV names.

    Args:
        parc: Include the cortical DKT parcels (default True). When False, only the
            subcortical and global structures are listed.

    Returns:
        LabelListResult with the structure names as they appear in the volumes CSV.
    """
    cortical = _CORTICAL_PARCEL_NAMES if parc else []
    total = len(_SUBCORTICAL_LABELS) + (2 * len(cortical) if parc else 0)
    result: LabelListResult = {
        "parc": parc,
        "total_structures": total,
        "subcortical_and_global": _SUBCORTICAL_LABELS,
        "cortical_parcels": cortical,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            f"State that FastSurfer labels {total} structures "
            f"({'with' if parc else 'without'} cortical parcellation).\n"
            "List a few relevant subcortical structures (e.g. left/right thalamus) "
            "rather than dumping the full list unless asked.\n"
            "Cortical parcels are reported per hemisphere as ctx-lh-<name> / ctx-rh-<name>."
        ),
    }
    return result
