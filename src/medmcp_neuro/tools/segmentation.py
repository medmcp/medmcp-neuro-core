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
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools._neuro import cuda_unavailable_note, detect_devices, find_binary, nii_stem

# --- input sanity thresholds ---------------------------------------------------
# FastSurferVINN conforms inputs to ~1mm internally and handles ~0.7-1.0mm native,
# so these guard against pathological *clinical* inputs (thick-slice 2D), not normal
# 1mm variation. Mildly-off resolution is a warning; strongly anisotropic or
# thick-slice data is a hard error (overridable with force=True).
_VOXEL_TARGET_MM: tuple[float, float] = (0.7, 1.3)  # comfortable band; outside -> warn
_VOXEL_BLOCK_MM: float = 2.0  # any voxel dim at/above this -> block (thick slices)
_ANISO_WARN: float = 1.5  # max/min zoom ratio above this -> warn
_ANISO_BLOCK: float = 2.0  # ...at/above this -> block

# Filename tokens that name a non-T1w contrast. NIfTI headers carry no contrast /
# sequence field (unlike DICOM), so a BIDS-style filename suffix is the only signal
# we have for modality — hence a warning, never a hard block.
_NON_T1W_TOKENS: frozenset[str] = frozenset(
    {"flair", "t2w", "t2star", "t2", "dwi", "dti", "swi", "pd", "pdw", "bold", "asl", "angio", "ct"}
)

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
    warnings: list[str]
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


def _fastsurfer_python() -> str | None:
    """Return the interpreter for FastSurfer's isolated venv, if available.

    FastSurfer pins its own torch, so it lives in a separate venv
    (/opt/fastsurfer-venv) and run_fastsurfer.sh is pointed at it via ``--py``.
    Overridable with $FASTSURFER_PYTHON; returns None to fall back to the script's
    default interpreter (e.g. in tests / non-container installs).
    """
    override = os.environ.get("FASTSURFER_PYTHON")
    if override:
        return override
    candidate = Path("/opt/fastsurfer-venv/bin/python")
    return str(candidate) if candidate.is_file() else None


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


def _read_voxel_zooms(input_path: Path) -> tuple[float, ...]:
    """Return the voxel sizes (mm) from a NIfTI header.

    Isolates the nibabel access: ``load`` carries untyped ``**kwargs`` and the base
    ``FileBasedImage`` doesn't statically expose ``get_zooms``, so we narrow to a
    SpatialImage (which all NIfTIs are) to keep the rest of the module strictly typed.

    Raises:
        ValueError: If the file is not a spatial image (no voxel geometry).
    """
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage

    img = nib.load(str(input_path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(img, SpatialImage):
        msg = f"{input_path} is not a spatial image (no voxel geometry)"
        raise ValueError(msg)
    return tuple(float(z) for z in img.header.get_zooms())


def _check_input(input_path: Path, force: bool) -> list[str]:
    """Sanity-check the input image before handing it to FastSurfer.

    FastSurfer is trained on T1-weighted MRI and conforms inputs to ~1mm, but it
    does not validate contrast or resolution itself. This guards the two failure
    modes we can detect:

    * **Modality** — NIfTI headers carry no contrast field, so we can only flag a
      *filename* that names a non-T1w contrast (BIDS suffix). Always a warning,
      never a block: a file named ``sub01.nii.gz`` is not necessarily wrong.
    * **Resolution** — voxel zooms live in the header. Mildly-off resolution or
      mild anisotropy is a warning; strongly anisotropic or thick-slice data
      (typical of 2D clinical acquisitions) is a hard error unless ``force=True``,
      because conforming it to 1mm yields a garbage segmentation.

    Args:
        input_path: NIfTI file to inspect.
        force: Downgrade the pathological-resolution error to a warning and run anyway.

    Returns:
        Non-fatal warnings to surface to the caller (empty when the input looks fine).

    Raises:
        ValueError: On pathological resolution unless ``force`` is True.
    """
    warnings: list[str] = []

    # Modality: filename heuristic only (header has no contrast field). Split the
    # stem into BIDS-style fields so we match suffixes, not stray substrings.
    fields = {f for f in re.split(r"[^a-z0-9]+", nii_stem(input_path).lower()) if f}
    contrast = sorted(fields & _NON_T1W_TOKENS)
    if contrast:
        warnings.append(
            f"Filename names a non-T1w contrast ({', '.join(contrast)}); FastSurfer is "
            "trained on T1-weighted MRI and results on other contrasts are unreliable. "
            "Contrast cannot be confirmed from the NIfTI header, so this is a warning only."
        )

    # Resolution: read voxel zooms from the header.
    try:
        zooms = _read_voxel_zooms(input_path)[:3]
    except Exception as exc:  # unreadable/odd header: skip the check, don't fail the run
        warnings.append(
            f"Could not read voxel resolution from the header ({exc}); skipping resolution check."
        )
        return warnings

    if len(zooms) < 3 or any(z <= 0 for z in zooms):
        warnings.append(f"Unexpected voxel dimensions {zooms}; skipping resolution check.")
        return warnings

    lo, hi = min(zooms), max(zooms)
    aniso = hi / lo
    res_str = "x".join(f"{z:.2f}" for z in zooms) + " mm"

    if hi >= _VOXEL_BLOCK_MM or aniso >= _ANISO_BLOCK:
        msg = (
            f"Input voxel resolution {res_str} (anisotropy {aniso:.1f}x) is outside "
            "FastSurfer's supported range — strongly anisotropic or thick-slice data "
            "(typical of 2D clinical acquisitions) conforms to a garbage segmentation."
        )
        if not force:
            raise ValueError(f"{msg} Pass force=True to run anyway.")
        warnings.append(
            f"{msg} Running anyway because force=True; quality may be severely degraded."
        )
    elif aniso >= _ANISO_WARN or not (_VOXEL_TARGET_MM[0] <= lo and hi <= _VOXEL_TARGET_MM[1]):
        warnings.append(
            f"Input resolution {res_str} (anisotropy {aniso:.1f}x) is outside FastSurfer's "
            f"ideal ~{_VOXEL_TARGET_MM[0]:g}-{_VOXEL_TARGET_MM[1]:g} mm isotropic band; "
            "results may be less accurate."
        )

    return warnings


def segment_brain(
    input_path: Path,
    output_dir: Path | None = None,
    device: str = "auto",
    threads: int = 4,
    force: bool = False,
) -> SegmentResult:
    """Segment brain structures from a NIfTI image using FastSurfer.

    Runs FastSurfer's deep-learning segmentation (FastSurferVINN, ``--seg_only``)
    to label 95 cortical and subcortical structures, including left/right thalamus
    and the other subcortical volumes. No FreeSurfer license is required. Accepts
    full-head or skull-stripped T1w images; skull stripping is not required first.

    Outputs a discrete label map (``.mgz``) and a per-structure volume CSV. Call
    list_brain_segmentation_labels() to see the structures and CSV column names.

    The input is sanity-checked first (see _check_input): a filename that names a
    non-T1w contrast is flagged as a warning, and strongly anisotropic / thick-slice
    resolution is rejected unless ``force=True``. Non-fatal warnings are returned in
    the ``warnings`` field.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
        output_dir: Directory for outputs. Defaults to input_path's directory.
        device: 'auto' (default; cuda > mps > cpu), or force 'cuda' / 'mps' / 'cpu'.
        threads: CPU threads for pre/post-processing. Default 4.
        force: Run even when the input resolution is outside FastSurfer's supported
            range (otherwise such inputs raise ValueError). Default False.

    Returns:
        SegmentResult with paths to the segmentation label map and volume CSV, plus
        any non-fatal input warnings.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the input resolution is pathological and force is False.
        RuntimeError: If FastSurfer is not installed or segmentation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    warnings = _check_input(input_path, force=force)
    for warning in warnings:
        print(f"[medmcp-neuro] segment: WARNING: {warning}", file=sys.stderr, flush=True)

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
            "--t1",
            str(input_path),
            "--sid",
            stem,
            "--sd",
            sd,
            "--seg_only",
            "--asegdkt_segfile",
            str(seg_path),
            "--asegdkt_statsfile",
            str(stats_path),
            "--device",
            resolved_device,
            "--threads",
            str(threads),
            # MedMCP launches stack containers as root; FastSurfer refuses root
            # unless explicitly allowed.
            "--allow_root",
        ]
        fs_python = _fastsurfer_python()
        if fs_python:
            cmd += ["--py", fs_python]
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
        # FastSurfer's seg-only stream can exit non-zero on a cosmetic final step
        # (symlinking the stats into the standard <sid>/stats/ dir) even when the
        # segmentation and stats were written. Treat the presence of both expected
        # outputs as the success signal rather than the exit code.
        if not seg_path.exists() or not stats_path.exists():
            note = cuda_unavailable_note() if resolved_device == "cuda" else ""
            raise RuntimeError(
                f"FastSurfer failed (exit {proc.returncode}): {proc.stderr.strip()}.{note}"
            )
        if proc.returncode != 0:
            print(
                f"[medmcp-neuro] segment: FastSurfer exited {proc.returncode} but the "
                "segmentation and stats were produced; continuing.",
                file=sys.stderr,
                flush=True,
            )

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
        "warnings": warnings,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "If 'warnings' is non-empty, surface each warning to the user FIRST — the "
            "input may not be ideal for FastSurfer (wrong contrast or off resolution).\n"
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
