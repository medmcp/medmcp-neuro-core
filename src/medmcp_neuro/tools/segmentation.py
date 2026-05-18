"""Brain segmentation tool using SynthSeg (FreeSurfer)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools._neuro import find_binary, nii_stem

# Structures always measured (parc=True or False), as they appear in the volumes CSV.
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

# Additional cortical parcels measured when parc=True (Desikan-Killiany atlas).
# Names match the ctx-lh-* / ctx-rh-* FreeSurfer convention used in output CSVs.
_CORTICAL_PARCEL_NAMES: list[str] = [
    "bankssts",
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
    _render: str


class LabelListResult(TypedDict):
    """Available segmentation labels for a given SynthSeg configuration."""

    parc: bool
    total_structures: int
    subcortical_and_global: list[str]
    cortical_parcels: list[str]
    _render: str


def _find_synthseg() -> tuple[str, dict[str, str]]:
    """Return (absolute path to mri_synthseg, extra env vars needed to run it).

    Checks, in order: $MRI_SYNTHSEG override, PATH, venv bin dir, $FREESURFER_HOME,
    and common install locations (~/, /usr/local, /opt).  Also infers FREESURFER_HOME
    from the binary path when it is not already set — the mri_synthseg shell wrapper
    exits immediately if FREESURFER_HOME is missing.
    """
    binary = find_binary("mri_synthseg", "MRI_SYNTHSEG")
    if binary is None:
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home:
            candidate = Path(fs_home) / "bin" / "mri_synthseg"
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                binary = str(candidate)
    if binary is None:
        for candidate_home in [
            Path.home() / "freesurfer",
            Path("/usr/local/freesurfer"),
            Path("/opt/freesurfer"),
        ]:
            candidate = candidate_home / "bin" / "mri_synthseg"
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                binary = str(candidate)
                break
    if binary is None:
        raise RuntimeError(
            "mri_synthseg not found. Install FreeSurfer ≥ 7.3 from "
            "https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall "
            "and set $FREESURFER_HOME, or put mri_synthseg on $PATH."
        )
    env_extras: dict[str, str] = {}
    if not os.environ.get("FREESURFER_HOME"):
        inferred = Path(binary).parent.parent
        if (inferred / "FreeSurferEnv.sh").is_file():
            env_extras["FREESURFER_HOME"] = str(inferred)
    return binary, env_extras


def segment_brain(
    input_path: Path,
    output_dir: Path | None = None,
    parc: bool = True,
    robust: bool = False,
) -> SegmentResult:
    """Segment brain structures from a NIfTI image using SynthSeg.

    Uses FreeSurfer's SynthSeg — a contrast-agnostic deep learning segmentation
    tool that produces accurate results on T1w, T2w, FLAIR, DWI, and other
    contrasts without retraining. Accepts both full-head and skull-stripped
    images; skull stripping is not required beforehand.

    Outputs a discrete segmentation label map and a per-structure volume CSV.
    With parcellation enabled (default), 95 cortical and subcortical structures
    are labeled; without it, 33 coarser labels are produced. Call
    list_brain_segmentation_labels(parc=...) to see the full list of structures and
    exact CSV column names before or after segmentation.

    Args:
        input_path: Absolute path to the input NIfTI file (.nii or .nii.gz).
        output_dir: Directory where outputs are written.
            Defaults to the same directory as input_path.
        parc: Enable cortical parcellation (95 labels). When False, produces
            31 coarser subcortical / tissue labels. Default True.
        robust: Enable robust mode for low-quality or non-standard images
            (e.g. clinical protocols, extreme noise). Slower but more reliable.
            Default False.

    Returns:
        SegmentResult with paths to the segmentation label map and volume CSV.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError: If mri_synthseg is not installed or segmentation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    binary, env_extras = _find_synthseg()

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nii_stem(input_path)
    seg_path = out_dir / f"{stem}_dseg.nii.gz"
    volumes_path = out_dir / f"{stem}_volumes.csv"

    cmd = [binary, "--i", str(input_path), "--o", str(seg_path), "--vol", str(volumes_path)]
    if parc:
        cmd.append("--parc")
    if robust:
        cmd.append("--robust")

    print(
        f"[medmcp-neuro] segment: running SynthSeg on {input_path.name}...",
        file=sys.stderr,
        flush=True,
    )

    run_env = {**os.environ, **env_extras} if env_extras else None
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=run_env)

    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    if proc.returncode != 0:
        raise RuntimeError(f"SynthSeg failed (exit {proc.returncode}): {proc.stderr.strip()}")

    if not seg_path.exists():
        raise RuntimeError(f"Segmentation completed but output not found: {seg_path}")

    label_count = "95" if parc else "33"
    result: SegmentResult = {
        "seg_path": str(seg_path),
        "volumes_path": str(volumes_path),
        "input_path": str(input_path),
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the segmentation result as a compact key-value list:\n"
            "  Input:      <input_path>\n"
            "  Seg labels: <seg_path>\n"
            f"  Structures: {label_count} labels\n"
            "  Volumes:    <volumes_path>\n"
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Tell the user the output paths. If they want to inspect "
            "volumes, read the CSV at volumes_path and present it as a table. "
            "If the image was registered to a template, offer to warp the label map "
            "using apply_transform with interpolation=NearestNeighbor."
        ),
    }
    return result


def list_brain_segmentation_labels(parc: bool = True) -> LabelListResult:
    """List all brain structures measured by the segment_brain tool.

    Call this before or after segmentation to answer questions like 'is X
    measured?', 'which regions are available?', or 'what does parc add?'.
    Returns the exact column names that appear in the volumes CSV produced by
    segment_brain(), so users can reference them directly.

    Args:
        parc: Match the parcellation flag used (or planned) for segment_brain().
            True (default) returns all 95 structures including cortical parcels.
            False returns the 33 subcortical / global structures only.

    Returns:
        LabelListResult with subcortical_and_global (always present) and
        cortical_parcels (empty when parc=False).
    """
    cortical: list[str] = (
        [f"ctx-lh-{n}" for n in _CORTICAL_PARCEL_NAMES]
        + [f"ctx-rh-{n}" for n in _CORTICAL_PARCEL_NAMES]
        if parc
        else []
    )
    total = len(_SUBCORTICAL_LABELS) + len(cortical)
    return {
        "parc": parc,
        "total_structures": total,
        "subcortical_and_global": _SUBCORTICAL_LABELS,
        "cortical_parcels": cortical,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            f"Report that segment_brain() measures {total} structures "
            f"with parc={'True' if parc else 'False'}.\n"
            "Present subcortical_and_global as a simple bullet list.\n"
            + (
                "Group cortical_parcels by lobe:\n"
                "  Frontal: superiorfrontal, rostralmiddlefrontal, caudalmiddlefrontal, "
                "parsopercularis, parstriangularis, parsorbitalis, lateralorbitofrontal, "
                "medialorbitofrontal, precentral, paracentral, frontalpole\n"
                "  Parietal: superiorparietal, inferiorparietal, supramarginal, "
                "postcentral, precuneus\n"
                "  Temporal: superiortemporal, middletemporal, inferiortemporal, "
                "fusiform, transversetemporal, bankssts, temporalpole, "
                "parahippocampal, entorhinal\n"
                "  Occipital: lateraloccipital, lingual, cuneus, pericalcarine\n"
                "  Cingulate: rostralanteriorcingulate, caudalanteriorcingulate, "
                "posteriorcingulate, isthmuscingulate\n"
                "  Other: insula\n"
                "Each parcel is measured bilaterally (ctx-lh-* and ctx-rh-*).\n"
                if parc
                else "cortical_parcels is empty because parc=False.\n"
            )
            + "NEXT ACTION: If the user asked whether a specific structure is available, "
            "confirm yes/no and name the exact CSV column they should look for."
        ),
    }
