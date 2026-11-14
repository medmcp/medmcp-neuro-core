"""Volumes and cortical thickness from a finished recon-all subject.

All numbers are FreeSurfer's own, read verbatim from the subject's ``stats/``
files: ``aseg.stats`` (mri_segstats: subcortical volumes, eTIV and the other global
volumes) and ``lh.aparc.stats`` / ``rh.aparc.stats`` (mris_anatomical_stats:
thickness, area and volume per Desikan-Killiany region, and each hemisphere's mean
thickness and white-surface area). Nothing is recomputed or derived here.

A FastSurfer surface run (segment_brain with surfaces=True) writes the same files,
except that its cortical parcellation is the DKT atlas mapped from its segmentation
(``lh/rh.aparc.DKTatlas.mapped.stats``) unless FreeSurfer's surface registration ran
too; the tool then reads that and says which parcellation the numbers are for.
"""

import csv
import sys
from pathlib import Path
from typing import Any, TypedDict

from medmcp_neuro_core.tools.freesurfer._statsfile import StatsFile, read_stats
from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root
from medmcp_neuro_core.tools.freesurfer.recon import RESEARCH_NOTE

JsonDict = dict[str, Any]

# aparc.stats column -> CSV column (units from FreeSurfer's TableCol lines).
_PARCEL_COLUMNS: dict[str, str] = {
    "StructName": "region",
    "NumVert": "num_vertices",
    "SurfArea": "surface_area_mm2",
    "GrayVol": "gray_volume_mm3",
    "ThickAvg": "thickness_mean_mm",
    "ThickStd": "thickness_std_mm",
    "MeanCurv": "mean_curvature_mm-1",
    "GausCurv": "gaussian_curvature_mm-2",
    "FoldInd": "folding_index",
    "CurvInd": "curvature_index",
}
# aseg.stats global measures surfaced in the result itself (all are in the CSV).
_KEY_MEASURES: tuple[str, ...] = (
    "eTIV",
    "BrainSegVol",
    "BrainSegVolNotVent",
    "TotalGrayVol",
    "CortexVol",
    "CerebralWhiteMatterVol",
    "SubCortGrayVol",
    "VentricleChoroidVol",
)
_HEMI_MEASURES: dict[str, str] = {
    "MeanThickness": "mean_thickness_mm",
    "WhiteSurfArea": "white_surface_area_mm2",
    "NumVert": "num_vertices",
}


# Cortical parcellation stats, in order of preference: recon-all's Desikan-Killiany,
# then the DKT atlas (FastSurfer's mapped one, or recon-all's own).
_PARCELLATIONS: tuple[tuple[str, str], ...] = (
    ("aparc", "Desikan-Killiany (aparc)"),
    ("aparc.DKTatlas.mapped", "Desikan-Killiany-Tourville (aparc.DKTatlas.mapped, FastSurfer)"),
    ("aparc.DKTatlas", "Desikan-Killiany-Tourville (aparc.DKTatlas)"),
)


def _parcellation(stats_dir: Path) -> tuple[str, str] | None:
    """(file stem, description) of the first parcellation with stats for both hemispheres."""
    for stem, label in _PARCELLATIONS:
        if all((stats_dir / f"{h}.{stem}.stats").is_file() for h in ("lh", "rh")):
            return stem, label
    return None


class FreeSurferStatsResult(TypedDict):
    """Result of freesurfer_stats."""

    subject_dir: str
    volumes_csv: str
    parcels_csv: str
    measures_csv: str
    qc_png: str | None
    parcellation: str
    hemispheres: dict[str, dict[str, float]]
    key_measures_mm3: dict[str, float]
    aseg_volumes_mm3: dict[str, float]
    notes: list[str]
    _render: str


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def freesurfer_stats(
    subject_dir: str, output_dir: str | None = None, qc_snapshot: bool = True
) -> JsonDict:
    """Read volumes and cortical thickness from a finished recon-all or FastSurfer subject.

    Writes three CSV files with FreeSurfer's own numbers: subcortical volumes
    (aseg.stats), cortical thickness / area / volume per Desikan-Killiany region and
    hemisphere (lh/rh.aparc.stats), and the global measures (eTIV, total grey and
    white matter, mean thickness per hemisphere, ...). Optionally also a QC image of
    the segmentation over the T1.

    Args:
        subject_dir: The subject folder (absolute path): ``<output_dir>/<subject_id>``
            of run_recon_all, or ``surfaces.subject_dir`` of segment_brain with
            surfaces=True, with the run finished.
        output_dir: Where to write the CSVs and the QC image. Default:
            ``<subject_dir>/medmcp_stats``.
        qc_snapshot: Also write ``qc_aparc_aseg.png`` (aparc+aseg over the T1, three
            slices).

    Returns:
        Paths of the CSVs (``volumes_csv``, ``parcels_csv``, ``measures_csv``) and the
        ``qc_png``, per-hemisphere mean thickness and white-surface area, the key
        global volumes (eTIV, BrainSegVol, ...) and every aseg structure's volume.
    """
    ws = workspace_root()
    subject = resolve_inside(subject_dir, "subject_dir", ws)
    stats_dir = subject / "stats"
    parc = _parcellation(stats_dir)
    parc_stem, parc_label = parc if parc is not None else ("aparc", _PARCELLATIONS[0][1])
    needed = {
        "aseg.stats": stats_dir / "aseg.stats",
        "lh.aparc.stats": stats_dir / f"lh.{parc_stem}.stats",
        "rh.aparc.stats": stats_dir / f"rh.{parc_stem}.stats",
    }
    missing = [p.name for p in needed.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{subject} has no {', '.join('stats/' + m for m in missing)}: recon-all has not "
            "finished for this subject (check recon_all_status), or it is not a FreeSurfer "
            "subject folder."
        )
    out = resolve_inside(output_dir, "output_dir", ws) if output_dir else subject / "medmcp_stats"
    out.mkdir(parents=True, exist_ok=True)

    aseg = read_stats(needed["aseg.stats"])
    hemis: dict[str, StatsFile] = {
        "lh": read_stats(needed["lh.aparc.stats"]),
        "rh": read_stats(needed["rh.aparc.stats"]),
    }

    # Subcortical volumes, row by row as mri_segstats wrote them.
    volume_rows: list[list[Any]] = []
    aseg_volumes: dict[str, float] = {}
    for row in aseg.rows:
        name = row.get("StructName", "")
        volume = row.get("Volume_mm3", "")
        volume_rows.append([name, row.get("SegId", ""), row.get("NVoxels", ""), volume])
        aseg_volumes[name] = float(volume)
    volumes_csv = out / "aseg_volumes.csv"
    _write_csv(volumes_csv, ["structure", "seg_id", "n_voxels", "volume_mm3"], volume_rows)

    # Cortical regions, both hemispheres.
    columns = [c for c in hemis["lh"].columns if c in _PARCEL_COLUMNS]
    parcel_rows: list[list[Any]] = []
    for hemi, stats in hemis.items():
        for row in stats.rows:
            # Verbatim strings: exactly the digits FreeSurfer wrote.
            parcel_rows.append([hemi, *[row[c] for c in columns]])
    parcels_csv = out / "aparc_cortical.csv"
    _write_csv(parcels_csv, ["hemisphere", *[_PARCEL_COLUMNS[c] for c in columns]], parcel_rows)

    # Global measures from all three files.
    measure_rows: list[list[Any]] = []
    for source, stats in (("aseg", aseg), ("lh.aparc", hemis["lh"]), ("rh.aparc", hemis["rh"])):
        for m in stats.measures:
            measure_rows.append([source, m.name, m.description, m.value, m.unit])
    measures_csv = out / "global_measures.csv"
    _write_csv(measures_csv, ["source", "measure", "description", "value", "unit"], measure_rows)

    hemispheres: dict[str, dict[str, float]] = {}
    for hemi, stats in hemis.items():
        hemispheres[hemi] = {
            label: int(m.value) if m.unit == "unitless" and m.value.is_integer() else m.value
            for name, label in _HEMI_MEASURES.items()
            if (m := stats.measure(name)) is not None
        }
    key_measures = {
        name: m.value for name in _KEY_MEASURES if (m := aseg.measure(name)) is not None
    }

    notes: list[str] = []
    qc_png: str | None = None
    if qc_snapshot:
        try:
            from medmcp_neuro_core.tools.freesurfer._qc import qc_snapshot as _snapshot

            qc_png = str(_snapshot(subject, out / "qc_aparc_aseg.png"))
        except Exception as exc:  # the numbers stand without the picture
            print(f"[medmcp-freesurfer] QC snapshot failed: {exc}", file=sys.stderr, flush=True)
            notes.append(f"QC snapshot not written: {exc}")

    result: FreeSurferStatsResult = {
        "subject_dir": str(subject),
        "volumes_csv": str(volumes_csv),
        "parcels_csv": str(parcels_csv),
        "measures_csv": str(measures_csv),
        "qc_png": qc_png,
        "parcellation": parc_label,
        "hemispheres": hemispheres,
        "key_measures_mm3": key_measures,
        "aseg_volumes_mm3": aseg_volumes,
        "notes": notes,
        "_render": (
            "DISPLAY RULES:\n"
            "1. A small table per hemisphere (lh, rh): mean cortical thickness (mm) and "
            "white-surface area (mm2), from 'hemispheres'.\n"
            "2. eTIV and BrainSegVol (mm3) from 'key_measures_mm3'.\n"
            "3. If the user asked about specific structures, their volumes from "
            "'aseg_volumes_mm3' (names are FreeSurfer's, e.g. Left-Hippocampus); for a "
            "cortical region's thickness or area, read the matching rows of parcels_csv "
            "(columns hemisphere, region, thickness_mean_mm, surface_area_mm2, ...).\n"
            "4. The CSV paths and, if present, qc_png (show the image if you can). Name "
            "the cortical parcellation ('parcellation') the region values are for.\n"
            "Report numbers as given (FreeSurfer's own values); do not recompute them. "
            "Say once: " + RESEARCH_NOTE + "\n"
            "NEXT ACTION: Offer to look at specific structures or regions, or to compare "
            "with another subject's stats."
        ),
    }
    # Plain dict: FastMCP drops "_render" from a TypedDict's structured output.
    return dict(result)
