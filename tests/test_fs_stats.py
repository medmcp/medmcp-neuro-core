"""freesurfer_stats: FreeSurfer's numbers, verbatim, as CSV; and the QC image."""

import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from medmcp_neuro_core.tools.freesurfer._statsfile import read_stats
from medmcp_neuro_core.tools.freesurfer.stats import freesurfer_stats
from tests.conftest import write_stats


def _rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_stats_file_parser(tmp_path: Path) -> None:
    """Measures and table rows are read as FreeSurfer wrote them."""
    subject = write_stats(tmp_path / "s")
    aseg = read_stats(subject / "stats" / "aseg.stats")
    etiv = aseg.measure("eTIV")
    assert etiv is not None and etiv.value == 1500000.125 and etiv.unit == "mm^3"
    assert aseg.columns[:5] == ["Index", "SegId", "NVoxels", "Volume_mm3", "StructName"]
    assert aseg.rows[1]["StructName"] == "Left-Hippocampus"


def test_freesurfer_stats_csvs(workspace: Path) -> None:
    """Volumes, per-region cortical values and global measures, verbatim."""
    subject = write_stats(workspace / "subjects" / "synth01")
    result = freesurfer_stats(str(subject), qc_snapshot=False)

    assert result["hemispheres"]["lh"] == {
        "mean_thickness_mm": 2.51234,
        "white_surface_area_mm2": 80000.5,
        "num_vertices": 120000,
    }
    assert result["hemispheres"]["rh"]["mean_thickness_mm"] == 2.52345
    assert result["key_measures_mm3"]["eTIV"] == 1500000.125
    assert result["aseg_volumes_mm3"]["Left-Hippocampus"] == 3950.3

    volumes = _rows(result["volumes_csv"])
    assert volumes[1] == {
        "structure": "Left-Hippocampus",
        "seg_id": "17",
        "n_voxels": "3900",
        "volume_mm3": "3950.3",
    }
    parcels = _rows(result["parcels_csv"])
    assert len(parcels) == 4
    assert parcels[0]["hemisphere"] == "lh" and parcels[0]["region"] == "bankssts"
    # The digits FreeSurfer wrote, unchanged (0.090 stays 0.090).
    assert parcels[0]["thickness_mean_mm"] == "2.370"
    assert parcels[0]["mean_curvature_mm-1"] == "0.090"
    assert parcels[3]["hemisphere"] == "rh" and parcels[3]["region"] == "entorhinal"

    measures = _rows(result["measures_csv"])
    sources = {(m["source"], m["measure"]) for m in measures}
    assert ("aseg", "eTIV") in sources and ("rh.aparc", "MeanThickness") in sources
    assert Path(result["volumes_csv"]).parent == subject / "medmcp_stats"


def test_unfinished_subject(workspace: Path) -> None:
    """A subject without stats files is reported as not finished."""
    subject = workspace / "subjects" / "half"
    (subject / "stats").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not finished"):
        freesurfer_stats(str(subject))


def test_output_outside_workspace_is_refused(workspace: Path) -> None:
    """CSV output stays inside the workspace."""
    subject = write_stats(workspace / "subjects" / "synth01")
    with pytest.raises(ValueError, match="outside the workspace"):
        freesurfer_stats(str(subject), output_dir="/tmp/elsewhere")


def test_qc_snapshot(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The QC image is written from mri/T1.mgz and mri/aparc+aseg.mgz."""
    monkeypatch.setenv("FREESURFER_HOME", str(workspace / "no-freesurfer-here"))
    subject = write_stats(workspace / "subjects" / "synth01")
    (subject / "mri").mkdir()
    t1 = np.random.default_rng(0).integers(0, 255, (32, 32, 32)).astype(np.uint8)
    seg = np.zeros((32, 32, 32), dtype=np.int32)
    seg[10:20, 10:20, 10:20] = 17
    seg[12:16, 12:16, 12:16] = 1035
    volumes: list[tuple[str, Any]] = [("T1.mgz", t1), ("aparc+aseg.mgz", seg)]
    for name, data in volumes:
        nib.save(nib.MGHImage(data, np.eye(4)), str(subject / "mri" / name))  # pyright: ignore[reportUnknownMemberType]
    result = freesurfer_stats(str(subject))
    assert result["qc_png"] is not None
    png = Path(result["qc_png"])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert result["notes"] == []


def test_qc_failure_does_not_lose_the_numbers(workspace: Path) -> None:
    """Without the volumes the QC image is skipped with a note; the CSVs still come."""
    subject = write_stats(workspace / "subjects" / "synth01")
    result = freesurfer_stats(str(subject))
    assert result["qc_png"] is None
    assert result["notes"] and "T1.mgz" in result["notes"][0]
    assert Path(result["volumes_csv"]).is_file()


def test_fastsurfer_dkt_parcellation(workspace: Path) -> None:
    """A FastSurfer surface run's DKT stats are read when there is no aparc.stats."""
    subject = write_stats(workspace / "fastsurfer" / "synth02")
    for hemi in ("lh", "rh"):
        stats = subject / "stats"
        (stats / f"{hemi}.aparc.stats").rename(stats / f"{hemi}.aparc.DKTatlas.mapped.stats")
    result = freesurfer_stats(str(subject), qc_snapshot=False)
    assert result["parcellation"].startswith("Desikan-Killiany-Tourville")
    assert result["hemispheres"]["rh"]["mean_thickness_mm"] == 2.52345
    assert len(_rows(result["parcels_csv"])) == 4
