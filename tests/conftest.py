"""Shared fixtures: a workspace, synthetic images and synthetic FreeSurfer outputs.

Nothing here comes from a real scan. The stats files are made-up numbers in
FreeSurfer's exact file format, so the parsers are tested against the format and
the tests carry no third-party data.
"""

from collections.abc import Iterator
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ASEG_STATS = """\
# Title Segmentation Statistics
#
# generating_program mri_segstats
# subjectname synth01
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1100000.000000, mm^3
# Measure BrainSegNotVent, BrainSegVolNotVent, Brain Seg Volume Without Ventricles, 1090000.0, mm^3
# Measure Cortex, CortexVol, Total cortical gray matter volume, 450000.500000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 610000.250000, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Vol, 1500000.125, mm^3
# NTableCols 10
# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
1 4 4000 4100.5 Left-Lateral-Ventricle 38.0000 13.0000 11.0000 92.0000 81.0000
2 17 3900 3950.3 Left-Hippocampus 74.0000 7.0000 20.0000 100.0000 80.0000
3 53 4000 4050.0 Right-Hippocampus 74.0000 6.0000 43.0000 97.0000 54.0000
4 72 0 0.0 5th-Ventricle 0.0000 0.0000 0.0000 0.0000 0.0000
"""

APARC_TEMPLATE = """\
# Table of FreeSurfer cortical parcellation anatomical statistics
#
# generating_program mris_anatomical_stats
# hemi {hemi}
# Measure Cortex, NumVert, Number of Vertices, {nvert}, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, {area}, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, {thick}, mm
# NTableCols 10
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
bankssts 1500 1000 2300 2.370 0.430 0.090 0.015 9 1.0
entorhinal 600 400 1700 3.100 0.810 0.099 0.019 3 0.5
"""


def write_stats(subject: Path) -> Path:
    """Write synthetic aseg/lh.aparc/rh.aparc stats files into ``subject/stats``."""
    stats = subject / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / "aseg.stats").write_text(ASEG_STATS)
    (stats / "lh.aparc.stats").write_text(
        APARC_TEMPLATE.format(hemi="lh", nvert=120000, area="80000.5", thick="2.51234")
    )
    (stats / "rh.aparc.stats").write_text(
        APARC_TEMPLATE.format(hemi="rh", nvert=121000, area="81000.25", thick="2.52345")
    )
    return subject


def write_fs_nifti(path: Path, shape: tuple[int, ...], zooms: tuple[float, ...]) -> Path:
    """A small NIfTI with the given shape and voxel sizes (data: zeros)."""
    affine = np.diag([*zooms[:3], 1.0])
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A workspace root (``MEDMCP_WORKSPACE``) with no licence in it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("MEDMCP_WORKSPACE", str(ws))
    monkeypatch.delenv("MEDMCP_JOBS", raising=False)
    yield ws


@pytest.fixture
def licence(workspace: Path) -> Path:
    """A placeholder licence file at the default location (not a real key)."""
    path = workspace / ".medmcp" / "licenses" / "freesurfer.txt"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder-not-a-licence\n")
    return path


@pytest.fixture
def t1(workspace: Path) -> Path:
    """A 1 mm isotropic 3D T1 in the workspace."""
    return write_fs_nifti(workspace / "sub-01_T1w.nii.gz", (64, 64, 48), (1.0, 1.0, 1.0))
