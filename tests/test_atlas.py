"""Tests for atlas lesion load and cohort lesion-frequency maps.

Synthetic atlases and masks with hand-computable overlaps. The end-to-end tests run
real ANTs (antspyx is a runtime dependency) on 20³ volumes, with identity and
affine-only transforms, so the expected numbers are exact.
"""

import csv
import importlib.util
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

import medmcp_neuro_core.tools._atlas as _atlas
from medmcp_neuro_core.tools._overlap import frequency, label_rows, probabilistic_row
from medmcp_neuro_core.tools.atlas import (
    lesion_frequency_map,
    lesion_load_by_atlas,
    list_atlases,
)

_HAS_ANTS = importlib.util.find_spec("ants") is not None
needs_ants = pytest.mark.skipif(not _HAS_ANTS, reason="antspyx not installed")

_SHAPE = (20, 20, 20)
_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])  # 2 mm voxels -> 8 mm³
_VOX = 8.0


def _save(path: Path, data: Any, affine: Any = _AFFINE) -> Path:
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


def _data(path: str) -> Any:
    img: Any = nib.load(path)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(img.dataobj)


def _cst_map() -> Any:
    """p=1 in x 2..5, p=0.5 in x 6..9, inside y,z 0..9."""
    p = np.zeros(_SHAPE, dtype=np.float32)
    p[2:6, 0:10, 0:10] = 1.0
    p[6:10, 0:10, 0:10] = 0.5
    return p


def _lesion() -> Any:
    """4x5x5 = 100 voxels at x 4..7, y,z 0..4."""
    m = np.zeros(_SHAPE, dtype=np.uint8)
    m[4:8, 0:5, 0:5] = 1
    return m


# ── pure counting ──────────────────────────────────────────────────────────────


def test_probabilistic_row_thresholded_and_weighted() -> None:
    """Hand-computed thresholded and probability-weighted overlap."""
    row = probabilistic_row("CST", _lesion() > 0, _cst_map(), _VOX, 0.25)
    assert row["tract_volume_mm3"] == 800 * _VOX
    assert row["lesion_in_tract_mm3"] == 100 * _VOX
    assert row["percent_tract_lesioned"] == 12.5
    assert row["percent_lesion_in_tract"] == 100.0
    assert row["tract_weighted_volume_mm3"] == 600 * _VOX
    assert row["lesion_in_tract_weighted_mm3"] == 75 * _VOX
    assert row["percent_tract_lesioned_weighted"] == 12.5


def test_probabilistic_row_higher_threshold() -> None:
    """At 0.75 only the p=1 core is tract; half the lesion lies in it."""
    row = probabilistic_row("CST", _lesion() > 0, _cst_map(), _VOX, 0.75)
    assert row["tract_volume_mm3"] == 400 * _VOX
    assert row["lesion_in_tract_mm3"] == 50 * _VOX
    assert row["percent_lesion_in_tract"] == 50.0
    # weighted columns do not depend on the threshold
    assert row["lesion_in_tract_weighted_mm3"] == 75 * _VOX


def test_probabilistic_row_empty_tract_gives_none() -> None:
    """A tract with no voxels has undefined percentages, not zero or NaN."""
    row = probabilistic_row("X", _lesion() > 0, np.zeros(_SHAPE), _VOX, 0.25)
    assert row["percent_tract_lesioned"] is None
    assert row["percent_tract_lesioned_weighted"] is None
    assert row["lesion_in_tract_mm3"] == 0


def test_probabilistic_row_shape_mismatch() -> None:
    """Arrays on different grids are rejected."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        probabilistic_row("X", np.zeros((2, 2, 2), bool), np.zeros((3, 3, 3)), 1.0, 0.25)


def _labels() -> Any:
    lab = np.zeros(_SHAPE, dtype=np.int16)
    lab[0:10, :, 0:18] = 1
    lab[10:20, :, 0:18] = 2
    return lab


def _label_lesion() -> Any:
    m = np.zeros(_SHAPE, dtype=np.uint8)
    m[8:12, 0:2, 0:2] = 1  # 8 voxels in region 1, 8 in region 2
    m[0, 0, 19] = 1  # 1 voxel on background
    return m


def test_label_rows() -> None:
    """Per-region volumes, percentages and the outside-atlas row."""
    rows = label_rows(_label_lesion() > 0, _labels(), _VOX, [(1, "A"), (2, "B"), (3, "C")])
    by = {r["region"]: r for r in rows}
    assert by["A"]["region_volume_mm3"] == 10 * 20 * 18 * _VOX
    assert by["A"]["lesion_in_region_mm3"] == 8 * _VOX
    assert by["B"]["lesion_in_region_mm3"] == 8 * _VOX
    assert by["A"]["percent_lesion_in_region"] == round(100 * 8 / 17, 4)
    assert by["C"]["region_volume_mm3"] == 0
    assert by["C"]["percent_region_lesioned"] is None
    assert by["outside atlas"]["lesion_in_region_mm3"] == 1 * _VOX


def test_frequency() -> None:
    """Count / n per voxel."""
    a = np.zeros((2, 2, 2), bool)
    b = np.zeros((2, 2, 2), bool)
    a[0, 0, 0] = b[0, 0, 0] = True
    b[1, 1, 1] = True
    count, freq = frequency([a, b])
    assert count[0, 0, 0] == 2
    assert freq[0, 0, 0] == 1.0
    assert freq[1, 1, 1] == 0.5
    assert freq[0, 1, 0] == 0.0


# ── atlas registry ─────────────────────────────────────────────────────────────


def test_registry_has_no_restricted_atlases() -> None:
    """Only cleared atlases are bundled (JHU/XTRACT/Juelich are not redistributable)."""
    names = " ".join(_atlas.ATLASES).lower()
    for banned in ("jhu", "xtract", "juelich", "julich", "brainnetome"):
        assert banned not in names
    for spec in _atlas.ATLASES.values():
        assert spec.attribution
        assert all(len(s.sha256) == 64 for s in spec.sources)


def test_hcp_name() -> None:
    """HCP1065 file stems become readable tract names."""
    assert _atlas._hcp_name("Corticospinal_Tract_L") == "Corticospinal Tract left"  # pyright: ignore[reportPrivateUsage]
    assert _atlas._hcp_name("Corpus_Callosum_Body") == "Corpus Callosum Body"  # pyright: ignore[reportPrivateUsage]


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    """A changed upstream file is refused, not silently used."""

    def fake_retrieve(_url: str, dest: Path) -> None:
        Path(dest).write_bytes(b"not the pinned file")

    src = _atlas.Source("https://example.invalid/x", "0" * 64, "x.bin")
    with (
        patch.object(urllib.request, "urlretrieve", side_effect=fake_retrieve),
        pytest.raises(RuntimeError, match="Checksum mismatch"),
    ):
        _atlas._download(src, tmp_path)  # pyright: ignore[reportPrivateUsage]
    assert not (tmp_path / "x.bin").exists()


def test_list_atlases() -> None:
    """Every registry entry is listed."""
    res = list_atlases()
    assert {a["name"] for a in res["atlases"]} == set(_atlas.ATLASES)


# ── synthetic prepared atlases ─────────────────────────────────────────────────


@pytest.fixture
def atlas_root(tmp_path: Path) -> Path:
    """A prepared-atlas cache with a synthetic tract atlas and label atlas."""
    root = tmp_path / "atlases"
    tract = root / "pandora_tractseg"
    (tract / "maps").mkdir(parents=True)
    _save(tract / "maps" / "CST_left.nii.gz", _cst_map())
    _save(tract / "maps" / "OTHER.nii.gz", np.zeros(_SHAPE, np.float32))
    (tract / "labels.tsv").write_text(
        "index\tname\tfile\n"
        "1\tCorticospinal tract left\tmaps/CST_left.nii.gz\n"
        "2\tOther tract\tmaps/OTHER.nii.gz\n"
    )
    (tract / ".complete").touch()
    lab = root / "schaefer2018_100"
    lab.mkdir()
    _save(lab / "atlas_dseg.nii.gz", _labels())
    (lab / "labels.tsv").write_text("index\tname\tfile\n1\tA\t\n2\tB\t\n3\tC\t\n")
    (lab / ".complete").touch()
    return root


def _identity_mat(path: Path) -> str:
    import ants  # type: ignore[import-untyped]

    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    ants.write_transform(tx, str(path))  # pyright: ignore[reportUnknownMemberType]
    return str(path)


def _translation_mat(path: Path, offset_mm: tuple[float, float, float]) -> str:
    import ants  # type: ignore[import-untyped]

    tx = ants.create_ants_transform(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        transform_type="AffineTransform", dimension=3, translation=offset_mm
    )
    ants.write_transform(tx, str(path))  # pyright: ignore[reportUnknownMemberType]
    return str(path)


@needs_ants
def test_lesion_load_tract_mask_in_template_space(tmp_path: Path, atlas_root: Path) -> None:
    """Identity resample: thresholded and weighted numbers match the hand count."""
    mask = _save(tmp_path / "sub-01_lesion.nii.gz", _lesion())
    with patch.object(_atlas, "ATLAS_ROOT", atlas_root):
        res = lesion_load_by_atlas(mask, "pandora_tractseg", mask_in_template_space=True)
    assert res["lesion_volume_mm3"] == 100 * _VOX
    assert res["voxel_volume_mm3"] == _VOX
    (cst,) = res["corticospinal_rows"]
    assert cst["lesion_in_tract_mm3"] == 100 * _VOX
    assert cst["percent_tract_lesioned"] == 12.5
    assert cst["lesion_in_tract_weighted_mm3"] == 75 * _VOX
    assert [r["tract"] for r in res["rows_with_lesion"]] == ["Corticospinal tract left"]
    assert "Pandora" in res["attribution"]
    with open(res["csv_path"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[1]["percent_tract_lesioned"] == ""  # undefined, written empty
    assert Path(res["csv_path"]).name == "sub-01_lesion_atlas-pandora_tractseg_lesionload.csv"


@needs_ants
def test_lesion_load_identity_affine_matches(tmp_path: Path, atlas_root: Path) -> None:
    """An identity .mat through apply_transforms gives the same numbers as no transform."""
    mask = _save(tmp_path / "m.nii.gz", _lesion())
    mat = _identity_mat(tmp_path / "identity.mat")
    with patch.object(_atlas, "ATLAS_ROOT", atlas_root):
        res = lesion_load_by_atlas(
            mask, "pandora_tractseg", inverse_transforms=[mat], inverse_invert_flags=[False]
        )
    (cst,) = res["corticospinal_rows"]
    assert cst["tract_volume_mm3"] == 800 * _VOX
    assert cst["lesion_in_tract_mm3"] == 100 * _VOX
    assert cst["lesion_in_tract_weighted_mm3"] == 75 * _VOX
    assert "warped" in res["method"]


@needs_ants
def test_lesion_load_translated_atlas(tmp_path: Path, atlas_root: Path) -> None:
    """A whole-voxel translation moves the tract by exactly that many voxels.

    ANTs works in LPS physical space, so a +4 mm ITK x-translation is -4 mm in the
    RAS NIfTI frame: each mask point x samples the atlas at x - 4 mm, i.e. the warped
    tract sits 2 voxels higher in x (p=1 at x 4..7, p=0.5 at x 8..11). The lesion
    (x 4..7) then lies entirely in the p=1 core.
    """
    mask = _save(tmp_path / "m.nii.gz", _lesion())
    mat = _translation_mat(tmp_path / "shift.mat", (4.0, 0.0, 0.0))
    with patch.object(_atlas, "ATLAS_ROOT", atlas_root):
        res = lesion_load_by_atlas(
            mask, "pandora_tractseg", inverse_transforms=[mat], inverse_invert_flags=[False]
        )
    (cst,) = res["corticospinal_rows"]
    assert cst["tract_volume_mm3"] == 800 * _VOX  # still fully inside the grid
    assert cst["lesion_in_tract_mm3"] == 100 * _VOX
    assert cst["lesion_in_tract_weighted_mm3"] == 100 * _VOX  # was 75 unshifted
    assert cst["percent_tract_lesioned_weighted"] == round(100 * 100 / 600, 4)


@needs_ants
def test_lesion_load_labels_and_saved_atlas(tmp_path: Path, atlas_root: Path) -> None:
    """Label atlas through ANTs (nearest neighbour), with the warped atlas saved."""
    mask = _save(tmp_path / "m.nii.gz", _label_lesion())
    with patch.object(_atlas, "ATLAS_ROOT", atlas_root):
        res = lesion_load_by_atlas(
            mask, "schaefer2018_100", mask_in_template_space=True, save_warped_atlas=True
        )
    by = {r["region"]: r for r in res["rows_with_lesion"]}
    assert set(by) == {"A", "B", "outside atlas"}
    assert by["A"]["lesion_in_region_mm3"] == 8 * _VOX
    assert res["probability_threshold"] is None
    assert res["corticospinal_rows"] == []
    assert res["warped_atlas_path"] is not None
    warped = _data(res["warped_atlas_path"])
    assert set(np.unique(warped).tolist()) == {0, 1, 2}


@needs_ants
def test_lesion_load_different_grid_uses_mask_spacing(tmp_path: Path, atlas_root: Path) -> None:
    """A 1 mm mask over the same field: volumes use the mask's own 1 mm³ voxels."""
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = -0.5  # voxel centres of the 1 mm grid nest inside the 2 mm grid
    m = np.zeros((40, 40, 40), np.uint8)
    m[8:16, 0:10, 0:10] = 1  # = lesion x 4..7 (2 mm) -> 800 voxels of 1 mm³
    mask = _save(tmp_path / "m1mm.nii.gz", m, affine)
    with patch.object(_atlas, "ATLAS_ROOT", atlas_root):
        res = lesion_load_by_atlas(mask, "pandora_tractseg", mask_in_template_space=True)
    assert res["voxel_volume_mm3"] == 1.0
    assert res["lesion_volume_mm3"] == 800.0
    (cst,) = res["corticospinal_rows"]
    assert cst["lesion_in_tract_mm3"] == 800.0


def test_lesion_load_requires_one_space(tmp_path: Path) -> None:
    """Neither or both space specifications is an error."""
    mask = _save(tmp_path / "m.nii.gz", _lesion())
    with patch("medmcp_neuro_core.tools.atlas._check_antspy"):
        with pytest.raises(ValueError, match="exactly one"):
            lesion_load_by_atlas(mask, "pandora_tractseg")
        with pytest.raises(ValueError, match="exactly one"):
            lesion_load_by_atlas(
                mask,
                "pandora_tractseg",
                inverse_transforms=[str(mask)],
                mask_in_template_space=True,
            )


def test_lesion_load_bad_threshold_and_flags(tmp_path: Path) -> None:
    """Threshold outside (0, 1] and flag/transform length mismatch are rejected."""
    mask = _save(tmp_path / "m.nii.gz", _lesion())
    with patch("medmcp_neuro_core.tools.atlas._check_antspy"):
        with pytest.raises(ValueError, match="probability_threshold"):
            lesion_load_by_atlas(
                mask, "pandora_tractseg", mask_in_template_space=True, probability_threshold=0
            )
        with pytest.raises(ValueError, match="one entry per transform"):
            lesion_load_by_atlas(
                mask,
                "pandora_tractseg",
                inverse_transforms=[str(mask)],
                inverse_invert_flags=[True, False],
            )
        with pytest.raises(FileNotFoundError):
            lesion_load_by_atlas(
                mask, "pandora_tractseg", inverse_transforms=[str(tmp_path / "nope.mat")]
            )


# ── lesion_frequency_map ───────────────────────────────────────────────────────


def test_lesion_frequency_map_template_space(tmp_path: Path) -> None:
    """Masks already on the template grid: count / n, files and figure written."""
    tmpl = _save(tmp_path / "tmpl.nii.gz", np.ones(_SHAPE, np.float32))
    masks: list[Path] = []
    for i in range(4):
        m = np.zeros(_SHAPE, np.uint8)
        m[5:8, 5:8, 5:8] = 1  # in every subject
        if i < 2:
            m[12, 12, 12] = 1  # in half
        masks.append(_save(tmp_path / f"sub-{i}_lesion.nii.gz", m))
    out = tmp_path / "out"
    with patch("medmcp_neuro_core.tools.atlas.get_mni152_1mm", return_value=tmpl):
        res = lesion_frequency_map(masks, out)
    freq = _data(res["frequency_map_path"])
    count = _data(res["count_map_path"])
    assert res["n_masks"] == 4
    assert freq[6, 6, 6] == 1.0
    assert freq[12, 12, 12] == 0.5
    assert count[12, 12, 12] == 2
    assert res["max_frequency"] == 1.0
    assert res["voxels_with_any_lesion"] == 27 + 1
    assert Path(res["figure_path"]).stat().st_size > 0
    assert res["template_space_masks"] == [str(p) for p in masks]


@needs_ants
def test_lesion_frequency_map_resamples_other_grid(tmp_path: Path) -> None:
    """A mask on a coarser grid is resampled onto the template grid first."""
    tmpl = _save(tmp_path / "tmpl.nii.gz", np.ones(_SHAPE, np.float32))
    coarse_aff = np.diag([4.0, 4.0, 4.0, 1.0])
    coarse_aff[:3, 3] = 1.0  # 4 mm voxel centres = centres of 2x2x2 blocks of 2 mm voxels
    m = np.zeros((10, 10, 10), np.uint8)
    m[2, 2, 2] = 1  # covers 2 mm voxels 4..5 in each axis
    coarse = _save(tmp_path / "coarse.nii.gz", m, coarse_aff)
    with patch("medmcp_neuro_core.tools.atlas.get_mni152_1mm", return_value=tmpl):
        res = lesion_frequency_map([coarse], tmp_path / "out")
    assert res["voxels_with_any_lesion"] == 8
    assert res["template_space_masks"][0].endswith("coarse_space-MNI152NLin2009cAsym.nii.gz")


def test_lesion_frequency_map_validation(tmp_path: Path) -> None:
    """Empty input and transform-list length mismatch are rejected."""
    mask = _save(tmp_path / "m.nii.gz", _lesion())
    with pytest.raises(ValueError, match="must not be empty"):
        lesion_frequency_map([], tmp_path)
    with pytest.raises(ValueError, match="one transform list per mask"):
        lesion_frequency_map([mask], tmp_path, forward_transforms_list=[])
