"""Tests for select_mask_of_interest (label and lesion-of-interest selection)."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from medmcp_neuro_core.tools._neuro import nifti_load, nifti_save
from medmcp_neuro_core.tools.mask import select_mask_of_interest

_AFFINE_2MM = np.diag([2.0, 2.0, 2.0, 1.0])  # 8 mm^3 per voxel


def test_select_single_label_preserves_id(tmp_path: Path) -> None:
    """When overlay_label_path is None, keeps only selected label_ids with IDs preserved."""
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[0:2, 0:2, 0:2] = 1  # 8 voxels (excluded)
    arr[3:5, 3:5, 3:5] = 2  # 8 voxels (selected)
    label_path = nifti_save(arr, _AFFINE_2MM, tmp_path / "sub-01_tumor-dseg.nii.gz")

    res = select_mask_of_interest(label_path, label_ids=2)

    assert res["overlaid_mask_path"] is None
    assert res["overlay_label_path"] is None
    assert res["selected_label_ids"] == [2]
    assert res["voxel_count"] == 8

    out_path = Path(res["selected_mask_path"])
    assert out_path.is_file()
    assert out_path.name == "sub-01_tumor-dseg_label-2_dseg.nii.gz"

    saved, _, _ = nifti_load(out_path)
    assert np.unique(saved).tolist() == [0, 2]
    assert int((saved == 2).sum()) == 8
    assert np.all(saved[0:2, 0:2, 0:2] == 0)


def test_select_multiple_fastsurfer_labels_preserves_all_ids(tmp_path: Path) -> None:
    """Supports FastSurfer .mgz inputs without overlay_label_path and never merges IDs."""
    arr: Any = np.zeros((6, 6, 6), dtype=np.int32)
    arr[0:2, 0:1, 0:1] = 17  # Left-Hippocampus: 2 voxels
    arr[4:6, 0:2, 0:1] = 53  # Right-Hippocampus: 4 voxels
    arr[2:4, 2:4, 2:4] = 2  # Left-Cerebral-White-Matter (excluded)

    mgz_path = tmp_path / "sub-01_dseg.mgz"
    mgh = nib.MGHImage(arr, np.eye(4, dtype=np.float32))
    nib.save(mgh, str(mgz_path))  # pyright: ignore[reportUnknownMemberType]

    res = select_mask_of_interest(mgz_path, label_ids=[17, 53, 17])

    assert res["selected_label_ids"] == [17, 53]
    assert res["voxel_count"] == 6
    assert res["overlaid_mask_path"] is None

    saved, _, _ = nifti_load(Path(res["selected_mask_path"]))
    assert np.unique(saved).tolist() == [0, 17, 53]
    assert int((saved == 17).sum()) == 2
    assert int((saved == 53).sum()) == 4
    assert np.all(saved[2:4, 2:4, 2:4] == 0)


def test_isolate_lesions_with_overlay_label_path(tmp_path: Path) -> None:
    """With overlay_label_path, returns filtered lesion IDs AND overlaid dseg classes."""
    lesion_map = np.zeros((8, 8, 8), dtype=np.int16)
    lesion_map[0:2, 0:3, 0:1] = 1  # Lesion 1: 6 voxels (selected)
    lesion_map[2:4, 0:2, 0:1] = 2  # Lesion 2: 4 voxels (excluded)
    lesion_map[5:7, 5:7, 5:6] = 3  # Lesion 3: 4 voxels (selected)

    dseg = np.zeros((8, 8, 8), dtype=np.int16)
    # Lesion 1 internal subregions: 2 of label 1, 3 of label 2, 1 of label 3
    dseg[0:2, 0:1, 0:1] = 1
    dseg[0:2, 1:2, 0:1] = 2
    dseg[0:1, 2:3, 0:1] = 2
    dseg[1:2, 2:3, 0:1] = 3
    # Lesion 2 internal subregion (must be zeroed out)
    dseg[2:4, 0:2, 0:1] = 3
    # Lesion 3 internal subregion: 4 of label 3
    dseg[5:7, 5:7, 5:6] = 3

    lesion_path = nifti_save(lesion_map, _AFFINE_2MM, tmp_path / "sub-01_lesion-map.nii.gz")
    dseg_path = nifti_save(dseg, _AFFINE_2MM, tmp_path / "sub-01_tumor-dseg.nii.gz")
    out_dir = tmp_path / "out"

    res = select_mask_of_interest(
        label_path=lesion_path,
        label_ids=[1, 3],
        overlay_label_path=dseg_path,
        output_dir=out_dir,
    )

    assert res["voxel_count"] == 10
    assert res["overlay_label_path"] == str(dseg_path)

    sel_path = Path(res["selected_mask_path"])
    assert res["overlaid_mask_path"] is not None
    overlay_out_path = Path(res["overlaid_mask_path"])
    assert sel_path.parent == out_dir
    assert overlay_out_path.parent == out_dir
    assert sel_path.is_file()
    assert overlay_out_path.is_file()

    # Filtered instance map preserves lesion IDs 1 and 3 (not merged!)
    sel_data, _, _ = nifti_load(sel_path)
    assert np.unique(sel_data).tolist() == [0, 1, 3]
    assert int((sel_data == 1).sum()) == 6
    assert int((sel_data == 3).sum()) == 4
    assert np.all(sel_data[2:4, 0:2, 0:1] == 0)

    # Overlaid dseg keeps tumor classes 1, 2, 3 inside lesions 1 and 3, zeroes lesion 2
    overlay_data, _, _ = nifti_load(overlay_out_path)
    assert np.unique(overlay_data).tolist() == [0, 1, 2, 3]
    assert int((overlay_data == 1).sum()) == 2
    assert int((overlay_data == 2).sum()) == 3
    assert int((overlay_data == 3).sum()) == 5
    assert np.all(overlay_data[2:4, 0:2, 0:1] == 0)


def test_empty_selection_when_ids_absent(tmp_path: Path) -> None:
    """Selecting absent label IDs produces an empty mask and reports 0 voxels."""
    arr = np.ones((4, 4, 4), dtype=np.int16)
    label_path = nifti_save(arr, _AFFINE_2MM, tmp_path / "sub-01_dseg.nii.gz")

    res = select_mask_of_interest(label_path, label_ids=99)

    assert res["voxel_count"] == 0
    assert "empty" in res["_render"].lower()
    saved, _, _ = nifti_load(Path(res["selected_mask_path"]))
    assert int(saved.sum()) == 0


def test_grid_mismatch_raises(tmp_path: Path) -> None:
    """Shape or affine mismatch between label_path and overlay_label_path raises ValueError."""
    p_label = nifti_save(np.ones((4, 4, 4), dtype=np.int16), _AFFINE_2MM, tmp_path / "lab.nii.gz")
    p_shape = nifti_save(np.ones((5, 4, 4), dtype=np.int16), _AFFINE_2MM, tmp_path / "shape.nii.gz")
    p_aff = nifti_save(
        np.ones((4, 4, 4), dtype=np.int16),
        np.diag([1.0, 1.0, 1.0, 1.0]),
        tmp_path / "aff.nii.gz",
    )

    with pytest.raises(ValueError, match="Grid mismatch"):
        select_mask_of_interest(p_label, label_ids=1, overlay_label_path=p_shape)

    with pytest.raises(ValueError, match="Grid mismatch"):
        select_mask_of_interest(p_label, label_ids=1, overlay_label_path=p_aff)


def test_invalid_label_ids_raises(tmp_path: Path) -> None:
    """Empty or non-positive label_ids raise ValueError."""
    p_label = nifti_save(np.ones((4, 4, 4), dtype=np.int16), _AFFINE_2MM, tmp_path / "lab.nii.gz")

    with pytest.raises(ValueError, match="at least one"):
        select_mask_of_interest(p_label, label_ids=[])

    with pytest.raises(ValueError, match="positive integers"):
        select_mask_of_interest(p_label, label_ids=0)

    with pytest.raises(ValueError, match="positive integers"):
        select_mask_of_interest(p_label, label_ids=[1, -2])

    with pytest.raises(ValueError, match="positive integers"):
        select_mask_of_interest(p_label, label_ids=True)


def test_missing_files_raise(tmp_path: Path) -> None:
    """Missing label_path or overlay_label_path raises FileNotFoundError."""
    p_label = nifti_save(np.ones((4, 4, 4), dtype=np.int16), _AFFINE_2MM, tmp_path / "lab.nii.gz")

    with pytest.raises(FileNotFoundError):
        select_mask_of_interest(tmp_path / "missing.nii.gz", label_ids=1)

    with pytest.raises(FileNotFoundError):
        select_mask_of_interest(
            p_label, label_ids=1, overlay_label_path=tmp_path / "missing_overlay.nii.gz"
        )
