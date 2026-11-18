"""Tests for shared helpers in _neuro."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from medmcp_neuro_core.tools._neuro import Device, nifti_load, nifti_save, resolve_device

_DETECT = "medmcp_neuro_core.tools._neuro.detect_devices"


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_explicit_device_passes_through(device: Device) -> None:
    """An explicit device is returned unchanged (no torch probe needed)."""
    # No patching: an explicit device must not even call detect_devices.
    with patch(_DETECT, side_effect=AssertionError("should not probe for explicit device")):
        assert resolve_device(device) == device


def test_auto_prefers_cuda() -> None:
    """Resolve 'auto' to cuda when it is available."""
    with patch(_DETECT, return_value=["cpu", "cuda", "mps"]):
        assert resolve_device("auto") == "cuda"


def test_auto_picks_mps_when_no_cuda() -> None:
    """Resolve 'auto' to mps when cuda is absent but mps is available."""
    with patch(_DETECT, return_value=["cpu", "mps"]):
        assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu() -> None:
    """Resolve 'auto' to cpu when no accelerator is available."""
    with patch(_DETECT, return_value=["cpu"]):
        assert resolve_device("auto") == "cpu"


def test_nifti_save_and_load_roundtrip(tmp_path: Path) -> None:
    """nifti_save creates parent dirs and round-trips array, affine, and zooms."""
    arr = np.zeros((4, 5, 6), dtype=np.int16)
    arr[1:3, 2:4, 0:2] = 7
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    out_path = tmp_path / "nested" / "test.nii.gz"

    written = nifti_save(arr, affine, out_path, zooms=(1.5, 2.0, 2.5))
    assert written == out_path
    assert out_path.is_file()

    loaded, loaded_aff, zooms = nifti_load(out_path, dtype=np.int64)
    assert loaded.dtype == np.int64
    assert np.array_equal(loaded, arr)
    assert np.allclose(loaded_aff, affine)
    assert np.allclose(zooms, (1.5, 2.0, 2.5))


def test_nifti_load_singleton_and_true_4d(tmp_path: Path) -> None:
    """nifti_load squeezes (X, Y, Z, 1) singleton 4-D and rejects multi-frame 4-D."""
    affine = np.eye(4)
    p_single = nifti_save(np.ones((4, 4, 4, 1), dtype=np.int16), affine, tmp_path / "single.nii.gz")
    arr, _, _ = nifti_load(p_single)
    assert arr.shape == (4, 4, 4)

    p_multi = nifti_save(np.ones((4, 4, 4, 2), dtype=np.int16), affine, tmp_path / "multi.nii.gz")
    with pytest.raises(ValueError, match="3-D volume"):
        nifti_load(p_multi)


def test_nifti_load_missing_raises(tmp_path: Path) -> None:
    """nifti_load raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        nifti_load(tmp_path / "missing.nii.gz")
