"""Tests for the FastSurfer-backed segment_brain tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from medmcp_neuro.tools.segmentation import segment_brain

_SUBPROCESS_RUN = "medmcp_neuro.tools.segmentation.subprocess.run"
_FIND_FASTSURFER = "medmcp_neuro.tools.segmentation._find_fastsurfer"
_FAKE_BINARY = "/opt/FastSurfer/run_fastsurfer.sh"


def _write_nifti(path: Path, zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    """Write a tiny valid NIfTI with the given voxel zooms (mm)."""
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


_FAKE_STATS = (
    "# Title Segmentation Statistics\n"
    "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean\n"
    "  1  10  9000  9123.4  Left-Thalamus   95.0\n"
    "  2  49  8800  8900.1  Right-Thalamus  94.5\n"
)


def _mock_subprocess_run(
    cmd: list[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: int,
) -> MagicMock:
    """Fake subprocess.run: create the seg map + a stats file, return success."""
    seg_path = cmd[cmd.index("--asegdkt_segfile") + 1]
    stats_path = cmd[cmd.index("--asegdkt_statsfile") + 1]
    Path(seg_path).touch()
    Path(stats_path).write_text(_FAKE_STATS)
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _run_with_mock(
    tmp_path: Path,
    filename: str = "sub-01_T1w.nii.gz",
) -> tuple[dict[str, object], MagicMock]:
    """Run segment_brain with subprocess.run mocked; return (result, mock_run).

    device="cpu" so _resolve_device short-circuits without importing torch.
    """
    inp = _write_nifti(tmp_path / filename)
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run,
    ):
        result = segment_brain(inp, device="cpu")
    return result, mock_run  # type: ignore[return-value]


# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        pytest.raises(FileNotFoundError, match="Input not found"),
    ):
        segment_brain(tmp_path / "nonexistent.nii.gz", device="cpu")


def test_missing_binary_raises(tmp_path: Path) -> None:
    """RuntimeError when FastSurfer is not installed."""
    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz")
    with (
        patch(_FIND_FASTSURFER, side_effect=RuntimeError("run_fastsurfer.sh not found")),
        pytest.raises(RuntimeError, match=r"run_fastsurfer\.sh not found"),
    ):
        segment_brain(inp, device="cpu")


# --- output location ---


def test_output_written_next_to_input_by_default(tmp_path: Path) -> None:
    """When output_dir is omitted, outputs are written next to the input."""
    result, _ = _run_with_mock(tmp_path)
    assert Path(str(result["seg_path"])).parent == tmp_path


def test_output_written_to_output_dir(tmp_path: Path) -> None:
    """When output_dir is given, outputs are written there and the directory is created."""
    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz")
    out_dir = tmp_path / "segs"
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, output_dir=out_dir, device="cpu")
    assert Path(str(result["seg_path"])).parent == out_dir
    assert out_dir.exists()


# --- subprocess arguments ---


def test_seg_only_and_device_passed(tmp_path: Path) -> None:
    """--seg_only is always passed and the requested device is forwarded."""
    _, mock_run = _run_with_mock(tmp_path)
    cmd = mock_run.call_args[0][0]
    assert "--seg_only" in cmd
    assert "--allow_root" in cmd  # stack containers run as root
    assert cmd[cmd.index("--device") + 1] == "cpu"


# --- volumes parsing ---


def test_volumes_csv_written_from_stats(tmp_path: Path) -> None:
    """The volumes CSV is derived from FastSurfer's stats (structure, volume_mm3)."""
    result, _ = _run_with_mock(tmp_path)
    content = Path(str(result["volumes_path"])).read_text()
    assert "structure,volume_mm3" in content
    assert "Left-Thalamus,9123.4" in content
    assert "Right-Thalamus,8900.1" in content


# --- return dict ---


def test_return_keys(tmp_path: Path) -> None:
    """Return dict contains all required keys."""
    result, _ = _run_with_mock(tmp_path)
    assert result["seg_path"]
    assert result["volumes_path"]
    assert result["input_path"]
    assert result["device"] == "cpu"
    assert result["_render"]


def test_seg_path_naming(tmp_path: Path) -> None:
    """seg_path uses the _dseg.mgz suffix (MGZ renders natively in the viewer)."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert str(result["seg_path"]).endswith("sub-01_T1w_dseg.mgz")


def test_volumes_path_naming(tmp_path: Path) -> None:
    """volumes_path uses the _volumes.csv suffix."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert str(result["volumes_path"]).endswith("sub-01_T1w_volumes.csv")


def test_render_contains_next_action(tmp_path: Path) -> None:
    """_render includes a NEXT ACTION directive."""
    result, _ = _run_with_mock(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])


# --- exit-code tolerance (FastSurfer's cosmetic post-step can exit non-zero) ---


def test_nonzero_exit_with_outputs_succeeds(tmp_path: Path) -> None:
    """Non-zero exit is tolerated when seg + stats were produced (cosmetic failure)."""

    def mock(cmd: list[str], *, capture_output: bool, text: bool, timeout: int) -> MagicMock:
        Path(cmd[cmd.index("--asegdkt_segfile") + 1]).touch()
        Path(cmd[cmd.index("--asegdkt_statsfile") + 1]).write_text(_FAKE_STATS)
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "cp: cannot create .../stats/aseg+DKT.stats"
        return result

    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz")
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=mock),
    ):
        result = segment_brain(inp, device="cpu")
    assert "Left-Thalamus,9123.4" in Path(str(result["volumes_path"])).read_text()


def test_nonzero_exit_without_outputs_raises(tmp_path: Path) -> None:
    """Non-zero exit AND no outputs is a genuine failure -> raise."""

    def mock(cmd: list[str], *, capture_output: bool, text: bool, timeout: int) -> MagicMock:
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "CUDA error"
        return result

    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz")
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=mock),
        pytest.raises(RuntimeError, match="FastSurfer failed"),
    ):
        segment_brain(inp, device="cpu")


# --- input sanity checks (modality + resolution) ---


def test_isotropic_t1w_has_no_warnings(tmp_path: Path) -> None:
    """A clean 1mm isotropic T1w produces no input warnings."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert result["warnings"] == []


def test_non_t1w_filename_warns_but_runs(tmp_path: Path) -> None:
    """A non-T1w contrast in the filename is a warning, not a block."""
    inp = _write_nifti(tmp_path / "sub-01_FLAIR.nii.gz")
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, device="cpu")
    assert any("non-T1w contrast" in w for w in result["warnings"])
    assert result["seg_path"]  # still ran


def test_mild_anisotropy_warns_but_runs(tmp_path: Path) -> None:
    """Mildly anisotropic resolution warns but does not block."""
    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz", zooms=(1.0, 1.0, 1.6))
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, device="cpu")
    assert any("less accurate" in w for w in result["warnings"])
    assert result["seg_path"]


def test_thick_slice_raises_without_force(tmp_path: Path) -> None:
    """Thick-slice / strongly anisotropic data is rejected unless force=True."""
    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz", zooms=(1.0, 1.0, 5.0))
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        pytest.raises(ValueError, match="supported range"),
    ):
        segment_brain(inp, device="cpu")


def test_thick_slice_runs_with_force(tmp_path: Path) -> None:
    """force=True downgrades the resolution error to a warning and runs."""
    inp = _write_nifti(tmp_path / "sub-01_T1w.nii.gz", zooms=(1.0, 1.0, 5.0))
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, device="cpu", force=True)
    assert any("force=True" in w for w in result["warnings"])
    assert result["seg_path"]


def test_unreadable_header_skips_resolution_check(tmp_path: Path) -> None:
    """A non-NIfTI file warns that resolution couldn't be read, but does not block."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.write_bytes(b"not a nifti")
    with (
        patch(_FIND_FASTSURFER, return_value=_FAKE_BINARY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, device="cpu")
    assert any("Could not read voxel resolution" in w for w in result["warnings"])
    assert result["seg_path"]
