"""Tests for the FastSurfer-backed segment_brain tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools.segmentation import segment_brain

_SUBPROCESS_RUN = "medmcp_neuro.tools.segmentation.subprocess.run"
_FIND_FASTSURFER = "medmcp_neuro.tools.segmentation._find_fastsurfer"
_FAKE_BINARY = "/opt/FastSurfer/run_fastsurfer.sh"

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
    inp = tmp_path / filename
    inp.touch()
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
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
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
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
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
