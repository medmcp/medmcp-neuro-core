"""Tests for segment_brain tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools.segmentation import segment_brain

_SUBPROCESS_RUN = "medmcp_neuro.tools.segmentation.subprocess.run"
_FIND_SYNTHSEG = "medmcp_neuro.tools.segmentation._find_synthseg"
_FAKE_BINARY = "/usr/local/freesurfer/bin/mri_synthseg"
_FAKE_FIND_RETURN: tuple[str, dict[str, str]] = (_FAKE_BINARY, {})


def _mock_subprocess_run(
    cmd: list[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: int,
    env: object = None,
) -> MagicMock:
    """Fake subprocess.run: create output files and return success."""
    seg_path = cmd[cmd.index("--o") + 1]
    volumes_path = cmd[cmd.index("--vol") + 1]
    Path(seg_path).touch()
    Path(volumes_path).write_text("subject,structure,volume_mm3\n")
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _run_with_mock(
    tmp_path: Path,
    filename: str = "sub-01_T1w.nii.gz",
    parc: bool = True,
    robust: bool = False,
) -> tuple[dict[str, object], MagicMock]:
    """Run segment_brain with subprocess.run mocked; return (result, mock_run)."""
    inp = tmp_path / filename
    inp.touch()
    with (
        patch(_FIND_SYNTHSEG, return_value=_FAKE_FIND_RETURN),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run,
    ):
        result = segment_brain(inp, parc=parc, robust=robust)
    return result, mock_run  # type: ignore[return-value]


# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with (
        patch(_FIND_SYNTHSEG, return_value=_FAKE_FIND_RETURN),
        pytest.raises(FileNotFoundError, match="Input not found"),
    ):
        segment_brain(tmp_path / "nonexistent.nii.gz")


def test_missing_binary_raises(tmp_path: Path) -> None:
    """RuntimeError when mri_synthseg is not installed."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    with (
        patch(_FIND_SYNTHSEG, side_effect=RuntimeError("mri_synthseg not found")),
        pytest.raises(RuntimeError, match="mri_synthseg not found"),
    ):
        segment_brain(inp)


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
        patch(_FIND_SYNTHSEG, return_value=_FAKE_FIND_RETURN),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run),
    ):
        result = segment_brain(inp, output_dir=out_dir)
    assert Path(str(result["seg_path"])).parent == out_dir
    assert out_dir.exists()


# --- subprocess arguments ---


def test_parc_flag_added_when_enabled(tmp_path: Path) -> None:
    """--parc is passed to mri_synthseg when parc=True."""
    _, mock_run = _run_with_mock(tmp_path, parc=True)
    assert "--parc" in mock_run.call_args[0][0]


def test_parc_flag_absent_when_disabled(tmp_path: Path) -> None:
    """--parc is not passed when parc=False."""
    _, mock_run = _run_with_mock(tmp_path, parc=False)
    assert "--parc" not in mock_run.call_args[0][0]


def test_robust_flag_added_when_enabled(tmp_path: Path) -> None:
    """--robust is passed to mri_synthseg when robust=True."""
    _, mock_run = _run_with_mock(tmp_path, robust=True)
    assert "--robust" in mock_run.call_args[0][0]


def test_robust_flag_absent_by_default(tmp_path: Path) -> None:
    """--robust is not passed when robust=False."""
    _, mock_run = _run_with_mock(tmp_path, robust=False)
    assert "--robust" not in mock_run.call_args[0][0]


# --- return dict ---


def test_return_keys(tmp_path: Path) -> None:
    """Return dict contains all required keys."""
    result, _ = _run_with_mock(tmp_path)
    assert result["seg_path"]
    assert result["volumes_path"]
    assert result["input_path"]
    assert result["_render"]


def test_seg_path_naming(tmp_path: Path) -> None:
    """seg_path uses _dseg.nii.gz suffix."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert str(result["seg_path"]).endswith("sub-01_T1w_dseg.nii.gz")


def test_volumes_path_naming(tmp_path: Path) -> None:
    """volumes_path uses _volumes.csv suffix."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert str(result["volumes_path"]).endswith("sub-01_T1w_volumes.csv")


def test_render_label_count_parc(tmp_path: Path) -> None:
    """_render reports 95 labels when parc=True."""
    result, _ = _run_with_mock(tmp_path, parc=True)
    assert "95" in str(result["_render"])


def test_render_label_count_no_parc(tmp_path: Path) -> None:
    """_render reports 33 labels when parc=False."""
    result, _ = _run_with_mock(tmp_path, parc=False)
    assert "33" in str(result["_render"])


def test_render_contains_next_action(tmp_path: Path) -> None:
    """_render includes NEXT ACTION directive."""
    result, _ = _run_with_mock(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])
