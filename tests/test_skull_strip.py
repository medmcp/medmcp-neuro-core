"""Tests for skull_strip tool."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools.skull_strip import DeviceChoiceResult, SkullStripResult, skull_strip

_DETECT = "medmcp_neuro.tools.skull_strip.detect_devices"
_SUBPROCESS_RUN = "medmcp_neuro.tools.skull_strip.subprocess.run"


def _mock_subprocess_run(
    cmd: list[str],
    *,
    input: str,
    capture_output: bool,
    text: bool,
    timeout: int,
) -> MagicMock:
    """Fake subprocess.run: parse brain_path from input JSON and touch the file."""
    args = json.loads(input)
    Path(args["brain_path"]).touch()
    result = MagicMock()
    result.returncode = 0
    result.stdout = json.dumps({"ok": True})
    result.stderr = ""
    return result


def _run_with_mock(
    tmp_path: Path,
    device: str = "cpu",
    filename: str = "sub-01_T1w.nii.gz",
) -> tuple[SkullStripResult, MagicMock]:
    """Run skull_strip with subprocess.run mocked; return (result, mock_run)."""
    inp = tmp_path / filename
    inp.touch()
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run:
        result = skull_strip(inp, device=device)
    return cast(SkullStripResult, result), mock_run


# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with pytest.raises(FileNotFoundError, match="Input not found"):
        skull_strip(tmp_path / "nonexistent.nii.gz")


# --- device detection ---


def _device_choice(tmp_path: Path, devices: list[str]) -> DeviceChoiceResult:
    """Call skull_strip without a device and return the DeviceChoiceResult."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with patch(_DETECT, return_value=devices):
        result = skull_strip(inp)
    assert "recommended_device" in result  # narrows to DeviceChoiceResult
    return result


def test_device_none_always_returns_device_choice(tmp_path: Path) -> None:
    """When device=None, always returns DeviceChoiceResult without running."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with (
        patch(_DETECT, return_value=["cpu"]),
        patch(_SUBPROCESS_RUN) as mock_run,
    ):
        result = skull_strip(inp)
    choice = cast(DeviceChoiceResult, result)
    assert "available_devices" in choice
    assert "recommended_device" in choice
    assert "brain_path" in choice
    assert "NEXT ACTION" in choice["_render"]
    mock_run.assert_not_called()


def test_device_none_cpu_only_recommends_cpu(tmp_path: Path) -> None:
    """When only CPU is available, recommended_device is cpu with a duration warning."""
    choice = _device_choice(tmp_path, ["cpu"])
    assert choice["recommended_device"] == "cpu"
    assert "minutes" in choice["_render"]  # duration warning present


def test_device_none_cuda_recommends_cuda(tmp_path: Path) -> None:
    """When CUDA is available, recommended_device is cuda."""
    choice = _device_choice(tmp_path, ["cpu", "cuda"])
    assert choice["recommended_device"] == "cuda"
    assert choice["available_devices"] == ["cpu", "cuda"]


def test_device_none_mps_recommends_mps(tmp_path: Path) -> None:
    """When MPS is available, recommended_device is mps."""
    choice = _device_choice(tmp_path, ["cpu", "mps"])
    assert choice["recommended_device"] == "mps"


# --- output location ---


def test_output_written_next_to_input_by_default(tmp_path: Path) -> None:
    """When output_dir is omitted, skull-stripped file is written next to the input."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run):
        result = cast(SkullStripResult, skull_strip(inp, device="cpu"))
    assert Path(result["brain_path"]).parent == tmp_path


def test_output_written_to_output_dir(tmp_path: Path) -> None:
    """When output_dir is given, skull-stripped file is written there."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    out_dir = tmp_path / "stripped"
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run):
        result = cast(SkullStripResult, skull_strip(inp, output_dir=out_dir, device="cpu"))
    assert Path(result["brain_path"]).parent == out_dir
    assert out_dir.exists()


# --- subprocess arguments ---


def test_cpu_disables_tta(tmp_path: Path) -> None:
    """Subprocess receives use_tta=False when device is cpu."""
    _, mock_run = _run_with_mock(tmp_path, device="cpu")
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["use_tta"] is False


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_enables_tta(tmp_path: Path, device: str) -> None:
    """Subprocess receives use_tta=True for GPU/MPS devices."""
    _, mock_run = _run_with_mock(tmp_path, device=device)
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["use_tta"] is True


def test_subprocess_receives_correct_stem(tmp_path: Path) -> None:
    """Subprocess receives the NIfTI stem without extension."""
    _, mock_run = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["stem"] == "sub-01_T1w"


# --- return dict ---


def test_return_keys(tmp_path: Path) -> None:
    """Return dict contains required keys."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert result["brain_path"]
    assert result["input_path"]
    assert result["device"]
    assert result["_render"]


def test_return_paths_naming(tmp_path: Path) -> None:
    """brain_path uses _skullstripped suffix."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert result["brain_path"].endswith("sub-01_T1w_skullstripped.nii.gz")


def test_render_contains_next_action(tmp_path: Path) -> None:
    """_render includes NEXT ACTION directive."""
    result, _ = _run_with_mock(tmp_path)
    assert "NEXT ACTION" in result["_render"]
