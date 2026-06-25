"""Tests for skull_strip tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools._neuro import Device
from medmcp_neuro.tools.skull_strip import skull_strip

_SUBPROCESS_RUN = "medmcp_neuro.tools.skull_strip.subprocess.run"


def _mock_subprocess_run(
    cmd: list[str],
    *,
    input: str,
    capture_output: bool,
    text: bool,
    timeout: int,
) -> MagicMock:
    """Fake subprocess.run: touch brain_path and write ok result to result_path."""
    args = json.loads(input)
    Path(args["brain_path"]).touch()
    Path(args["result_path"]).write_text(json.dumps({"ok": True}))
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _run_with_mock(
    tmp_path: Path,
    device: Device = "cpu",
    filename: str = "sub-01_T1w.nii.gz",
) -> tuple[dict[str, object], MagicMock]:
    """Run skull_strip with subprocess.run mocked; return (result, mock_run)."""
    inp = tmp_path / filename
    inp.touch()
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run:
        result = skull_strip(inp, device=device)
    return result, mock_run  # type: ignore[return-value]


# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with pytest.raises(FileNotFoundError, match="Input not found"):
        skull_strip(tmp_path / "nonexistent.nii.gz")


# --- output location ---


def test_output_written_next_to_input_by_default(tmp_path: Path) -> None:
    """When output_dir is omitted, skull-stripped file is written next to the input."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run):
        result = skull_strip(inp)
    assert Path(result["brain_path"]).parent == tmp_path


def test_output_written_to_output_dir(tmp_path: Path) -> None:
    """When output_dir is given, skull-stripped file is written there."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    out_dir = tmp_path / "stripped"
    with patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run):
        result = skull_strip(inp, output_dir=out_dir)
    assert Path(result["brain_path"]).parent == out_dir
    assert out_dir.exists()


# --- subprocess arguments ---


def test_cpu_disables_tta(tmp_path: Path) -> None:
    """Subprocess receives use_tta=False when device is cpu."""
    _, mock_run = _run_with_mock(tmp_path, device="cpu")
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["use_tta"] is False


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_enables_tta(tmp_path: Path, device: Device) -> None:
    """Subprocess receives use_tta=True for GPU/MPS devices."""
    _, mock_run = _run_with_mock(tmp_path, device=device)
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["use_tta"] is True


def test_auto_resolves_to_accelerator_and_enables_tta(tmp_path: Path) -> None:
    """device='auto' resolves via resolve_device; an accelerator enables TTA and is reported."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    with (
        patch("medmcp_neuro.tools.skull_strip.resolve_device", return_value="cuda"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run,
    ):
        result = skull_strip(inp, device="auto")
    assert result["device"] == "cuda"
    assert json.loads(mock_run.call_args[1]["input"])["use_tta"] is True


def test_auto_resolving_to_cpu_disables_tta(tmp_path: Path) -> None:
    """device='auto' resolving to cpu disables TTA and reports the resolved device."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    with (
        patch("medmcp_neuro.tools.skull_strip.resolve_device", return_value="cpu"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_subprocess_run) as mock_run,
    ):
        result = skull_strip(inp, device="auto")
    assert result["device"] == "cpu"
    assert json.loads(mock_run.call_args[1]["input"])["use_tta"] is False


def test_subprocess_receives_correct_stem(tmp_path: Path) -> None:
    """Subprocess receives the NIfTI stem without extension."""
    _, mock_run = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert call_args["stem"] == "sub-01_T1w"


def test_subprocess_receives_result_path(tmp_path: Path) -> None:
    """Subprocess receives a result_path for writing the outcome JSON."""
    _, mock_run = _run_with_mock(tmp_path)
    call_args = json.loads(mock_run.call_args[1]["input"])
    assert "result_path" in call_args


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
    assert str(result["brain_path"]).endswith("sub-01_T1w_skullstripped.nii.gz")


def test_render_contains_next_action(tmp_path: Path) -> None:
    """_render includes NEXT ACTION directive."""
    result, _ = _run_with_mock(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])
