"""Tests for skull_strip tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools.skull_strip import skull_strip

# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with pytest.raises(FileNotFoundError, match="Input not found"):
        skull_strip(tmp_path / "nonexistent.nii.gz", tmp_path / "out")


def test_missing_binary_raises(tmp_path: Path) -> None:
    """RuntimeError when hd-bet binary is not on PATH."""
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with (
        patch("medmcp_neuro.tools.skull_strip.find_binary", return_value=None),
        pytest.raises(RuntimeError, match="hd-bet"),
    ):
        skull_strip(inp, tmp_path / "out")


# --- command construction ---


def _run_with_mock(
    tmp_path: Path,
    device: str = "cpu",
    filename: str = "sub-01_T1w.nii.gz",
) -> tuple[dict[str, object], list[str]]:
    """Helper: run skull_strip with subprocess.run mocked, return (result, cmd)."""
    inp = tmp_path / filename
    inp.touch()
    out = tmp_path / "out"
    with (
        patch("medmcp_neuro.tools.skull_strip.find_binary", return_value="/fake/hd-bet"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = skull_strip(inp, out, device=device)
        cmd: list[str] = mock_run.call_args[0][0]
    return result, cmd


def test_creates_output_dir(tmp_path: Path) -> None:
    """output_dir is created if it does not exist."""
    out = tmp_path / "new" / "nested" / "dir"
    inp = tmp_path / "brain.nii.gz"
    inp.touch()
    with (
        patch("medmcp_neuro.tools.skull_strip.find_binary", return_value="/fake/hd-bet"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        skull_strip(inp, out)
    assert out.exists()


def test_command_includes_save_mask(tmp_path: Path) -> None:
    """Command always passes --save_bet_mask."""
    _, cmd = _run_with_mock(tmp_path)
    assert "--save_bet_mask" in cmd


def test_cpu_disables_tta(tmp_path: Path) -> None:
    """--disable_tta is added when device is cpu."""
    _, cmd = _run_with_mock(tmp_path, device="cpu")
    assert "--disable_tta" in cmd


def test_cuda_keeps_tta(tmp_path: Path) -> None:
    """--disable_tta is NOT added when device is cuda."""
    _, cmd = _run_with_mock(tmp_path, device="cuda")
    assert "--disable_tta" not in cmd


def test_mps_keeps_tta(tmp_path: Path) -> None:
    """--disable_tta is NOT added when device is mps."""
    _, cmd = _run_with_mock(tmp_path, device="mps")
    assert "--disable_tta" not in cmd


def test_output_prefix_uses_brain_suffix(tmp_path: Path) -> None:
    """Output prefix passed to hd-bet ends with _brain (no extension)."""
    _, cmd = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    o_idx = cmd.index("-o")
    output_prefix = cmd[o_idx + 1]
    assert output_prefix.endswith("sub-01_T1w_brain")
    assert not output_prefix.endswith(".nii.gz")


# --- return dict ---


def test_return_keys(tmp_path: Path) -> None:
    """Return dict contains required keys with correct naming."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert "brain_path" in result
    assert "mask_path" in result
    assert "input_path" in result
    assert "device" in result
    assert "_render" in result


def test_return_paths_naming(tmp_path: Path) -> None:
    """brain_path and mask_path use _brain and _brain_mask suffixes."""
    result, _ = _run_with_mock(tmp_path, filename="sub-01_T1w.nii.gz")
    assert str(result["brain_path"]).endswith("sub-01_T1w_brain.nii.gz")
    assert str(result["mask_path"]).endswith("sub-01_T1w_brain_mask.nii.gz")


def test_render_contains_next_action(tmp_path: Path) -> None:
    """_render includes NEXT ACTION directive."""
    result, _ = _run_with_mock(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])
