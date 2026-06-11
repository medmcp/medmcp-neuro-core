"""Tests for the segment_ms_lesions (LST-AI) tool and its command building."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools import _run_lstai
from medmcp_neuro.tools.lesion_segmentation import segment_ms_lesions

_SUBPROCESS_RUN = "medmcp_neuro.tools.lesion_segmentation.subprocess.run"


# --- device_flag mapping ---


def test_device_flag_cpu() -> None:
    """Cpu maps to the literal 'cpu'."""
    assert _run_lstai.device_flag("cpu", 0) == "cpu"


def test_device_flag_cuda_uses_gpu_id() -> None:
    """Cuda maps to the GPU id as a string."""
    assert _run_lstai.device_flag("cuda", 2) == "2"


def test_device_flag_mps_rejected() -> None:
    """MPS is unsupported by LST-AI and raises."""
    with pytest.raises(ValueError, match="MPS is unsupported"):
        _run_lstai.device_flag("mps", 0)


# --- install discovery ---


def test_require_lst_bin_missing_raises() -> None:
    """require_lst_bin raises with install guidance when lst is absent."""
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value=None),
        pytest.raises(RuntimeError, match="LST-AI is not installed"),
    ):
        _run_lstai.require_lst_bin()


def test_require_lst_bin_returns_path() -> None:
    """require_lst_bin returns the located binary."""
    with patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"):
        assert _run_lstai.require_lst_bin() == "/x/lst"


# --- command building ---


def test_command_includes_core_flags(tmp_path: Path) -> None:
    """Command wires t1/flair/output/temp/device through; no --stripped by default."""
    cmd = _run_lstai.build_command(
        lst_bin="/x/lst",
        t1_path=tmp_path / "t1.nii.gz",
        flair_path=tmp_path / "flair.nii.gz",
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "tmp",
        device="cpu",
        skull_stripped=False,
        extra_args=[],
    )
    assert cmd[0] == "/x/lst"
    assert "--t1" in cmd and "--flair" in cmd and "--temp" in cmd
    assert "--output" in cmd and "--device" in cmd
    assert "--stripped" not in cmd


def test_command_stripped_flag(tmp_path: Path) -> None:
    """skull_stripped=True adds LST-AI's --stripped (not --skull-stripped)."""
    cmd = _run_lstai.build_command(
        lst_bin="/x/lst",
        t1_path=tmp_path / "t1.nii.gz",
        flair_path=tmp_path / "flair.nii.gz",
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "tmp",
        device="cpu",
        skull_stripped=True,
        extra_args=[],
    )
    assert "--stripped" in cmd
    assert "--skull-stripped" not in cmd


# --- tool: input validation ---


def test_missing_t1_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the T1w input is missing."""
    flair = tmp_path / "flair.nii.gz"
    flair.touch()
    with pytest.raises(FileNotFoundError, match="T1w input not found"):
        segment_ms_lesions(tmp_path / "missing_t1.nii.gz", flair)


def test_missing_flair_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the FLAIR input is missing."""
    t1 = tmp_path / "t1.nii.gz"
    t1.touch()
    with pytest.raises(FileNotFoundError, match="FLAIR input not found"):
        segment_ms_lesions(t1, tmp_path / "missing_flair.nii.gz")


# --- tool: end-to-end with mocked subprocess ---


def _mock_run_writes_outputs(
    cmd: list[str],
    *,
    capture_output: bool,
    text: bool,
    env: dict[str, str],
    timeout: int,
) -> MagicMock:
    """Fake subprocess.run writing LST-AI's real output basenames into --output."""
    out_dir = Path(cmd[cmd.index("--output") + 1])
    (out_dir / "space-flair_seg-lst.nii.gz").touch()
    (out_dir / "space-flair_desc-annotated_seg-lst.nii.gz").touch()
    (out_dir / "lesion_stats.csv").touch()
    (out_dir / "annotated_lesion_stats.csv").touch()
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _run(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    """Run segment_ms_lesions with lst/greedy/subprocess mocked."""
    t1 = tmp_path / "sub-01_T1w.nii.gz"
    flair = tmp_path / "sub-01_FLAIR.nii.gz"
    t1.touch()
    flair.touch()
    out_dir = tmp_path / "out"
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"),
        patch.object(_run_lstai, "ensure_greedy", return_value="/x/greedy"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_run_writes_outputs),
    ):
        return segment_ms_lesions(t1, flair, output_dir=out_dir, **kwargs)  # type: ignore[arg-type,return-value]


def test_run_returns_mask_annotation_and_stats(tmp_path: Path) -> None:
    """A successful run reports the mask, the annotated map, and the stats CSVs."""
    result = _run(tmp_path)
    assert str(result["lesion_mask_path"]).endswith("space-flair_seg-lst.nii.gz")
    assert str(result["annotated_path"]).endswith("space-flair_desc-annotated_seg-lst.nii.gz")
    assert len(result["output_files"]) == 2  # type: ignore[arg-type]
    assert len(result["stats_files"]) == 2  # type: ignore[arg-type]


def test_run_passes_stripped_when_requested(tmp_path: Path) -> None:
    """skull_stripped=True reaches the lst command as --stripped."""
    t1 = tmp_path / "t1.nii.gz"
    flair = tmp_path / "flair.nii.gz"
    t1.touch()
    flair.touch()
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"),
        patch.object(_run_lstai, "ensure_greedy", return_value="/x/greedy"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_run_writes_outputs) as mock_run,
    ):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", skull_stripped=True)
    assert "--stripped" in mock_run.call_args[0][0]


def test_run_prepends_sidecar_bin_to_path(tmp_path: Path) -> None:
    """The sidecar venv's bin dir is first on PATH so LST-AI's bare hd-bet resolves there."""
    import os

    t1 = tmp_path / "t1.nii.gz"
    flair = tmp_path / "flair.nii.gz"
    t1.touch()
    flair.touch()
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/opt/lstvenv/bin/lst"),
        patch.object(_run_lstai, "ensure_greedy", return_value="/cache/bin/greedy"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_run_writes_outputs) as mock_run,
    ):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out")
    path_entries = mock_run.call_args.kwargs["env"]["PATH"].split(os.pathsep)
    assert path_entries[0] == "/opt/lstvenv/bin"
    assert "/cache/bin" in path_entries


def test_run_render_has_next_action(tmp_path: Path) -> None:
    """_render includes a NEXT ACTION directive."""
    result = _run(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])


def test_no_outputs_raises(tmp_path: Path) -> None:
    """If the run produces no NIfTI files, the tool raises."""
    t1 = tmp_path / "t1.nii.gz"
    flair = tmp_path / "flair.nii.gz"
    t1.touch()
    flair.touch()
    noop = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"),
        patch.object(_run_lstai, "ensure_greedy", return_value="/x/greedy"),
        patch(_SUBPROCESS_RUN, noop),
        pytest.raises(RuntimeError, match="no new NIfTI outputs"),
    ):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out")
