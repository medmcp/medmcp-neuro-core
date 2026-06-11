"""Tests for the segment_lesions (LST-AI) tool and its backend dispatch."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools import _run_lstai
from medmcp_neuro.tools.lesion_segmentation import segment_lesions

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


# --- backend resolution ---


def test_resolve_backend_native_explicit_missing() -> None:
    """Requesting native with no lst binary raises with install guidance."""
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value=None),
        pytest.raises(RuntimeError, match="lst' console script"),
    ):
        _run_lstai.resolve_backend("native")


def test_resolve_backend_auto_prefers_native() -> None:
    """Auto-select returns native when the lst binary is present."""
    with patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"):
        assert _run_lstai.resolve_backend(None) == "native"


def test_resolve_backend_auto_falls_back_to_docker() -> None:
    """Auto-select returns docker when native is absent but docker exists."""
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value=None),
        patch.object(_run_lstai, "docker_available", return_value=True),
    ):
        assert _run_lstai.resolve_backend(None) == "docker"


def test_resolve_backend_env_var_forces_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """$MEDMCP_LST_AI_BACKEND overrides auto-selection even when native exists."""
    monkeypatch.setenv("MEDMCP_LST_AI_BACKEND", "docker")
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"),
        patch.object(_run_lstai, "docker_available", return_value=True),
    ):
        assert _run_lstai.resolve_backend(None) == "docker"


def test_resolve_backend_env_var_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown $MEDMCP_LST_AI_BACKEND value raises."""
    monkeypatch.setenv("MEDMCP_LST_AI_BACKEND", "podman")
    with pytest.raises(ValueError, match="MEDMCP_LST_AI_BACKEND"):
        _run_lstai.resolve_backend(None)


def test_resolve_backend_none_available() -> None:
    """Auto-select raises when neither backend is available."""
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value=None),
        patch.object(_run_lstai, "docker_available", return_value=False),
        pytest.raises(RuntimeError, match="not available via either backend"),
    ):
        _run_lstai.resolve_backend(None)


# --- command building ---


def test_native_command_includes_core_flags(tmp_path: Path) -> None:
    """Native command wires t1/flair/output/temp/device through."""
    cmd = _run_lstai.build_native_command(
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
    assert "--skull-stripped" not in cmd


def test_native_command_skull_stripped_flag(tmp_path: Path) -> None:
    """skull_stripped=True adds --skull-stripped."""
    cmd = _run_lstai.build_native_command(
        lst_bin="/x/lst",
        t1_path=tmp_path / "t1.nii.gz",
        flair_path=tmp_path / "flair.nii.gz",
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "tmp",
        device="cpu",
        skull_stripped=True,
        extra_args=[],
    )
    assert "--skull-stripped" in cmd


def test_docker_command_mounts_and_gpu(tmp_path: Path) -> None:
    """Docker command bind-mounts inputs/output and requests the GPU for cuda."""
    cmd = _run_lstai.build_docker_command(
        image="img:latest",
        t1_path=tmp_path / "t1.nii.gz",
        flair_path=tmp_path / "flair.nii.gz",
        output_dir=tmp_path / "out",
        device="0",
        skull_stripped=False,
        extra_args=[],
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in cmd
    assert "img:latest" in cmd
    # input paths are rewritten to in-container locations
    assert "/data/t1in/t1.nii.gz" in cmd
    assert "/data/out" in cmd


def test_docker_command_cpu_no_gpu(tmp_path: Path) -> None:
    """Docker command omits --gpus for cpu."""
    cmd = _run_lstai.build_docker_command(
        image="img:latest",
        t1_path=tmp_path / "t1.nii.gz",
        flair_path=tmp_path / "flair.nii.gz",
        output_dir=tmp_path / "out",
        device="cpu",
        skull_stripped=False,
        extra_args=[],
    )
    assert "--gpus" not in cmd


# --- tool: input validation ---


def test_missing_t1_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the T1w input is missing."""
    flair = tmp_path / "flair.nii.gz"
    flair.touch()
    with pytest.raises(FileNotFoundError, match="T1w input not found"):
        segment_lesions(tmp_path / "missing_t1.nii.gz", flair)


def test_missing_flair_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the FLAIR input is missing."""
    t1 = tmp_path / "t1.nii.gz"
    t1.touch()
    with pytest.raises(FileNotFoundError, match="FLAIR input not found"):
        segment_lesions(t1, tmp_path / "missing_flair.nii.gz")


# --- tool: end-to-end with mocked backend ---


def _mock_run_writes_seg(
    cmd: list[str],
    *,
    capture_output: bool,
    text: bool,
    env: dict[str, str],
    timeout: int,
) -> MagicMock:
    """Fake subprocess.run: write a seg output into the --output directory."""
    out_dir = Path(cmd[cmd.index("--output") + 1])
    (out_dir / "space-flair_seg-lst.nii.gz").touch()
    (out_dir / "space-flair_seg-lst_labeled.nii.gz").touch()
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _run_native(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    """Run segment_lesions forcing the native backend with subprocess mocked."""
    t1 = tmp_path / "sub-01_T1w.nii.gz"
    flair = tmp_path / "sub-01_FLAIR.nii.gz"
    t1.touch()
    flair.touch()
    out_dir = tmp_path / "out"
    with (
        patch.object(_run_lstai, "native_lst_bin", return_value="/x/lst"),
        patch.object(_run_lstai, "ensure_greedy", return_value="/x/greedy"),
        patch(_SUBPROCESS_RUN, side_effect=_mock_run_writes_seg),
    ):
        return segment_lesions(t1, flair, output_dir=out_dir, backend="native", **kwargs)  # type: ignore[arg-type,return-value]


def test_native_run_returns_mask_and_annotation(tmp_path: Path) -> None:
    """A successful native run reports the seg mask and the labelled map."""
    result = _run_native(tmp_path)
    assert str(result["lesion_mask_path"]).endswith("space-flair_seg-lst.nii.gz")
    assert str(result["annotated_path"]).endswith("space-flair_seg-lst_labeled.nii.gz")
    assert result["backend"] == "native"
    assert len(result["output_files"]) == 2  # type: ignore[arg-type]


def test_native_run_render_has_next_action(tmp_path: Path) -> None:
    """_render includes a NEXT ACTION directive."""
    result = _run_native(tmp_path)
    assert "NEXT ACTION" in str(result["_render"])


def test_no_outputs_raises(tmp_path: Path) -> None:
    """If the backend produces no NIfTI files, the tool raises."""
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
        segment_lesions(t1, flair, output_dir=tmp_path / "out", backend="native")
