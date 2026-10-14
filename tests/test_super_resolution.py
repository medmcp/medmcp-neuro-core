"""Tests for the ECLARE/SMORE-backed super_resolve tool (upstream CLIs mocked)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from medmcp_neuro_core.tools.super_resolution import SRMethod, super_resolve

_MOD = "medmcp_neuro_core.tools.super_resolution"
_SUBPROCESS_RUN = f"{_MOD}.subprocess.run"
_DETECT = f"{_MOD}.detect_devices"
_FIND = f"{_MOD}.find_binary"


def _write_nifti(
    path: Path,
    zooms: tuple[float, ...] = (1.0, 1.0, 4.0),
    shape: tuple[int, ...] = (8, 8, 4),
) -> Path:
    """Write a tiny NIfTI with the given voxel sizes (mm)."""
    affine = np.diag([*zooms[:3], 1.0])
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
    """Fake upstream CLI: write the super-resolved volume where upstream would."""
    exe = Path(cmd[0]).name
    if exe == "run-eclare":
        inp = Path(_arg(cmd, "--in-fpath"))
        out = Path(_arg(cmd, "--out-dir")) / inp.name.replace(".nii", "_eclare.nii")
        _write_nifti(out, (1.0, 1.0, 1.0), (8, 8, 16))
    elif exe == "smore-train":
        Path(_arg(cmd, "--weight-dir")).mkdir(parents=True)
    elif exe == "smore-test":
        _write_nifti(Path(_arg(cmd, "--out-fpath")), (1.0, 1.0, 1.0), (8, 8, 16))
    kwargs["stdout"].write("100%|##########| done\n")
    proc = MagicMock()
    proc.returncode = 0
    return proc


def _run(
    inp: Path,
    method: SRMethod = "eclare",
    devices: tuple[str, ...] = ("cpu", "cuda"),
    side_effect: Any = _fake_run,
    **kwargs: Any,
) -> tuple[dict[str, Any], MagicMock]:
    with (
        patch(_SUBPROCESS_RUN, side_effect=side_effect) as mock_run,
        patch(_DETECT, return_value=list(devices)),
        patch(_FIND, side_effect=lambda name, env: f"/app/.venv/bin/{name}"),  # pyright: ignore[reportUnknownLambdaType]
    ):
        result = super_resolve(inp, method=method, **kwargs)
    return dict(result), mock_run


def _cmds(mock_run: MagicMock) -> list[list[str]]:
    return [c.args[0] for c in mock_run.call_args_list]


# --- input validation ---


def test_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the input does not exist."""
    with pytest.raises(FileNotFoundError, match="Input not found"):
        super_resolve(tmp_path / "missing.nii.gz")


def test_isotropic_input_is_skipped(tmp_path: Path) -> None:
    """Isotropic input: no subprocess, no output, a warning, skipped=True."""
    inp = _write_nifti(tmp_path / "iso.nii.gz", (1.0, 1.0, 1.0))
    result, mock_run = _run(inp)
    assert result["skipped"] is True
    assert result["output_path"] is None
    assert any("isotropic" in w for w in result["warnings"])
    mock_run.assert_not_called()
    assert list(tmp_path.iterdir()) == [inp]


def test_4d_input_raises(tmp_path: Path) -> None:
    """4-D series are rejected before anything runs."""
    inp = _write_nifti(tmp_path / "bold.nii.gz", (1.0, 1.0, 4.0, 2.0), (8, 8, 4, 3))
    with pytest.raises(ValueError, match="3-D volume"):
        _run(inp)


def test_two_coarse_axes_raises(tmp_path: Path) -> None:
    """Two equally coarse axes are not a single through-plane acquisition."""
    inp = _write_nifti(tmp_path / "x.nii.gz", (1.0, 4.0, 4.0))
    with pytest.raises(ValueError, match="two equally coarse"):
        _run(inp)


def test_smore_rejects_anisotropic_inplane(tmp_path: Path) -> None:
    """SMORE requires isotropic in-plane resolution (upstream assertion)."""
    inp = _write_nifti(tmp_path / "x.nii.gz", (0.5, 1.0, 5.0))
    with pytest.raises(ValueError, match="isotropic in-plane"):
        _run(inp, method="smore")


def test_eclare_warns_on_anisotropic_inplane(tmp_path: Path) -> None:
    """ECLARE runs on anisotropic in-plane input but says what it does."""
    inp = _write_nifti(tmp_path / "x.nii.gz", (0.5, 1.0, 5.0))
    result, _ = _run(inp)
    assert any("In-plane resolution is anisotropic" in w for w in result["warnings"])


def test_mild_anisotropy_warns(tmp_path: Path) -> None:
    """Anisotropy between the skip and mild thresholds runs with a warning."""
    inp = _write_nifti(tmp_path / "x.nii.gz", (1.0, 1.0, 1.3))
    result, _ = _run(inp)
    assert result["skipped"] is False
    assert any("mild anisotropy" in w for w in result["warnings"])


@pytest.mark.parametrize(("kwarg", "value"), [("slice_thickness_mm", 0.0), ("n_patches", 10)])
def test_bad_parameters_raise(tmp_path: Path, kwarg: str, value: float) -> None:
    """Non-positive thickness and tiny patch counts are rejected."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    with pytest.raises(ValueError, match=kwarg):
        kwargs: dict[str, Any] = {kwarg: value}
        _run(inp, **kwargs)


# --- device ---


def test_auto_without_cuda_raises(tmp_path: Path) -> None:
    """'auto' never silently falls back to CPU."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        _run(inp, devices=("cpu",))


def test_explicit_cpu_runs_with_warning(tmp_path: Path) -> None:
    """device='cpu' runs, passes upstream's CPU gpu-id (-1), and warns."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    result, mock_run = _run(inp, devices=("cpu",), device="cpu")
    assert result["device"] == "cpu"
    assert _arg(_cmds(mock_run)[0], "--gpu-id") == "-1"
    assert any("CPU" in w for w in result["warnings"])


def test_auto_resolves_to_cuda_gpu0(tmp_path: Path) -> None:
    """'auto' with CUDA present resolves to cuda and GPU index 0."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    result, mock_run = _run(inp)
    assert result["device"] == "cuda"
    assert _arg(_cmds(mock_run)[0], "--gpu-id") == "0"


# --- ECLARE ---


def test_eclare_output_naming_and_geometry(tmp_path: Path) -> None:
    """Output is {stem}_eclare.nii.gz beside the input, with geometry read back."""
    inp = _write_nifti(tmp_path / "sub-01_FLAIR.nii.gz")
    result, mock_run = _run(inp)
    assert result["output_path"] == str(tmp_path / "sub-01_FLAIR_eclare.nii.gz")
    assert Path(result["output_path"]).exists()
    assert result["input_voxel_size_mm"] == [1.0, 1.0, 4.0]
    assert result["output_voxel_size_mm"] == [1.0, 1.0, 1.0]
    assert result["input_shape"] == [8, 8, 4]
    assert result["output_shape"] == [8, 8, 16]
    assert result["through_plane_axis"] == 2
    assert result["n_patches"] == 1_000_000
    assert mock_run.call_count == 1
    # The temporary work directory is cleaned up.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "sub-01_FLAIR.nii.gz",
        "sub-01_FLAIR_eclare.nii.gz",
    ]


def test_eclare_uncompressed_input_keeps_extension(tmp_path: Path) -> None:
    """A .nii input yields a .nii output (upstream keeps the extension)."""
    inp = _write_nifti(tmp_path / "scan.nii")
    result, _ = _run(inp)
    assert result["output_path"] == str(tmp_path / "scan_eclare.nii")


def test_eclare_output_dir(tmp_path: Path) -> None:
    """output_dir is created and used."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    out = tmp_path / "sr"
    result, _ = _run(inp, output_dir=out)
    assert Path(result["output_path"]).parent == out


def test_eclare_slice_thickness_is_relative_to_inplane(tmp_path: Path) -> None:
    """ECLARE takes thickness per in-plane voxel: 5 mm / 0.5 mm -> 10."""
    inp = _write_nifti(tmp_path / "x.nii.gz", (0.5, 0.5, 6.0))
    _, mock_run = _run(inp, slice_thickness_mm=5.0, n_patches=20000)
    cmd = _cmds(mock_run)[0]
    assert _arg(cmd, "--relative-slice-thickness") == "10"
    assert _arg(cmd, "--n-patches") == "20000"


def test_eclare_defaults_pass_no_optional_flags(tmp_path: Path) -> None:
    """Without overrides, upstream defaults (ESPRESO, default n_patches) are used."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    _, mock_run = _run(inp)
    cmd = _cmds(mock_run)[0]
    assert "--relative-slice-thickness" not in cmd
    assert "--n-patches" not in cmd


# --- SMORE ---


def test_smore_runs_train_then_test(tmp_path: Path) -> None:
    """SMORE calls smore-train then smore-test (not run-smore, which hides failures)."""
    inp = _write_nifti(tmp_path / "sub-01_T2w.nii.gz")
    result, mock_run = _run(inp, method="smore", slice_thickness_mm=4.0)
    train, test = _cmds(mock_run)
    assert Path(train[0]).name == "smore-train"
    assert Path(test[0]).name == "smore-test"
    assert _arg(train, "--slice-thickness") == "4"  # SMORE takes mm
    assert _arg(train, "--weight-dir") == _arg(test, "--weight-dir")
    assert result["output_path"] == str(tmp_path / "sub-01_T2w_smore.nii.gz")
    assert result["n_patches"] == 832_000


# --- failures ---


def test_nonzero_exit_raises_with_log_tail(tmp_path: Path) -> None:
    """An upstream failure surfaces the end of its log."""
    inp = _write_nifti(tmp_path / "x.nii.gz")

    def failing(cmd: list[str], **kwargs: Any) -> MagicMock:
        kwargs["stdout"].write("Traceback ...\nAssertionError: boom\n")
        proc = MagicMock()
        proc.returncode = 1
        return proc

    with pytest.raises(RuntimeError, match="boom"):
        _run(inp, side_effect=failing)


def test_missing_output_raises(tmp_path: Path) -> None:
    """Exit 0 without an output file is still a failure."""
    inp = _write_nifti(tmp_path / "x.nii.gz")

    def silent(cmd: list[str], **kwargs: Any) -> MagicMock:
        proc = MagicMock()
        proc.returncode = 0
        return proc

    with pytest.raises(RuntimeError, match="wrote no output"):
        _run(inp, side_effect=silent)


def test_render_mentions_next_action(tmp_path: Path) -> None:
    """The display rules tell the agent what to do next."""
    inp = _write_nifti(tmp_path / "x.nii.gz")
    result, _ = _run(inp)
    assert "NEXT ACTION" in result["_render"]


@pytest.mark.asyncio
async def test_tool_registered() -> None:
    """super_resolve is exposed by the MCP server, without an mps device option."""
    from medmcp_neuro_core.server import mcp

    tools = {t.name: t for t in await mcp.list_tools()}
    assert "super_resolve" in tools
    schema = tools["super_resolve"].inputSchema
    assert schema["properties"]["device"]["enum"] == ["auto", "cuda", "cpu"]
    assert schema["properties"]["method"]["enum"] == ["eclare", "smore"]
