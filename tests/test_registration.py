"""Tests for registration tools."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro.tools.registration import (
    ApplyTransformResult,
    CoregisterResult,
    RegisterToTemplateResult,
    TransformType,
    apply_transform,
    coregister,
    register_to_template,
)

_SUBPROCESS_RUN = "medmcp_neuro.tools.registration.subprocess.run"
_GET_TEMPLATE = "medmcp_neuro.tools.registration.get_mni152_1mm"
_CHECK_ANTSPY = "medmcp_neuro.tools.registration._check_antspy"


def _mock_register(cmd: list[str], *, input: str, **_kwargs: object) -> MagicMock:
    """Simulate _run_ants.py: touch output files and write result to result_path."""
    args: dict[str, object] = json.loads(input)
    outprefix = str(args["outprefix"])
    ants_type = str(args["type_of_transform"])
    out_registered = str(args["out_registered"])
    result_path = str(args["result_path"])

    Path(out_registered).touch()
    Path(outprefix + "0GenericAffine.mat").touch()
    fwd: list[str] = [outprefix + "0GenericAffine.mat"]
    inv: list[str] = [outprefix + "0GenericAffine.mat"]
    inv_flags: list[bool] = [True]

    if ants_type in ("SyN", "SyNQuick"):
        Path(outprefix + "1Warp.nii.gz").touch()
        Path(outprefix + "1InverseWarp.nii.gz").touch()
        fwd = [outprefix + "1Warp.nii.gz", outprefix + "0GenericAffine.mat"]
        inv = [outprefix + "0GenericAffine.mat", outprefix + "1InverseWarp.nii.gz"]
        inv_flags = [True, False]

    Path(result_path).write_text(
        json.dumps(
            {
                "ok": True,
                "fwdtransforms": fwd,
                "invtransforms": inv,
                "inverse_invert_flags": inv_flags,
            }
        )
    )
    mock = MagicMock()
    mock.returncode = 0
    mock.stderr = ""
    mock.stdout = ""
    return mock


def _mock_apply(cmd: list[str], *, input: str, **_kwargs: object) -> MagicMock:
    """Simulate _run_ants.py apply_transforms: touch the output file."""
    args: dict[str, object] = json.loads(input)
    Path(str(args["out_path"])).touch()
    Path(str(args["result_path"])).write_text(json.dumps({"ok": True}))
    mock = MagicMock()
    mock.returncode = 0
    mock.stderr = ""
    mock.stdout = ""
    return mock


# ── register_to_template ───────────────────────────────────────────────────────


def _run_register(
    tmp_path: Path,
    filename: str = "sub-01_T1w.nii.gz",
    transform_type: TransformType = "syn",
    template_name: str | None = None,
    output_dir: Path | None = None,
    skull_stripped: bool = False,
) -> RegisterToTemplateResult:
    inp = tmp_path / filename
    inp.touch()
    tmpl = tmp_path / (template_name or "MNI152.nii.gz")
    tmpl.touch()
    out = output_dir if output_dir is not None else tmp_path / "out"
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        return register_to_template(
            inp, transform_type=transform_type, skull_stripped=skull_stripped, output_dir=out
        )


def test_register_to_template_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(FileNotFoundError, match="Input not found"),
    ):
        register_to_template(
            tmp_path / "nonexistent.nii.gz", transform_type="rigid", skull_stripped=False
        )


def test_register_to_template_missing_template_raises(tmp_path: Path) -> None:
    """FileNotFoundError when the resolved template does not exist."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmp_path / "missing.nii.gz"),
        pytest.raises(FileNotFoundError, match="Template not found"),
    ):
        register_to_template(inp, transform_type="syn", skull_stripped=False)


def test_register_to_template_bids_output_name(tmp_path: Path) -> None:
    """Registered image is named {stem}_space-MNI152NLin2009cAsym.nii.gz."""
    result = _run_register(tmp_path, filename="sub-01_T1w.nii.gz", transform_type="syn")
    assert result["registered_path"].endswith("sub-01_T1w_space-MNI152NLin2009cAsym.nii.gz")


def test_register_to_template_default_output_dir(tmp_path: Path) -> None:
    """When output_dir is omitted, output is written next to the input file."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        result = register_to_template(inp, transform_type="rigid", skull_stripped=False)
    assert Path(result["registered_path"]).parent == tmp_path


def test_register_to_template_synquick_payload(tmp_path: Path) -> None:
    """``synquick`` sends type_of_transform=SyNQuick to the subprocess."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        register_to_template(inp, transform_type="synquick", skull_stripped=False)
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "SyNQuick"


def test_register_to_template_syn_payload(tmp_path: Path) -> None:
    """``syn`` sends type_of_transform=SyN to the subprocess."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        register_to_template(inp, transform_type="syn", skull_stripped=False)
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "SyN"


def test_register_to_template_syn_forward_transforms(tmp_path: Path) -> None:
    """SyN forward_transforms contains warp then affine."""
    result = _run_register(tmp_path, transform_type="syn")
    fwd = result["forward_transforms"]
    assert any("1Warp.nii.gz" in t for t in fwd)
    assert any("0GenericAffine.mat" in t for t in fwd)
    warp_idx = next(i for i, t in enumerate(fwd) if "1Warp.nii.gz" in t)
    affine_idx = next(i for i, t in enumerate(fwd) if "0GenericAffine.mat" in t)
    assert warp_idx < affine_idx


def test_register_to_template_syn_inverse_transforms(tmp_path: Path) -> None:
    """SyN inverse_transforms contains affine and InverseWarp."""
    result = _run_register(tmp_path, transform_type="syn")
    inv = result["inverse_transforms"]
    assert any("0GenericAffine.mat" in t for t in inv)
    assert any("1InverseWarp.nii.gz" in t for t in inv)


def test_register_to_template_syn_inverse_invert_flags(tmp_path: Path) -> None:
    """SyN inverse_invert_flags: True for affine, False for InverseWarp."""
    result = _run_register(tmp_path, transform_type="syn")
    assert result["inverse_invert_flags"] == [True, False]


def test_register_to_template_affine_transforms(tmp_path: Path) -> None:
    """Affine: single mat file in forward_transforms; inverse_invert_flags=[True]."""
    result = _run_register(tmp_path, transform_type="affine")
    assert len(result["forward_transforms"]) == 1
    assert result["forward_transforms"][0].endswith("0GenericAffine.mat")
    assert result["inverse_invert_flags"] == [True]


def test_register_to_template_similarity_payload(tmp_path: Path) -> None:
    """Similarity sends type_of_transform=Similarity to the subprocess."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        register_to_template(inp, transform_type="similarity", skull_stripped=False)
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "Similarity"


def test_register_to_template_custom_template_space_label(tmp_path: Path) -> None:
    """Space label is derived from a custom template's filename."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    custom_tmpl = tmp_path / "tpl_MNIPediatricAsym.nii.gz"
    custom_tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        result = register_to_template(
            inp, transform_type="rigid", skull_stripped=False, template_path=custom_tmpl
        )
    assert "MNIPediatricAsym" in result["registered_path"]


def test_register_to_template_render_has_next_action(tmp_path: Path) -> None:
    """_render contains NEXT ACTION directive."""
    result = _run_register(tmp_path)
    assert "NEXT ACTION" in result["_render"]


def test_register_to_template_skull_stripped_calls_brain_template(tmp_path: Path) -> None:
    """skull_stripped=True passes skull_stripped=True to get_mni152_1mm."""
    inp = tmp_path / "sub-01_T1w_desc-brain.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152_brain.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl) as mock_tmpl,
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        register_to_template(
            inp, transform_type="rigid", output_dir=tmp_path / "out", skull_stripped=True
        )
    mock_tmpl.assert_called_once_with(skull_stripped=True)


def test_register_to_template_no_skull_stripped_calls_full_template(tmp_path: Path) -> None:
    """skull_stripped=False passes skull_stripped=False to get_mni152_1mm."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl) as mock_tmpl,
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        register_to_template(
            inp, transform_type="rigid", skull_stripped=False, output_dir=tmp_path / "out"
        )
    mock_tmpl.assert_called_once_with(skull_stripped=False)


def test_register_to_template_custom_template_ignores_skull_stripped(tmp_path: Path) -> None:
    """When template_path is provided, skull_stripped has no effect on template selection."""
    inp = tmp_path / "sub-01_T1w_desc-brain.nii.gz"
    inp.touch()
    custom_tmpl = tmp_path / "tpl_custom_brain.nii.gz"
    custom_tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE) as mock_tmpl,
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        result = register_to_template(
            inp,
            transform_type="rigid",
            output_dir=tmp_path / "out",
            template_path=custom_tmpl,
            skull_stripped=True,
        )
    mock_tmpl.assert_not_called()
    assert result["template_path"] == str(custom_tmpl)


def test_register_to_template_subprocess_receives_result_path(tmp_path: Path) -> None:
    """Subprocess payload includes result_path for writing the outcome JSON."""
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    tmpl = tmp_path / "MNI152.nii.gz"
    tmpl.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_GET_TEMPLATE, return_value=tmpl),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        register_to_template(inp, transform_type="rigid", skull_stripped=False)
    payload = json.loads(mock_run.call_args[1]["input"])
    assert "result_path" in payload


# ── coregister ─────────────────────────────────────────────────────────────────


def _run_coregister(
    tmp_path: Path,
    fixed_name: str = "sub-01_T1w.nii.gz",
    moving_names: list[str] | None = None,
    transform_type: str = "rigid",
    output_dir: Path | None = None,
) -> CoregisterResult:
    if moving_names is None:
        moving_names = ["sub-01_FLAIR.nii.gz"]
    fixed = tmp_path / fixed_name
    fixed.touch()
    moving = [tmp_path / n for n in moving_names]
    for m in moving:
        m.touch()
    out = output_dir if output_dir is not None else tmp_path / "out"
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        return coregister(fixed, moving, transform_type=transform_type, output_dir=out)  # type: ignore[arg-type]


def test_coregister_missing_fixed_raises(tmp_path: Path) -> None:
    """FileNotFoundError when fixed_path does not exist."""
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(FileNotFoundError, match="Fixed image not found"),
    ):
        coregister(tmp_path / "nonexistent.nii.gz", [moving], transform_type="rigid")


def test_coregister_missing_moving_raises(tmp_path: Path) -> None:
    """FileNotFoundError when a moving image does not exist."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(FileNotFoundError, match="Moving image"),
    ):
        coregister(fixed, [tmp_path / "nonexistent.nii.gz"], transform_type="rigid")


def test_coregister_empty_moving_raises(tmp_path: Path) -> None:
    """ValueError when moving_paths is empty."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(ValueError, match="moving_paths must not be empty"),
    ):
        coregister(fixed, [], transform_type="rigid")


def test_coregister_bids_output_name(tmp_path: Path) -> None:
    """Registered image is named {moving_stem}_space-{fixed_label}.nii.gz."""
    result = _run_coregister(
        tmp_path,
        fixed_name="sub-01_T1w.nii.gz",
        moving_names=["sub-01_FLAIR.nii.gz"],
    )
    assert result["registered_paths"][0].endswith("sub-01_FLAIR_space-T1w.nii.gz")


def test_coregister_default_output_dir(tmp_path: Path) -> None:
    """When output_dir is omitted, output is written next to the fixed image."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register),
    ):
        result = coregister(fixed, [moving], transform_type="rigid")
    assert Path(result["registered_paths"][0]).parent == tmp_path


def test_coregister_multiple_moving(tmp_path: Path) -> None:
    """All moving images produce registered outputs and transform prefixes."""
    result = _run_coregister(
        tmp_path,
        moving_names=["sub-01_FLAIR.nii.gz", "sub-01_T2w.nii.gz", "sub-01_b0.nii.gz"],
    )
    assert len(result["registered_paths"]) == 3
    assert len(result["transform_prefixes"]) == 3


def test_coregister_rigid_payload(tmp_path: Path) -> None:
    """Rigid sends type_of_transform=Rigid in subprocess payload."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        coregister(fixed, [moving], transform_type="rigid")
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "Rigid"


def test_coregister_affine_payload(tmp_path: Path) -> None:
    """Affine sends type_of_transform=Affine in subprocess payload."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        coregister(fixed, [moving], transform_type="affine")
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "Affine"


def test_coregister_render_has_next_action(tmp_path: Path) -> None:
    """_render contains NEXT ACTION directive."""
    result = _run_coregister(tmp_path)
    assert "NEXT ACTION" in result["_render"]


def test_coregister_render_lists_registered_paths(tmp_path: Path) -> None:
    """_render embeds the actual registered image paths."""
    result = _run_coregister(tmp_path, moving_names=["sub-01_FLAIR.nii.gz", "sub-01_T2w.nii.gz"])
    for p in result["registered_paths"]:
        assert p in result["_render"]


def test_coregister_similarity_payload(tmp_path: Path) -> None:
    """Similarity sends type_of_transform=Similarity in subprocess payload."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        coregister(fixed, [moving], transform_type="similarity")
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["type_of_transform"] == "Similarity"


def test_coregister_syn_forward_transforms(tmp_path: Path) -> None:
    """SyN coregister captures warp + affine in forward_transforms_list."""
    result = _run_coregister(tmp_path, transform_type="syn")
    fwd = result["forward_transforms_list"][0]
    assert any("1Warp.nii.gz" in t for t in fwd)
    assert any("0GenericAffine.mat" in t for t in fwd)


def test_coregister_synquick_inverse_invert_flags(tmp_path: Path) -> None:
    """SyNQuick coregister has correct inverse_invert_flags_list."""
    result = _run_coregister(tmp_path, transform_type="synquick")
    assert result["inverse_invert_flags_list"][0] == [True, False]


def test_coregister_rigid_forward_transforms_list(tmp_path: Path) -> None:
    """Rigid coregister forward_transforms_list contains the affine mat."""
    result = _run_coregister(tmp_path, transform_type="rigid")
    fwd = result["forward_transforms_list"][0]
    assert len(fwd) == 1
    assert fwd[0].endswith("0GenericAffine.mat")


def test_coregister_subprocess_receives_result_path(tmp_path: Path) -> None:
    """Subprocess payload includes result_path for writing the outcome JSON."""
    fixed = tmp_path / "sub-01_T1w.nii.gz"
    fixed.touch()
    moving = tmp_path / "sub-01_FLAIR.nii.gz"
    moving.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_register) as mock_run,
    ):
        coregister(fixed, [moving], transform_type="rigid")
    payload = json.loads(mock_run.call_args[1]["input"])
    assert "result_path" in payload


# ── apply_transform ────────────────────────────────────────────────────────────


def _run_apply(
    tmp_path: Path,
    input_name: str = "sub-01_brainmask.nii.gz",
    ref_name: str = "sub-01_T1w_space-MNI152NLin2009cAsym.nii.gz",
    transforms: list[str] | None = None,
    interpolation: str = "Linear",
    output_space: str | None = None,
    invert_flags: list[bool] | None = None,
    output_dir: Path | None = None,
) -> ApplyTransformResult:
    inp = tmp_path / input_name
    ref = tmp_path / ref_name
    inp.touch()
    ref.touch()
    tfs = transforms or ["/path/to/1Warp.nii.gz", "/path/to/0GenericAffine.mat"]
    out = output_dir if output_dir is not None else tmp_path / "out"
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_apply),
    ):
        return apply_transform(
            inp,
            ref,
            tfs,
            output_dir=out,
            interpolation=interpolation,  # type: ignore[arg-type]
            output_space=output_space,
            invert_flags=invert_flags,
        )


def test_apply_transform_missing_input_raises(tmp_path: Path) -> None:
    """FileNotFoundError when input_path does not exist."""
    ref = tmp_path / "ref.nii.gz"
    ref.touch()
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(FileNotFoundError, match="Input not found"),
    ):
        apply_transform(tmp_path / "nonexistent.nii.gz", ref, [])


def test_apply_transform_missing_reference_raises(tmp_path: Path) -> None:
    """FileNotFoundError when reference_path does not exist."""
    inp = tmp_path / "sub-01_brainmask.nii.gz"
    inp.touch()
    with (
        patch(_CHECK_ANTSPY),
        pytest.raises(FileNotFoundError, match="Reference not found"),
    ):
        apply_transform(inp, tmp_path / "nonexistent.nii.gz", [])


def test_apply_transform_bids_output_name_from_reference(tmp_path: Path) -> None:
    """Space label is inferred from the last BIDS entity of reference_path."""
    result = _run_apply(
        tmp_path,
        input_name="sub-01_brainmask.nii.gz",
        ref_name="sub-01_T1w.nii.gz",
    )
    assert result["output_path"].endswith("sub-01_brainmask_space-T1w.nii.gz")


def test_apply_transform_default_output_dir(tmp_path: Path) -> None:
    """When output_dir is omitted, output is written next to the input file."""
    result = _run_apply(tmp_path, output_dir=None)
    assert result["output_path"].startswith(str(tmp_path) + "/")


def test_apply_transform_explicit_output_space(tmp_path: Path) -> None:
    """Explicit output_space overrides filename-derived label."""
    result = _run_apply(
        tmp_path,
        ref_name="MNI152NLin2009cAsym_res-01_T1w.nii.gz",
        output_space="MNI152NLin2009cAsym",
    )
    assert "space-MNI152NLin2009cAsym" in result["output_path"]


def test_apply_transform_interpolation_in_payload(tmp_path: Path) -> None:
    """Interpolation is forwarded in the subprocess JSON payload."""
    inp = tmp_path / "sub-01_brainmask.nii.gz"
    ref = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    ref.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_apply) as mock_run,
    ):
        apply_transform(inp, ref, [], interpolation="NearestNeighbor")
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["interpolator"] == "NearestNeighbor"


def test_apply_transform_transforms_in_payload(tmp_path: Path) -> None:
    """Transform paths are forwarded in the subprocess JSON payload."""
    tfs = ["/warp/1Warp.nii.gz", "/warp/0GenericAffine.mat"]
    inp = tmp_path / "sub-01_brainmask.nii.gz"
    ref = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    ref.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_apply) as mock_run,
    ):
        apply_transform(inp, ref, tfs)
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["transforms"] == tfs


def test_apply_transform_invert_flags_in_payload(tmp_path: Path) -> None:
    """invert_flags are forwarded in the subprocess JSON payload."""
    inp = tmp_path / "sub-01_brainmask.nii.gz"
    ref = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    ref.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_apply) as mock_run,
    ):
        apply_transform(inp, ref, ["/t.mat"], invert_flags=[True])
    payload = json.loads(mock_run.call_args[1]["input"])
    assert payload["invert_flags"] == [True]


def test_apply_transform_no_invert_flags_omits_key(tmp_path: Path) -> None:
    """When invert_flags is None, the payload key is omitted."""
    inp = tmp_path / "sub-01_brainmask.nii.gz"
    ref = tmp_path / "sub-01_T1w.nii.gz"
    inp.touch()
    ref.touch()
    with (
        patch(_CHECK_ANTSPY),
        patch(_SUBPROCESS_RUN, side_effect=_mock_apply) as mock_run,
    ):
        apply_transform(inp, ref, [])
    payload = json.loads(mock_run.call_args[1]["input"])
    assert "invert_flags" not in payload


def test_apply_transform_render_has_next_action(tmp_path: Path) -> None:
    """_render contains NEXT ACTION directive."""
    result = _run_apply(tmp_path)
    assert "NEXT ACTION" in result["_render"]
