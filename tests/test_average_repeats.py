"""Tests for average_repeats on synthetic volumes (no real scans)."""

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from medmcp_neuro_core.tools import average_repeats as avg
from medmcp_neuro_core.tools.average_repeats import average_repeats, default_output_name


def _phantom(shift_vox: int = 0, seed: int = 0) -> Any:
    """An asymmetric synthetic 'head' (ellipsoid with off-centre blobs) plus noise."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:64, 0:64, 0:64].astype(np.float32)
    x = x - shift_vox
    head = ((x - 32) ** 2 / 22**2 + (y - 32) ** 2 / 26**2 + (z - 32) ** 2 / 24**2) < 1
    texture = 15 * np.sin(x / 5) * np.cos(y / 7) + 10 * np.sin(z / 4)
    vol = head * (80.0 + texture)
    for cx, cy, cz, r, v in ((24, 36, 30, 7, 60), (40, 22, 38, 5, -40), (34, 44, 22, 4, 90)):
        vol = vol + (((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) < r**2) * v
    noise = rng.normal(0, 2, (64, 64, 64)).astype(np.float32)
    return (np.asarray(vol, dtype=np.float32) + noise).astype(np.float32)


def _save(path: Path, data: Any, zooms: float = 1.0, offset: float = 0.0) -> Path:
    affine = np.diag([zooms, zooms, zooms, 1.0])
    affine[0, 3] = offset
    nib.save(nib.Nifti1Image(data, affine), str(path))  # pyright: ignore[reportUnknownMemberType]
    return path


@pytest.fixture(autouse=True)
def _workspace(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MEDMCP_FREESURFER_HOME", str(tmp_path / "no-freesurfer"))


def _repeats(tmp_path: Path, shifts: tuple[int, ...] = (0, 0, 0)) -> list[str]:
    return [
        str(_save(tmp_path / f"sub-01_ses-1_run-0{i + 1}_T1w.nii.gz", _phantom(s, seed=i)))
        for i, s in enumerate(shifts)
    ]


def test_default_output_names() -> None:
    """BIDS names keep the suffix last; others get _average."""
    assert (
        default_output_name(Path("sub-01_ses-1_run-01_T1w.nii.gz"))
        == "sub-01_ses-1_desc-average_T1w.nii.gz"
    )
    assert (
        default_output_name(Path("OAS2_0001_MR1_mpr-1_anon.nii.gz"))
        == "OAS2_0001_MR1_anon_average.nii.gz"
    )


def test_identical_grid_plain_mean(tmp_path: Path) -> None:
    """register='none' writes the exact voxel-wise mean, float32, first scan's header."""
    paths = _repeats(tmp_path)
    res = average_repeats(input_paths=paths)
    group = res["groups"][0]
    assert group["status"] == "averaged"
    assert list(res)[:2] == ["tell_the_user", "written"]
    assert 1 <= len(res["tell_the_user"]) <= 5
    assert all(len(line) <= 120 for line in res["tell_the_user"])
    out = Path(group["output"])
    assert out.name == "sub-01_ses-1_desc-average_T1w.nii.gz"
    img: Any = nib.load(str(out))  # pyright: ignore[reportUnknownMemberType]
    first: Any = nib.load(paths[0])  # pyright: ignore[reportUnknownMemberType]
    scans: list[Any] = [nib.load(p) for p in paths]  # pyright: ignore[reportUnknownMemberType]
    expected: Any = np.mean([np.asarray(s.dataobj) for s in scans], axis=0)
    assert img.get_data_dtype() == np.float32
    np.testing.assert_allclose(np.asarray(img.dataobj), expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(img.affine, first.affine)
    assert group["flagged_scans"] == []
    assert all(r["corr_to_reference"] > 0.95 for r in group["qc"])
    for key in ("png", "log_json", "qc_csv"):
        assert Path(group[key]).is_file()
    assert sorted(res["written"]) == sorted(
        [group["output"], group["png"], group["log_json"], group["qc_csv"]]
    )
    log = json.loads(Path(group["log_json"]).read_text())
    assert log["register"] == "none" and len(log["inputs"]) == 3 and log["qc"]


def test_grid_mismatch_refused_for_none(tmp_path: Path) -> None:
    """Different voxel sizes: no plain mean, a reason and a hint to use rigid."""
    a = _save(tmp_path / "a_run-01_T1w.nii.gz", _phantom())
    b = _save(tmp_path / "a_run-02_T1w.nii.gz", _phantom(), zooms=1.2)
    res = average_repeats(input_paths=[str(a), str(b)])
    group = res["groups"][0]
    assert group["status"] == "refused"
    assert "voxel size" in group["reason"]
    assert any("rigid" in line for line in res["tell_the_user"])
    assert res["written"] == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [a.name, b.name]


def test_different_contrast_refused(tmp_path: Path) -> None:
    """A T1w and a T2w are never averaged."""
    a = _save(tmp_path / "s_run-01_T1w.nii.gz", _phantom())
    b = _save(tmp_path / "s_run-01_T2w.nii.gz", _phantom())
    res = average_repeats(input_paths=[str(a), str(b)])
    assert res["groups"][0]["status"] == "refused"
    assert "different sequences" in res["tell_the_user"][0]


def test_shifted_scan_flagged(tmp_path: Path) -> None:
    """A scan moved by 3 voxels is flagged, the plain mean still runs and warns."""
    paths = _repeats(tmp_path, shifts=(0, 0, 3))
    res = average_repeats(input_paths=paths)
    group = res["groups"][0]
    assert group["status"] == "averaged"
    assert group["flagged_scans"] == [3]
    assert group["qc"][2]["shift_mm"] == pytest.approx(3.0, abs=0.5)  # pyright: ignore[reportUnknownMemberType]
    assert group["qc"][0]["shift_mm"] < 0.5
    assert any(
        "Scan 3 (run-03) shifted" in line and "rigid" in line for line in res["tell_the_user"]
    )


def test_outputs_not_overwritten(tmp_path: Path) -> None:
    """A second call refuses to replace the outputs unless overwrite=True."""
    paths = _repeats(tmp_path)
    first = average_repeats(input_paths=paths)
    out = Path(first["groups"][0]["output"])
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="already exist"):
        average_repeats(input_paths=paths)
    assert out.read_bytes() == before
    again = average_repeats(input_paths=paths, overwrite=True)
    assert again["groups"][0]["output"] == str(out)


def test_output_never_replaces_an_input(tmp_path: Path) -> None:
    """output_path pointing at a source is refused even with overwrite."""
    paths = _repeats(tmp_path)
    with pytest.raises(ValueError, match="overwrite an input"):
        average_repeats(input_paths=paths, output_path=paths[0], overwrite=True)


def test_folder_pattern_exclude_and_groups(tmp_path: Path) -> None:
    """A folder and pattern find the repeats; exclude leaves one out; groups batch sessions."""
    for ses in ("1", "2"):
        d = tmp_path / f"ses-{ses}"
        d.mkdir()
        for i in range(3):
            _save(d / f"sub-01_ses-{ses}_run-0{i + 1}_T1w.nii.gz", _phantom(seed=i))
    res = average_repeats(
        folder=str(tmp_path / "ses-1"),
        pattern="*_T1w.nii.gz",
        exclude=["sub-01_ses-1_run-03_T1w.nii.gz"],
    )
    group = res["groups"][0]
    assert len(group["inputs"]) == 2 and len(group["excluded"]) == 1
    # Running again on the folder must not pick up the average it wrote.
    res2 = average_repeats(
        folder=str(tmp_path / "ses-1"), pattern="*_T1w.nii.gz", output_dir=str(tmp_path / "o")
    )
    assert len(res2["groups"][0]["inputs"]) == 3
    batch = average_repeats(
        groups=[
            sorted(str(p) for p in (tmp_path / "ses-2").glob("*.nii.gz")),
            sorted(str(p) for p in (tmp_path / "ses-1").glob("*run-*.nii.gz")),
        ],
        output_dir=str(tmp_path / "batch"),
    )
    assert [g["status"] for g in batch["groups"]] == ["averaged", "averaged"]


def test_rigid_falls_back_to_antspy_without_freesurfer(tmp_path: Path) -> None:
    """register='rigid' with no FreeSurfer: ANTsPy aligns the shifted scan, mean is sharp."""
    paths = _repeats(tmp_path, shifts=(0, 3))
    res = average_repeats(input_paths=paths, register="rigid")
    group = res["groups"][0]
    assert group["status"] == "averaged"
    assert group["engine"].startswith("ANTsPy rigid")
    assert any("FreeSurfer not used" in line for line in res["tell_the_user"])
    assert group["qc"][1]["corr_after_rigid"] > group["qc"][1]["corr_to_reference"]
    mats = [p for p in res["written"] if p.endswith(".mat")]
    assert len(mats) == 1


def test_rigid_uses_mri_robust_template_with_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FreeSurfer and a licence, mri_robust_template runs with --satit and the LTAs."""
    home = tmp_path / "fs"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "mri_robust_template").write_text("#!/bin/sh\n")
    monkeypatch.setenv("MEDMCP_FREESURFER_HOME", str(home))
    lic = tmp_path / ".medmcp" / "licenses" / "freesurfer.txt"
    lic.parent.mkdir(parents=True)
    lic.write_text("someone@example.org\n12345\n abcdefg\n hijklmn\n")
    paths = _repeats(tmp_path, shifts=(0, 1))
    seen: dict[str, Any] = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        if cmd[0].endswith("mri_robust_template"):
            seen["cmd"] = cmd
            seen["env"] = kwargs["env"]
            template = cmd[cmd.index("--template") + 1]
            _save(Path(template), _phantom())
            for lta in cmd[cmd.index("--lta") + 1 :]:
                Path(lta).write_text("lta\n")
            return _Done()
        return real_run(cmd, **kwargs)  # pyright: ignore[reportUnknownVariableType]

    real_run = avg.subprocess.run
    monkeypatch.setattr(avg.subprocess, "run", fake_run)
    res = average_repeats(input_paths=paths, register="rigid")
    group = res["groups"][0]
    assert group["engine"].startswith("FreeSurfer mri_robust_template")
    assert "--satit" in seen["cmd"]
    assert seen["env"]["FS_LICENSE"] == str(lic)
    assert len([p for p in res["written"] if p.endswith(".lta")]) == 2
    log_text = Path(group["log_json"]).read_text()
    assert "abcdefg" not in log_text and "abcdefg" not in json.dumps(res)
