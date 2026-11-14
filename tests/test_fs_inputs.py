"""Input checks: one 3D T1 at about 1 mm, a plain subject name, paths in the workspace."""

from pathlib import Path

import pytest

from medmcp_neuro_core.tools.freesurfer._inputs import (
    check_subject_id,
    check_t1,
    default_subject_id,
)
from medmcp_neuro_core.tools.freesurfer._workspace import detect_workspace_mount, resolve_inside
from tests.conftest import write_fs_nifti


def test_good_t1_has_no_warnings(t1: Path) -> None:
    """A 1 mm isotropic T1 inside the 256 mm field of view passes silently."""
    assert check_t1(t1) == []


def test_4d_series_is_refused(workspace: Path) -> None:
    """recon-all needs one volume; a 4D series is refused with a clear reason."""
    path = write_fs_nifti(workspace / "bold.nii.gz", (16, 16, 16, 5), (1.0, 1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="one 3D volume"):
        check_t1(path)


def test_singleton_4th_dimension_is_accepted(workspace: Path) -> None:
    """A 3D volume stored with a trailing dimension of 1 is still one volume."""
    path = write_fs_nifti(workspace / "t1.nii.gz", (16, 16, 16, 1), (1.0, 1.0, 1.0, 1.0))
    assert not any("3D" in w for w in check_t1(path))


def test_thick_slices_are_refused_unless_forced(workspace: Path) -> None:
    """Thick-slice clinical data is refused; force turns it into a warning."""
    path = write_fs_nifti(workspace / "t1.nii.gz", (128, 128, 30), (1.0, 1.0, 5.0))
    with pytest.raises(ValueError, match="force=True"):
        check_t1(path)
    warnings = check_t1(path, force=True)
    assert any("force=True" in w for w in warnings)


def test_mildly_off_resolution_warns(workspace: Path) -> None:
    """1.5 mm isotropic is outside the comfortable band: a warning, not a refusal."""
    path = write_fs_nifti(workspace / "t1.nii.gz", (100, 100, 100), (1.5, 1.5, 1.5))
    warnings = check_t1(path)
    assert len(warnings) == 1 and "outside" in warnings[0]


def test_large_field_of_view_warns(workspace: Path) -> None:
    """recon-all conforms to 256 mm; a larger field of view is cropped."""
    path = write_fs_nifti(workspace / "t1.nii.gz", (300, 64, 64), (1.0, 1.0, 1.0))
    assert any("cropped" in w for w in check_t1(path))


def test_non_t1_filename_warns(workspace: Path) -> None:
    """A file name naming FLAIR is flagged (warning only: headers carry no contrast)."""
    path = write_fs_nifti(workspace / "sub-01_FLAIR.nii.gz", (64, 64, 48), (1.0, 1.0, 1.0))
    assert any("flair" in w for w in check_t1(path))


def test_missing_and_wrong_type(workspace: Path) -> None:
    """A missing file and a non-image file are refused."""
    with pytest.raises(FileNotFoundError):
        check_t1(workspace / "nope.nii.gz")
    dcm = workspace / "slice.dcm"
    dcm.write_bytes(b"x")
    with pytest.raises(ValueError, match="NIfTI"):
        check_t1(dcm)


def test_subject_ids() -> None:
    """Subject ids are plain folder names; the default comes from the file name."""
    assert default_subject_id(Path("/x/sub-01_T1w.nii.gz")) == "sub-01_T1w"
    assert default_subject_id(Path("/x/my scan (1).mgz")) == "my_scan_1"
    assert check_subject_id("IXI002-Guys") == "IXI002-Guys"
    for bad in ("../x", "a/b", "", ".hidden", "with space"):
        with pytest.raises(ValueError):
            check_subject_id(bad)


def test_paths_must_stay_in_the_workspace(workspace: Path) -> None:
    """Relative paths and paths outside the workspace are refused."""
    assert resolve_inside(workspace / "a", "x") == workspace / "a"
    with pytest.raises(ValueError, match="absolute"):
        resolve_inside("relative/path", "x")
    with pytest.raises(ValueError, match="outside the workspace"):
        resolve_inside("/etc/passwd", "x")
    with pytest.raises(ValueError, match="outside the workspace"):
        resolve_inside(workspace / ".." / "escape", "x")


def test_workspace_mount_detection(tmp_path: Path) -> None:
    """The single top-level bind mount is the workspace; nested and system mounts are not."""
    ws = tmp_path / "ws"
    (ws / "drive").mkdir(parents=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /proc rw - proc proc rw\n"
        f"3 1 8:1 /x {ws} rw - ext4 /dev/sda1 rw\n"
        f"4 3 8:1 /y {ws}/drive rw - ext4 /dev/sda1 rw\n"
        "5 1 8:1 /z /etc/hosts rw - ext4 /dev/sda1 rw\n"
    )
    assert detect_workspace_mount(mountinfo) == ws
