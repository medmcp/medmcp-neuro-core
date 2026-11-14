"""The user's licence file: found by path, never read, never run without."""

import os
from pathlib import Path

import pytest

from medmcp_neuro_core.tools.freesurfer._license import (
    REGISTRATION_URL,
    LicenseMissingError,
    find_license,
)


def test_missing_licence_explains_what_to_do(workspace: Path) -> None:
    """Without a licence the message names the registration page and the exact path."""
    with pytest.raises(LicenseMissingError) as exc:
        find_license()
    message = str(exc.value)
    assert REGISTRATION_URL in message
    assert str(workspace / ".medmcp" / "licenses" / "freesurfer.txt") in message


def test_default_location(licence: Path) -> None:
    """The default is <workspace>/.medmcp/licenses/freesurfer.txt."""
    assert find_license() == licence


def test_explicit_path_inside_workspace(workspace: Path) -> None:
    """license_path may point anywhere inside the workspace."""
    other = workspace / "keys" / "license.txt"
    other.parent.mkdir()
    other.write_text("placeholder\n")
    assert find_license(str(other)) == other


def test_explicit_path_outside_workspace_is_refused(workspace: Path, tmp_path: Path) -> None:
    """A licence outside the workspace is not used."""
    outside = tmp_path / "license.txt"
    outside.write_text("placeholder\n")
    with pytest.raises(ValueError, match="outside the workspace"):
        find_license(str(outside))


def test_empty_file_is_not_a_licence(licence: Path) -> None:
    """An empty placeholder file counts as missing."""
    licence.write_text("")
    with pytest.raises(LicenseMissingError):
        find_license()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_licence_is_never_read(licence: Path) -> None:
    """Only the path is checked: an unreadable licence file is still accepted."""
    licence.chmod(0)
    try:
        assert find_license() == licence
    finally:
        licence.chmod(0o600)


def test_no_workspace_and_no_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a workspace there is no default location, and the message says so."""
    monkeypatch.delenv("MEDMCP_WORKSPACE", raising=False)
    monkeypatch.setattr("medmcp_neuro_core.tools.freesurfer._license.workspace_root", lambda: None)
    with pytest.raises(LicenseMissingError, match="workspace"):
        find_license()
