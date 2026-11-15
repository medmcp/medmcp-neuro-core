"""The user's licence file: checked once, by structure, never shown, never run without.

Every licence here is synthetic (see conftest.FAKE_LICENCE); no test reads a real one.
"""

import os
from pathlib import Path

import pytest

from medmcp_neuro_core.tools.freesurfer._license import (
    REGISTRATION_URL,
    LicenseError,
    LicenseInvalidError,
    LicenseMissingError,
    find_license,
)
from tests.conftest import write_fake_licence

# Distinctive content that must never appear in any message.
SECRET_LINES = ("secret.person@example.org", "987654321", "*SECRETKEYXYZ", "FSSECRETXYZ")


def _no_content(message: str) -> None:
    for line in SECRET_LINES:
        assert line not in message


def _hands_off(message: str) -> None:
    """Every licence message tells the model not to copy, chmod or print the file."""
    for word in ("copy", "chmod", "print"):
        assert word in message, word


def test_missing_licence_explains_what_to_do(workspace: Path) -> None:
    """Without a licence the message names the registration page and the exact path."""
    with pytest.raises(LicenseMissingError) as exc:
        find_license()
    message = str(exc.value)
    assert exc.value.status == "licence_required"
    assert REGISTRATION_URL in message
    assert str(workspace / ".medmcp" / "licenses" / "freesurfer.txt") in message
    _hands_off(message)


def test_default_location(licence: Path) -> None:
    """The default is <workspace>/.medmcp/licenses/freesurfer.txt."""
    assert find_license() == licence


def test_explicit_path_inside_workspace(workspace: Path) -> None:
    """license_path may point anywhere inside the workspace."""
    other = write_fake_licence(workspace / "keys" / "license.txt")
    assert find_license(str(other)) == other


def test_three_line_licence(workspace: Path) -> None:
    """Older licences have no fourth line; they are accepted too."""
    path = workspace / "keys" / "license.txt"
    path.parent.mkdir()
    path.write_text("someone@example.org\n12345\n *ABCDEFGHIJ\n")
    assert find_license(str(path)) == path


def test_explicit_path_outside_workspace_is_refused(workspace: Path, tmp_path: Path) -> None:
    """A licence outside the workspace is not used, with a licence-specific error."""
    outside = write_fake_licence(tmp_path / "license.txt")
    with pytest.raises(LicenseInvalidError, match="outside the workspace") as exc:
        find_license(str(outside))
    assert exc.value.status == "licence_invalid"
    assert str(exc.value).startswith("FreeSurfer licence problem")
    _hands_off(str(exc.value))


def test_empty_file_is_not_a_licence(licence: Path) -> None:
    """An empty placeholder file is refused as empty."""
    licence.write_text("")
    with pytest.raises(LicenseInvalidError, match="is empty"):
        find_license()


def test_folder_is_not_a_licence(workspace: Path) -> None:
    """Pointing at the folder instead of the file says so."""
    folder = workspace / ".medmcp" / "licenses"
    folder.mkdir(parents=True)
    with pytest.raises(LicenseInvalidError, match="not a file"):
        find_license(str(folder))


@pytest.mark.parametrize(
    "content",
    [
        "placeholder-not-a-licence\n",
        "secret.person@example.org\nnot-a-number\n *SECRETKEYXYZ\n",
        "987654321\nsecret.person@example.org\n *SECRETKEYXYZ\n",
        "secret.person@example.org\n987654321\n",
        "line\n" * 20,
    ],
)
def test_wrong_structure_is_refused_without_showing_content(workspace: Path, content: str) -> None:
    """A file not shaped like a licence is refused; the message shows none of it."""
    path = workspace / "keys" / "license.txt"
    path.parent.mkdir()
    path.write_text(content)
    with pytest.raises(LicenseInvalidError, match="structure of a FreeSurfer licence") as exc:
        find_license(str(path))
    _no_content(str(exc.value))
    assert "placeholder" not in str(exc.value)
    _hands_off(str(exc.value))


def test_binary_file_is_refused(workspace: Path) -> None:
    """A binary file (not UTF-8 text) is not a licence."""
    path = workspace / "keys" / "license.txt"
    path.parent.mkdir()
    path.write_bytes(b"\xff\xfe\x00\x01" * 8)
    with pytest.raises(LicenseInvalidError, match="structure"):
        find_license(str(path))


def test_huge_file_is_refused(workspace: Path) -> None:
    """Anything far larger than a licence is refused before parsing."""
    path = workspace / "keys" / "license.txt"
    path.parent.mkdir()
    path.write_text("x" * (70 * 1024))
    with pytest.raises(LicenseInvalidError, match="too large"):
        find_license(str(path))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_licence_is_a_licence_problem(licence: Path) -> None:
    """An unreadable file is refused with a licence error that forbids chmod."""
    licence.chmod(0)
    try:
        with pytest.raises(LicenseInvalidError, match="not allowed to read") as exc:
            find_license()
        _hands_off(str(exc.value))
    finally:
        licence.chmod(0o600)


def test_valid_licence_is_not_changed(licence: Path) -> None:
    """Checking leaves the file exactly as it was (content, mode, no copies)."""
    licence.chmod(0o600)
    before = (licence.read_bytes(), licence.stat().st_mode)
    siblings = sorted(licence.parent.iterdir())
    find_license()
    assert (licence.read_bytes(), licence.stat().st_mode) == before
    assert sorted(licence.parent.iterdir()) == siblings


def test_every_licence_error_is_a_licence_error(workspace: Path) -> None:
    """The tools catch one class for every licence problem."""
    assert issubclass(LicenseMissingError, LicenseError)
    assert issubclass(LicenseInvalidError, LicenseError)


def test_no_workspace_and_no_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a workspace there is no default location, and the message says so."""
    monkeypatch.delenv("MEDMCP_WORKSPACE", raising=False)
    monkeypatch.setattr("medmcp_neuro_core.tools.freesurfer._license.workspace_root", lambda: None)
    with pytest.raises(LicenseMissingError, match="workspace"):
        find_license()
