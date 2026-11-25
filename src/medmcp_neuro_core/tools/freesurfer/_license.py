"""Finding and checking the user's FreeSurfer licence key, without ever showing it.

FreeSurfer's programs refuse to run without a licence file (a free registration
emails one). It is the user's personal key: this stack never copies it, moves it,
changes its permissions, logs it, prints it or puts it in a result, and no image
ships one. The tools check it once per call -- the file exists, can be read, and
has the structure of a FreeSurfer licence (an email address, a number, a key) -- and
hand its *path* to FreeSurfer as ``FS_LICENSE``. The check reads the few lines in
memory and keeps nothing; no error message contains any of the file's content.

Every licence problem raises a :class:`LicenseError` whose message is about the
licence only, so a licence problem is never confused with anything else (for
example MedMCP's background-job slot being busy, which is not a licence problem).
"""

from pathlib import Path

from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root

REGISTRATION_URL = "https://surfer.nmr.mgh.harvard.edu/registration.html"
# Where the tools look by default, relative to the workspace.
DEFAULT_RELATIVE = Path(".medmcp") / "licenses" / "freesurfer.txt"
# A FreeSurfer licence is a few lines; anything far larger is not one.
_MAX_BYTES = 64 * 1024
# license.txt: email, number, key (and, since FreeSurfer 7, a second key line).
_MIN_LINES = 3
_MAX_LINES = 8

# What the model must never do with the file, repeated in every licence message.
HANDS_OFF = (
    "Never copy, move, chmod, open or print the licence file, and never ask for its "
    "contents: pass its path as it is, and let the user fix the file."
)


class LicenseError(Exception):
    """The FreeSurfer licence cannot be used; the message says what the user must do.

    ``status`` is what the tools report: ``licence_required`` when there is no
    licence file, ``licence_invalid`` when there is one that cannot be used.
    """

    status = "licence_invalid"


class LicenseMissingError(LicenseError, FileNotFoundError):
    """No FreeSurfer licence file at the given or default path."""

    status = "licence_required"


class LicenseInvalidError(LicenseError, ValueError):
    """A file is there but is not a usable FreeSurfer licence.

    Unreadable, not shaped like a licence, a folder, or outside the workspace.
    """


def instructions(expected: Path | None) -> str:
    """How to get and place a licence, for the user."""
    where = (
        str(expected)
        if expected is not None
        else f"<workspace>/{DEFAULT_RELATIVE.as_posix()} (inside your MedMCP workspace)"
    )
    return (
        "FreeSurfer needs a licence file, and none was found. It is free: register at "
        f"{REGISTRATION_URL} and FreeSurfer emails you a file called license.txt. Save "
        f"that file as {where}, then ask again. The licence is personal: it stays in "
        "your workspace and is never copied into a tool image, a log or a result. " + HANDS_OFF
    )


def _invalid(path: Path, problem: str, remedy: str) -> LicenseInvalidError:
    return LicenseInvalidError(
        f"FreeSurfer licence problem: {path} {problem}. {remedy} This is a problem with "
        f"the licence file only. {HANDS_OFF}"
    )


def _looks_like_licence(data: bytes) -> tuple[bool, int]:
    """Whether *data* has a FreeSurfer licence's structure; also its non-empty line count.

    The structure only: an email address on the first line, a number on the second
    and a key (one word) on the third. Nothing of the content is returned.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False, 0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not _MIN_LINES <= len(lines) <= _MAX_LINES:
        return False, len(lines)
    email, number, key = lines[0], lines[1], lines[2]
    ok = "@" in email and len(email.split()) == 1 and number.isdigit() and len(key.split()) == 1
    return ok, len(lines)


def check_license(path: Path) -> None:
    """Check that *path* is a readable file shaped like a FreeSurfer licence.

    Raises:
        LicenseMissingError: nothing at *path*.
        LicenseInvalidError: a folder, unreadable, empty, too large, or not shaped
            like a FreeSurfer licence.
    """
    if not path.exists():
        raise LicenseMissingError(instructions(path))
    if not path.is_file():
        raise _invalid(
            path,
            "is not a file (it is a folder or something else)",
            "Pass the path of the license.txt file itself.",
        )
    try:
        with path.open("rb") as fh:
            data = fh.read(_MAX_BYTES + 1)
    except PermissionError:
        raise _invalid(
            path,
            "exists but MedMCP is not allowed to read it",
            "The user must make the file readable for MedMCP themselves.",
        ) from None
    except OSError as exc:
        raise _invalid(
            path,
            f"exists but could not be read ({type(exc).__name__})",
            "The user must check the file.",
        ) from None
    if not data.strip():
        raise _invalid(
            path,
            "is empty",
            f"The user must save the license.txt that FreeSurfer emailed ({REGISTRATION_URL}) "
            "in its place.",
        )
    if len(data) > _MAX_BYTES:
        raise _invalid(
            path,
            "is far too large to be a FreeSurfer licence (a licence is a few lines)",
            "Pass the path of the license.txt file FreeSurfer emailed.",
        )
    ok, n_lines = _looks_like_licence(data)
    if not ok:
        raise _invalid(
            path,
            "does not have the structure of a FreeSurfer licence (an email address, a "
            f"number and a key on the first three lines; this file has {n_lines} non-empty "
            "lines)",
            "The user must pass the license.txt that FreeSurfer emailed, unchanged "
            f"(a new one: {REGISTRATION_URL}).",
        )


def find_license(license_path: str | Path | None = None) -> Path:
    """Return the path of the user's FreeSurfer licence file, after checking it.

    Args:
        license_path: An explicit licence file inside the workspace. Default:
            ``<workspace>/.medmcp/licenses/freesurfer.txt``.

    Raises:
        LicenseMissingError: no licence file; the message says how to get one and
            where to put it.
        LicenseInvalidError: the file is outside the workspace, unreadable, or not a
            FreeSurfer licence; the message says which, without any of its content.
    """
    ws = workspace_root()
    if license_path is not None and str(license_path).strip():
        try:
            path = resolve_inside(license_path, "license_path", ws)
        except ValueError as exc:
            raise LicenseInvalidError(
                f"FreeSurfer licence problem: {exc}. MedMCP only uses a licence inside the "
                f"workspace, by default {DEFAULT_RELATIVE.as_posix()} in it; the user must "
                f"save it there themselves. {HANDS_OFF}"
            ) from None
    elif ws is not None:
        path = ws / DEFAULT_RELATIVE
    else:
        raise LicenseMissingError(instructions(None))
    check_license(path)
    return path
