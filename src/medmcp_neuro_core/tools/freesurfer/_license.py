"""Finding the user's FreeSurfer licence key, without ever reading it.

FreeSurfer's programs refuse to run without a licence file (a free registration
emails one). It is the user's personal key: this stack never copies it, logs it,
prints it or puts it in a result, and no image ships one. The tools only check that
the file is there and hand its *path* to FreeSurfer as ``FS_LICENSE``; FreeSurfer
itself reads and checks it.
"""

from pathlib import Path

from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root

REGISTRATION_URL = "https://surfer.nmr.mgh.harvard.edu/registration.html"
# Where the tools look by default, relative to the workspace.
DEFAULT_RELATIVE = Path(".medmcp") / "licenses" / "freesurfer.txt"
# A FreeSurfer licence is a few lines; anything far larger is not one.
_MAX_BYTES = 64 * 1024


class LicenseMissingError(FileNotFoundError):
    """No usable FreeSurfer licence file; the message tells the user what to do."""


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
        "your workspace and is never copied into a tool image, a log or a result."
    )


def find_license(license_path: str | Path | None = None) -> Path:
    """Return the path of the user's FreeSurfer licence file.

    Args:
        license_path: An explicit licence file inside the workspace. Default:
            ``<workspace>/.medmcp/licenses/freesurfer.txt``.

    Raises:
        LicenseMissingError: the file does not exist, is empty, or is not a licence
            file (too large); the message says how to get one and where to put it.
        ValueError: ``license_path`` is outside the workspace.
    """
    ws = workspace_root()
    if license_path is not None and str(license_path).strip():
        path = resolve_inside(license_path, "license_path", ws)
    elif ws is not None:
        path = ws / DEFAULT_RELATIVE
    else:
        raise LicenseMissingError(instructions(None))
    try:
        size = path.stat().st_size if path.is_file() else -1
    except OSError:
        size = -1
    if size <= 0 or size > _MAX_BYTES:
        raise LicenseMissingError(instructions(path))
    return path
