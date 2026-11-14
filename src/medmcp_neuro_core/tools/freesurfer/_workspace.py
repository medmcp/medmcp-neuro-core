"""The workspace, and keeping every path a tool touches inside it.

A stack container gets exactly one bind mount from the core, the workspace
(``-v <ws>:<ws>``), and no environment. A host-native install inherits
``MEDMCP_WORKSPACE`` from the core instead. Either way every input the tools read
and every output they write must lie inside it.
"""

import os
import re
from pathlib import Path

# Mount points that are never the workspace: the container's own filesystem, the
# kernel's pseudo-filesystems, and what the runtime injects.
_SYSTEM_MOUNT_PREFIXES: tuple[str, ...] = (
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/run",
    "/var/run",
    "/app",
    "/opt",
)
_PSEUDO_FS: frozenset[str] = frozenset(
    {"proc", "sysfs", "devpts", "mqueue", "cgroup", "cgroup2", "overlay", "devtmpfs", "nsfs"}
)


def _unescape_mount(path: str) -> str:
    r"""Undo mountinfo's octal escaping (``\040`` for a space, etc.)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), path)


def detect_workspace_mount(mountinfo: Path = Path("/proc/self/mountinfo")) -> Path | None:
    """Find the workspace bind mount inside a stack container.

    Returns the single top-level non-system directory mount (nested mounts, such as
    the core's external drives or a licence file mounted read-only, live inside it
    and are ignored), or ``None`` when there is none or more than one candidate.
    """
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return None
    candidates: list[str] = []
    for line in lines:
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields = left.split()
        fstype = right.split()[0] if right.split() else ""
        if len(fields) < 5 or fstype in _PSEUDO_FS:
            continue
        mount_point = _unescape_mount(fields[4])
        if mount_point == "/" or any(
            mount_point == p or mount_point.startswith(p + "/") for p in _SYSTEM_MOUNT_PREFIXES
        ):
            continue
        if not Path(mount_point).is_dir():
            continue
        candidates.append(mount_point.rstrip("/"))
    top = sorted(
        {c for c in candidates if not any(c != o and c.startswith(o + "/") for o in candidates)}
    )
    return Path(top[0]) if len(top) == 1 else None


def workspace_root() -> Path | None:
    """The workspace: ``MEDMCP_WORKSPACE`` if set, else the container's bind mount."""
    env = os.environ.get("MEDMCP_WORKSPACE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return detect_workspace_mount()


def resolve_inside(path: str | Path, what: str, workspace: Path | None = None) -> Path:
    """Resolve *path* (absolute) and check that it lies inside the workspace.

    Args:
        path: The path as given in the tool call.
        what: What the path is, for the error message (``"t1_path"``).
        workspace: The workspace root; detected when omitted. When no workspace can
            be determined (a host-native development install) nothing is enforced.

    Raises:
        ValueError: the path is relative, or outside the workspace.
    """
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{what} must be an absolute path, got {str(path)!r}")
    resolved = raw.resolve()
    ws = workspace if workspace is not None else workspace_root()
    if ws is not None and not resolved.is_relative_to(ws):
        raise ValueError(f"{what} {resolved} is outside the workspace {ws}")
    return resolved
