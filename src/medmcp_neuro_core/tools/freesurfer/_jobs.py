"""recon-all job records and progress: everything lives in files in the workspace.

A recon-all run is a standalone process (``medmcp-freesurfer-job call
run_recon_all ...``, see :mod:`medmcp_neuro_core.job`) that needs nothing from the
MCP server. It records its life in ``<registry>/<job_id>/``:

- ``state.json`` -- written by the job: status, pid, hostname, timestamps, exit
  code, error, a heartbeat refreshed every few seconds, and the request it ran.
- ``job.log`` -- recon-all's console output.

``<registry>`` is ``<workspace>/.medmcp/freesurfer/jobs`` (or, when no workspace
can be determined, ``<output_dir>/.medmcp-freesurfer/jobs``). FreeSurfer's own
files in the subject folder -- ``scripts/recon-all-status.log``,
``scripts/recon-all.done`` / ``recon-all.error`` -- are the other source of truth;
the status tool reads both, never in-process state, so it works from any container
that mounts the workspace.
"""

import contextlib
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Literal, cast

from medmcp_neuro_core.tools.freesurfer._workspace import workspace_root

JsonDict = dict[str, Any]

STATE_FILE = "state.json"
LOG_FILE = "job.log"

# The job refreshes its heartbeat this often. Seen from another container, a
# heartbeat older than STALE_AFTER_SEC means the job is gone.
HEARTBEAT_SEC = 10.0
STALE_AFTER_SEC = 45.0

ACTIVE = frozenset({"starting", "running"})
Status = Literal["starting", "running", "done", "failed", "stopped", "interrupted", "unknown"]

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Where each recon-all step starts, as a fraction of the whole run, measured on a
# full classic run (recon-all -all -no-v8 -parallel -openmp 8, IXI002-Guys T1,
# 0.92 h). Used only to show rough progress; a step name not listed here keeps the
# last known fraction.
_STEP_START: dict[str, float] = {
    "MotionCor": 0.0,
    "Talairach": 0.002,
    "Talairach Failure Detection": 0.016,
    "Nu Intensity Correction": 0.016,
    "Intensity Normalization": 0.028,
    "Skull Stripping": 0.036,
    "EM Registration": 0.052,
    "CA Normalize": 0.062,
    "CA Reg": 0.067,
    "SubCort Seg": 0.304,
    "CC Seg": 0.574,
    "Merge ASeg": 0.576,
    "Intensity Normalization2": 0.576,
    "Mask BFS": 0.588,
    "WM Segmentation": 0.588,
    "Fill": 0.603,
    "Tessellate": 0.609,
    "Smooth1": 0.61,
    "Inflation1": 0.61,
    "QSphere": 0.611,
    "Fix Topology": 0.621,
    "Smooth2": 0.707,
    "Inflation2": 0.707,
    "Curv .H and .K": 0.709,
    "Sphere": 0.716,
    "Surf Reg": 0.758,
    "Jacobian white": 0.828,
    "AvgCurv": 0.828,
    "Cortical Parc": 0.828,
    "white curv": 0.936,
    "white area": 0.936,
    "pial curv": 0.936,
    "pial area": 0.936,
    "thickness": 0.936,
    "area and vertex vol": 0.94,
    "Curvature Stats": 0.946,
    "Cortical ribbon mask": 0.947,
    "Cortical Parc 2": 0.957,
    "Cortical Parc 3": 0.959,
    "WM/GM Contrast": 0.96,
    "Relabel Hypointensities": 0.961,
    "APas-to-ASeg": 0.962,
    "AParc-to-ASeg aparc": 0.963,
    "AParc-to-ASeg aparc.a2009s": 0.967,
    "AParc-to-ASeg aparc.DKTatlas": 0.972,
    "WMParc": 0.976,
    "Parcellation Stats": 0.983,
    "Parcellation Stats 2": 0.984,
    "Parcellation Stats 3": 0.984,
    "ASeg Stats": 0.985,
    "BA_exvivo Labels": 0.986,
}
# "#@# CA Reg Sat Nov 14 08:29:14 UTC 2026": the step name, then a date(1) stamp.
_STEP_LINE = re.compile(r"^#@#\s+(.*?)\s+\w{3} \w{3} +\d+ [\d:]+ \S+ \d{4}\s*$")


def registry(output_dir: Path) -> Path:
    """Folder holding every job record (see the module docstring)."""
    ws = workspace_root()
    if ws is not None:
        return ws / ".medmcp" / "freesurfer" / "jobs"
    return output_dir / ".medmcp-freesurfer" / "jobs"


def job_id_for(subject_dir: Path) -> str:
    """Deterministic job id for one subject folder: ``<subject>-<8 hex>``."""
    digest = hashlib.sha1(str(subject_dir).encode()).hexdigest()[:8]
    return f"{_SAFE.sub('_', subject_dir.name)}-{digest}"


def job_dir(reg: Path, job_id: str) -> Path:
    """Folder of one job's record.

    Raises:
        ValueError: ``job_id`` is not a plain folder name.
    """
    if not job_id or _SAFE.search(job_id) or job_id in {".", ".."}:
        raise ValueError(f"invalid job id: {job_id!r}")
    return reg / job_id


def read_state(jdir: Path) -> JsonDict | None:
    """The job's ``state.json``, or None if missing or unreadable."""
    try:
        data: object = json.loads((jdir / STATE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return cast("JsonDict", data) if isinstance(data, dict) else None


def write_state(jdir: Path, state: JsonDict) -> None:
    """Atomically replace the job's ``state.json``."""
    jdir.mkdir(parents=True, exist_ok=True)
    tmp = jdir / f".{STATE_FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(jdir / STATE_FILE)


def _pid_alive_here(state: JsonDict) -> bool:
    """Whether the job's pid is alive in this container (same hostname); zombies count dead."""
    pid = state.get("pid")
    if not isinstance(pid, int) or state.get("hostname") != socket.gethostname():
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    after = stat.rsplit(")", 1)[-1].split()
    return bool(after) and after[0] not in {"Z", "X"}


def effective_status(state: JsonDict | None, now: float | None = None) -> Status:
    """Status as reported to the user: a 'running' job whose process is gone is 'interrupted'."""
    if state is None:
        return "unknown"
    status = str(state.get("status", "unknown"))
    if status not in ACTIVE:
        return cast("Status", status) if status in {"done", "failed", "stopped"} else "unknown"
    if state.get("hostname") == socket.gethostname():
        return cast("Status", status) if _pid_alive_here(state) else "interrupted"
    now = time.time() if now is None else now
    beat = state.get("heartbeat_at")
    if isinstance(beat, (int, float)) and now - float(beat) < STALE_AFTER_SEC:
        return cast("Status", status)
    return "interrupted"


def all_states(reg: Path) -> list[JsonDict]:
    """Every job record under *reg*, newest first."""
    found: list[JsonDict] = []
    if not reg.is_dir():
        return found
    for jdir in reg.iterdir():
        state = read_state(jdir)
        if state is not None:
            found.append(state)
    found.sort(key=lambda s: float(s.get("started_at") or 0), reverse=True)
    return found


# ── FreeSurfer's own files ──────────────────────────────────────────────────────


def _step_key(name: str) -> str:
    """Step name without its hemisphere: ``"Fix Topology lh"`` -> ``"Fix Topology"``."""
    return re.sub(r"\s+(lh|rh)$", "", name.strip())


def fs_progress(subject_dir: Path) -> JsonDict:
    """What recon-all's own files say about a subject.

    Returns ``fs_status`` (``"done"`` when ``scripts/recon-all.done`` exists,
    ``"error"`` when ``scripts/recon-all.error`` exists, ``"running"`` while
    ``scripts/IsRunning.*`` exists, ``"not started"`` without a status log, else
    ``"stopped"``), ``current_step``, ``steps_done`` (``#@#`` lines in
    ``recon-all-status.log``), ``fraction`` (rough, see ``_STEP_START``),
    ``runtime_hours`` (from ``recon-all.done``) and ``error_lines`` (the ERROR
    lines at the end of ``recon-all.log`` when it failed).
    """
    scripts = subject_dir / "scripts"
    status_log = scripts / "recon-all-status.log"
    out: JsonDict = {
        "fs_status": "not started",
        "current_step": None,
        "steps_done": 0,
        "fraction": 0.0,
        "runtime_hours": None,
        "error_lines": [],
    }
    steps: list[str] = []
    with contextlib.suppress(OSError):
        for line in status_log.read_text(errors="replace").splitlines():
            m = _STEP_LINE.match(line)
            if m:
                steps.append(m.group(1))
    out["steps_done"] = len(steps)
    if steps:
        out["current_step"] = steps[-1]
        fraction = 0.0
        for s in steps:
            fraction = max(fraction, _STEP_START.get(_step_key(s), fraction))
        out["fraction"] = round(fraction, 3)
    done = scripts / "recon-all.done"
    if done.is_file() and not (scripts / "recon-all.error").is_file():
        out["fs_status"] = "done"
        out["fraction"] = 1.0
        with contextlib.suppress(OSError):
            for line in done.read_text(errors="replace").splitlines():
                if line.startswith("RUNTIME_HOURS"):
                    with contextlib.suppress(ValueError, IndexError):
                        out["runtime_hours"] = float(line.split()[1])
    elif (scripts / "recon-all.error").is_file():
        out["fs_status"] = "error"
        out["error_lines"] = recon_all_errors(subject_dir)
    elif list(scripts.glob("IsRunning.*")):
        out["fs_status"] = "running"
    elif steps or status_log.is_file():
        out["fs_status"] = "stopped"
    return out


def recon_all_errors(subject_dir: Path, limit: int = 8) -> list[str]:
    """The last ERROR lines of ``scripts/recon-all.log`` (FreeSurfer's own messages)."""
    log = subject_dir / "scripts" / "recon-all.log"
    try:
        lines = log.read_text(errors="replace").splitlines()[-400:]
    except OSError:
        return []
    hits = [ln.strip() for ln in lines if "ERROR" in ln or "error" in ln.split(":", 1)[0]]
    return hits[-limit:]


def read_tail(path: Path, max_chars: int = 1500) -> str:
    """The end of a text file ('' if unreadable)."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_chars))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def fmt_minutes(seconds: float) -> str:
    """``3 h 05 min`` / ``12 min``."""
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"
