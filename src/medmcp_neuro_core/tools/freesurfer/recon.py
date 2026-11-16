"""recon-all as a background job: start it, follow it.

recon-all takes about an hour per scan, far beyond any tool-call timeout, so it
never runs inside a tool call. Where it runs depends on the core:

- A core that runs stack jobs says so (``MEDMCP_JOBS=1`` in this container's
  environment). ``run_recon_all`` then validates the call and answers with a
  ``medmcp_job`` request instead of starting anything; the core runs
  ``medmcp-neuro-core-job call run_recon_all <approved arguments>`` in a container
  of its own, outside its backend pool, so recon-all keeps running when the chat is
  idle or closed and when the core restarts.
- Otherwise (an older core) the job is a detached process inside the stack
  container, and the core removes that container when it is idle -- about two
  minutes after the last tool call. Such a run only survives while the stack keeps
  being called, and the result says so.

``recon_all_status`` reads only files in the workspace (the job record and
FreeSurfer's own logs), so it works from any chat and any container.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from medmcp_neuro_core.tools.freesurfer import _jobs
from medmcp_neuro_core.tools.freesurfer._inputs import (
    check_subject_id,
    check_t1,
    default_subject_id,
)
from medmcp_neuro_core.tools.freesurfer._license import (
    HANDS_OFF,
    LicenseError,
    LicenseInvalidError,
    LicenseMissingError,
    find_license,
    instructions,
)
from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root

JsonDict = dict[str, Any]
Pipeline = Literal["classic"]

# Set by a MedMCP core that starts stack jobs itself (see the module docstring).
JOBS_ENV = "MEDMCP_JOBS"
# Set in that job container; recorded in the job's state.
JOB_ID_ENV = "MEDMCP_JOB_ID"

# How long run_recon_all watches an in-container job for an immediate failure.
_STARTUP_WATCH_SEC = 15.0
_LAUNCHED: dict[str, subprocess.Popen[bytes]] = {}

RESEARCH_NOTE = (
    "FreeSurfer is research software, not approved for clinical use; its volumes and "
    "thickness values are not a diagnosis."
)
TIME_NOTE = (
    "recon-all takes about one hour per scan with 8 threads (longer with fewer, or on "
    "unusual scans)."
)
JOB_NOTE = (
    "MedMCP runs this as a background job in a container of its own. It keeps "
    "running when this chat is idle or closed, while other tools are used, and when "
    "MedMCP restarts; it ends when the run finishes, when it is stopped (Stop in the "
    "Tool stacks window), or when the server shuts down."
)
SESSION_WARNING = (
    "This MedMCP version cannot run background jobs, so the run happens inside the "
    "neuro tool container, which MedMCP removes when it is idle: about two minutes "
    "after the last neuro tool call, and when MedMCP restarts. It takes about an "
    "hour, so ask for its status regularly to keep it alive, or update MedMCP. An "
    "interrupted run must be started again in a new folder."
)

# What the model must know before a job is requested: MedMCP refuses a second job
# with an error that replaces the whole tool result, and the stack never sees it.
SLOT_REFUSAL_HINT = (
    "If MedMCP answers 'MedMCP did not start the background job: ... already "
    "running', its background-job slot is busy (one job at a time by default, possibly "
    "a job of another tool stack). That is NOT a licence problem and nothing is wrong "
    "with the licence file: wait until the running job is done (recon_all_status lists "
    "this stack's runs), then make the same call again. " + HANDS_OFF
)
WAITING = "waiting_for_job_slot"


def slot_busy_note(holder: JsonDict, what: str) -> str:
    """Why *what* did not start: the job slot is taken by *holder* (a described job)."""
    return (
        f"{what} was NOT started: MedMCP runs at most {_jobs.max_jobs()} background job"
        f"{'s' if _jobs.max_jobs() != 1 else ''} at a time, and {holder['pipeline']} for "
        f"{holder['subject_id']} (job {holder['job_id']}) is still running. This is NOT a "
        "licence problem: the licence file is fine and must not be touched. Nothing was "
        "queued; make the same call again when that job is done (recon_all_status "
        f"{holder['job_id']} shows its progress)."
    )


def core_runs_jobs() -> bool:
    """Whether the core offered to run the job in a container of its own."""
    return os.environ.get(JOBS_ENV, "").strip() == "1"


def available_threads() -> int:
    """CPUs this process may use."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def prepare_recon_all(
    t1_path: str,
    output_dir: str,
    subject_id: str | None = None,
    license_path: str | None = None,
    threads: int = 8,
    pipeline: Pipeline = "classic",
    force: bool = False,
) -> JsonDict:
    """Validate a run_recon_all call and return the job request.

    Shared by the tool and by the job entrypoint, so the job refuses exactly what
    the tool refuses.

    Raises:
        LicenseMissingError: no licence file (the message says how to get one).
        LicenseInvalidError: the licence file is unusable (the message says why).
        FileNotFoundError: the T1 image does not exist.
        FileExistsError: the subject folder already exists.
        ValueError: a path outside the workspace, a bad subject id, an unsupported
            pipeline, or an input the T1 checks refuse.
    """
    if pipeline != "classic":
        raise ValueError(
            f"pipeline {pipeline!r} is not available. Only 'classic' (recon-all -all "
            "-no-v8) is supported: FreeSurfer 8's default stream needs SynthSeg, "
            "SynthStrip and SynthMorph, which this image does not include."
        )
    ws = workspace_root()
    t1 = resolve_inside(t1_path, "t1_path", ws)
    subjects_dir = resolve_inside(output_dir, "output_dir", ws)
    if subjects_dir.exists() and not subjects_dir.is_dir():
        raise ValueError(f"output_dir {subjects_dir} exists and is not a folder")
    sid = check_subject_id(subject_id.strip() if subject_id else default_subject_id(t1))
    subject_dir = subjects_dir / sid
    if subject_dir.exists():
        raise FileExistsError(
            f"{subject_dir} already exists. recon-all does not overwrite a subject; "
            "choose another subject_id or output_dir (or remove the old folder yourself)."
        )
    if threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    licence = find_license(license_path)
    warnings = check_t1(t1, force=force)
    cpus = available_threads()
    if threads > cpus:
        warnings.append(f"threads={threads} is more than the {cpus} CPUs available; using {cpus}.")
        threads = cpus
    return {
        "t1_path": str(t1),
        "subjects_dir": str(subjects_dir),
        "subject_id": sid,
        "subject_dir": str(subject_dir),
        "license_path": str(licence),
        "threads": threads,
        "pipeline": pipeline,
        "force": force,
        "warnings": warnings,
        "job_id": _jobs.job_id_for(subject_dir),
        "registry": str(_jobs.registry(subjects_dir)),
    }


def recon_all_command(request: JsonDict) -> list[str]:
    """The recon-all command line for a validated request."""
    return [
        "recon-all",
        "-s",
        str(request["subject_id"]),
        "-i",
        str(request["t1_path"]),
        "-sd",
        str(request["subjects_dir"]),
        "-all",
        "-no-v8",
        "-parallel",
        "-openmp",
        str(request["threads"]),
    ]


def job_arguments(request: JsonDict) -> JsonDict:
    """The ``call run_recon_all`` arguments that reproduce a validated request."""
    return {
        "t1_path": request["t1_path"],
        "output_dir": request["subjects_dir"],
        "subject_id": request["subject_id"],
        "license_path": request["license_path"],
        "threads": request["threads"],
        "pipeline": request["pipeline"],
        "force": request["force"],
    }


class RunReconAllResult(TypedDict):
    """Result of run_recon_all (returned as a plain dict, see run_recon_all)."""

    status: str
    job_id: str | None
    subject_id: str | None
    subject_dir: str | None
    command: str | None
    threads: int | None
    warnings: list[str]
    log: str | None
    error: str | None
    note: str
    _render: str
    # Present when the job slot is busy (status waiting_for_job_slot): the job that
    # holds it, and the call to make again once it is done.
    running_job: NotRequired[dict[str, Any]]
    retry_call: NotRequired[dict[str, Any]]
    # Present when recon-all is handed to the core as a background job: the request
    # on the way out, replaced by the core with what it did about it.
    medmcp_job: NotRequired[dict[str, Any]]


def _licence_problem(exc: LicenseError) -> RunReconAllResult:
    message = str(exc)
    missing = isinstance(exc, LicenseMissingError)
    return {
        "status": exc.status,
        "job_id": None,
        "subject_id": None,
        "subject_dir": None,
        "command": None,
        "threads": None,
        "warnings": [],
        "log": None,
        "error": message,
        "note": message,
        "_render": (
            "DISPLAY RULES:\n"
            "recon-all was NOT started: "
            + (
                "FreeSurfer needs the user's licence file. "
                if missing
                else "the FreeSurfer licence file cannot be used. "
            )
            + "Tell the user exactly this, in plain words: "
            + message
            + "\n"
            "Do not ask the user to paste the licence into the chat. Never copy, move, "
            "chmod, open or print the licence file yourself, and never retry with a copy "
            "of it.\n"
            "NEXT ACTION: Wait for the user to fix the file, then call run_recon_all "
            "again with the same arguments."
        ),
    }


def _waiting_for_slot(request: JsonDict, holder: JsonDict) -> RunReconAllResult:
    note = slot_busy_note(holder, "recon-all")
    return {
        "status": WAITING,
        "job_id": None,
        "subject_id": str(request["subject_id"]),
        "subject_dir": str(request["subject_dir"]),
        "command": None,
        "threads": int(request["threads"]),
        "warnings": list(request["warnings"]),
        "log": None,
        "error": None,
        "note": note,
        "running_job": holder,
        "retry_call": {"tool": "run_recon_all", "arguments": job_arguments(request)},
        "_render": (
            "DISPLAY RULES:\n"
            "recon-all was NOT started because another background job is running; it is "
            "NOT a licence problem. Tell the user this note in plain words: " + note + "\n"
            "Do not touch the licence file (no copy, move, chmod or reading) and do not "
            "retry with other arguments.\n"
            "NEXT ACTION: Offer to check the running job (recon_all_status with "
            "running_job.job_id) and to make retry_call again once it is done. Do not poll "
            "in a loop unless the user asks you to wait."
        ),
    }


def launch_job(request: JsonDict, tool: str, arguments: JsonDict) -> subprocess.Popen[bytes]:
    """Start the job for a validated request as a detached process in this container."""
    reg = Path(str(request["registry"]))
    jdir = _jobs.job_dir(reg, str(request["job_id"]))
    jdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "medmcp_neuro_core.job", "call", tool, json.dumps(arguments)]
    with (jdir / _jobs.LOG_FILE).open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own session: never gets our stdio or signals
            close_fds=True,
        )
    _LAUNCHED[str(request["job_id"])] = proc
    return proc


def watch_startup(request: JsonDict, proc: subprocess.Popen[bytes]) -> JsonDict:
    """Wait briefly for the job to reach 'running', or to fail fast."""
    jdir = _jobs.job_dir(Path(str(request["registry"])), str(request["job_id"]))
    deadline = time.monotonic() + _STARTUP_WATCH_SEC
    state: JsonDict | None = None
    while time.monotonic() < deadline:
        state = _jobs.read_state(jdir)
        if state is not None and state.get("pid") == proc.pid and state.get("status") != "starting":
            break
        if proc.poll() is not None:
            state = _jobs.read_state(jdir)
            break
        time.sleep(0.5)
    if state is None or state.get("pid") != proc.pid:
        if proc.poll() is not None:
            return {
                "status": "failed",
                "error": f"job exited with code {proc.returncode}: "
                + _jobs.read_tail(jdir / _jobs.LOG_FILE, 800).strip(),
            }
        return {"status": "starting", "error": None}
    return {"status": _jobs.effective_status(state), "error": state.get("error")}


def run_recon_all(
    t1_path: str,
    output_dir: str,
    subject_id: str | None = None,
    license_path: str | None = None,
    threads: int = 8,
    pipeline: Pipeline = "classic",
    force: bool = False,
) -> JsonDict:
    """Start FreeSurfer recon-all (classic stream) on one T1 scan, in the background.

    Runs ``recon-all -all -no-v8 -parallel -openmp <threads>``: cortical surfaces,
    cortical thickness and parcellations, and the subcortical segmentation. Takes
    about an hour; returns at once with a ``job_id`` for recon_all_status. Results
    go to ``<output_dir>/<subject_id>/`` (a FreeSurfer subject folder); read them
    with freesurfer_stats when the run is done.

    Needs the user's FreeSurfer licence file (free registration), by default at
    ``<workspace>/.medmcp/licenses/freesurfer.txt``. The tool checks it (exists,
    readable, shaped like a licence); when it cannot be used nothing is started and
    ``status`` is ``licence_required`` or ``licence_invalid`` with what to do. Pass
    the licence path as it is: never copy, move, chmod, open or print the file.

    MedMCP runs one background job at a time by default. When this stack's own job
    already fills that slot, nothing is started and ``status`` is
    ``waiting_for_job_slot`` with the running job and the call to make later. If
    MedMCP itself answers "MedMCP did not start the background job: ... already
    running", the slot is busy too (maybe with another stack's job): that is NOT a
    licence problem; wait for the job to finish and make the same call again.

    Args:
        t1_path: The T1-weighted scan, a 3D NIfTI (.nii/.nii.gz) or .mgz file in
            the workspace (absolute path). About 1 mm isotropic is best.
        output_dir: The FreeSurfer subjects folder in the workspace (absolute path);
            created if missing.
        subject_id: Name of the subject folder to create. Default: from the T1 file
            name. Must not exist yet.
        license_path: The FreeSurfer licence file, if not at the default location.
            Must be inside the workspace; pass the path as it is.
        threads: CPU threads for recon-all (default 8).
        pipeline: ``"classic"`` (the only one available: recon-all -all -no-v8).
        force: Run even on thick-slice or strongly anisotropic scans, which the input
            check otherwise refuses.

    Returns:
        ``status`` (``requested`` when MedMCP runs the job, else ``starting`` /
        ``running`` / ``failed``; or ``licence_required`` / ``licence_invalid`` /
        ``waiting_for_job_slot``, when nothing was started), ``job_id``,
        ``subject_dir``, the ``command``, input ``warnings`` and a ``note`` on where
        and how long the job runs.
    """
    try:
        request = prepare_recon_all(
            t1_path, output_dir, subject_id, license_path, threads, pipeline, force
        )
    except LicenseError as exc:
        return dict(_licence_problem(exc))

    reg = Path(str(request["registry"]))
    jdir = _jobs.job_dir(reg, str(request["job_id"]))
    current = _jobs.read_state(jdir)
    if current is not None and _jobs.effective_status(current) in _jobs.ACTIVE:
        raise RuntimeError(
            f"recon-all for {request['subject_id']} is already running (job "
            f"{request['job_id']}); use recon_all_status to follow it."
        )
    as_job = core_runs_jobs()
    if as_job:
        holder = _jobs.slot_holder(reg, exclude=str(request["job_id"]))
        if holder is not None:
            return dict(_waiting_for_slot(request, _jobs.describe_job(holder)))
    Path(str(request["subjects_dir"])).mkdir(parents=True, exist_ok=True)

    if as_job:
        seen: JsonDict = {"status": "requested", "error": None}
    else:
        proc = launch_job(request, "run_recon_all", job_arguments(request))
        seen = watch_startup(request, proc)

    warnings: list[str] = list(request["warnings"])
    command = " ".join(recon_all_command(request))
    common = (
        "Report: job_id, subject_id, subject_dir and threads. If 'warnings' is non-empty, "
        "show each warning FIRST: the scan may not suit recon-all.\n"
    )
    if as_job:
        render = (
            "DISPLAY RULES:\n" + common + "Look at medmcp_job: only if it says started: true "
            "was recon-all started. Then say so, say it takes about an hour, and repeat "
            "this note verbatim: " + JOB_NOTE + "\n"
            "If medmcp_job does not say started: true, recon-all was NOT started; say that "
            "plainly and do not claim it is running.\n" + SLOT_REFUSAL_HINT + "\n"
            "Mention once: " + RESEARCH_NOTE + "\n"
            "NEXT ACTION: Tell the user they can ask for progress at any time "
            "(recon_all_status with the job_id or subject folder), also from a new chat, "
            "and that freesurfer_stats reads the results once it is done. Do not poll in a "
            "loop unless the user asks you to wait."
        )
        note = JOB_NOTE
    else:
        render = (
            "DISPLAY RULES:\n" + common + "If status is failed, show error and stop. "
            "Otherwise say recon-all was started in the background, takes about an hour, "
            "and repeat this note verbatim: " + SESSION_WARNING + "\n"
            "Mention once: " + RESEARCH_NOTE + "\n"
            "NEXT ACTION: Tell the user they can ask for progress at any time "
            "(recon_all_status). Do not poll in a loop unless the user asks you to wait."
        )
        note = SESSION_WARNING
    result: RunReconAllResult = {
        "status": str(seen["status"]),
        "job_id": str(request["job_id"]),
        "subject_id": str(request["subject_id"]),
        "subject_dir": str(request["subject_dir"]),
        "command": command,
        "threads": int(request["threads"]),
        "warnings": warnings,
        "log": str(jdir / _jobs.LOG_FILE),
        "error": str(seen["error"]) if seen.get("error") else None,
        "note": note + " " + TIME_NOTE,
        "_render": render,
    }
    if as_job:
        result["medmcp_job"] = {"name": f"recon-all {request['subject_id']}"}
    # A plain dict, not the TypedDict: FastMCP's structured output drops keys that
    # start with "_" from a TypedDict (losing _render) and fills an absent
    # NotRequired key with null (an invalid medmcp_job).
    return dict(result)


# ── status ──────────────────────────────────────────────────────────────────────


def _summary(state: JsonDict | None, subject_dir: Path, job_id: str | None) -> JsonDict:
    """One run's status, combining the job record with FreeSurfer's own files."""
    from medmcp_neuro_core.tools.freesurfer.surfaces import KIND, surf_errors, surf_progress

    surfaces = bool(state and state.get("kind") == KIND)
    fs = surf_progress(subject_dir) if surfaces else _jobs.fs_progress(subject_dir)
    if surfaces and fs["fs_status"] != "done" and _jobs.effective_status(state) == "failed":
        fs["error_lines"] = surf_errors(subject_dir)
    job_status = _jobs.effective_status(state)
    if fs["fs_status"] == "done":
        status = "done"
    elif fs["fs_status"] == "error":
        status = "failed"
    elif job_status in _jobs.ACTIVE or job_status in {"failed", "stopped", "interrupted"}:
        status = job_status
    elif state is None and fs["fs_status"] == "not started":
        status = "not_started"
    else:
        status = "unknown"
    now = time.time()
    started = state.get("started_at") if state else None
    finished = state.get("finished_at") if state else None
    elapsed: float | None = None
    if isinstance(started, (int, float)):
        end = float(finished) if isinstance(finished, (int, float)) else now
        elapsed = max(0.0, end - float(started))
    fraction: float | None = None if fs["fraction"] is None else float(fs["fraction"])
    remaining: str | None = None
    if (
        status == "running"
        and elapsed is not None
        and fraction is not None
        and 0.05 < fraction < 1.0
    ):
        remaining = _jobs.fmt_minutes(elapsed / fraction * (1.0 - fraction))
    error = state.get("error") if state else None
    if status == "failed" and not error and fs["error_lines"]:
        error = " | ".join(fs["error_lines"])
    beat = state.get("heartbeat_at") if state else None
    return {
        "job_id": job_id,
        "pipeline": "FastSurfer surfaces (recon-surf)" if surfaces else "FreeSurfer recon-all",
        "subject_id": subject_dir.name,
        "subject_dir": str(subject_dir),
        "status": status,
        "current_step": fs["current_step"],
        "steps_done": fs["steps_done"],
        "percent_done": None if fraction is None else round(100 * fraction),
        "elapsed": _jobs.fmt_minutes(elapsed) if elapsed is not None else None,
        "estimated_remaining": remaining,
        "runtime_hours": fs["runtime_hours"],
        "runs_in": (
            ("MedMCP background job" if state.get("core_job_id") else "stack container")
            if state
            else None
        ),
        "heartbeat_age_sec": round(now - float(beat)) if isinstance(beat, (int, float)) else None,
        "error": error,
        "recon_all_log": str(
            subject_dir / "scripts" / ("recon-surf.log" if surfaces else "recon-all.log")
        ),
        "job_log": str(Path(str(state["job_dir"])) / _jobs.LOG_FILE)
        if state and state.get("job_dir")
        else None,
    }


_STATUS_RENDER = (
    "DISPLAY RULES:\n"
    "Report: pipeline, status, subject_id, current_step, percent_done, elapsed and (if "
    "present) estimated_remaining, as a compact key-value list. percent_done and "
    "estimated_remaining are rough (based on a typical run); say 'about'. A FastSurfer "
    "surface run has no percent_done; give current_step and elapsed (about an hour in "
    "total).\n"
    "If status is failed, show error and the recon_all_log path; the pipeline's own "
    "messages are in that log.\n"
    "If status is interrupted, say the run stopped before finishing (its container "
    "ended) and must be started again with run_recon_all under a new subject_id. If "
    "status is stopped, say it was stopped before finishing (same remedy).\n"
    "NEXT ACTION: If status is done, offer freesurfer_stats(subject_dir) for volumes and "
    "cortical thickness. If running, tell the user they can ask again later; do not "
    "poll in a loop unless asked."
)


def recon_all_status(job_or_subject: str | None = None) -> JsonDict:
    """Report on recon-all and FastSurfer-surface runs: one run's progress, or all runs.

    Covers FreeSurfer recon-all (run_recon_all) and FastSurfer's surface pipeline
    (segment_brain with surfaces=True). Reads only files in the workspace (the job
    record and FreeSurfer's own logs in the subject folder), so it works from any
    chat, also after the chat that started the run is closed.

    Args:
        job_or_subject: A ``job_id`` from run_recon_all or segment_brain, a subject
            folder (absolute path, ``<output_dir>/<subject_id>``) or a subject_id.
            Omit to list every run recorded in this workspace.

    Returns:
        For one run: ``status`` (running / done / failed / interrupted /
        not_started), the ``current_step``, rough ``percent_done`` and
        ``estimated_remaining``, ``elapsed``, ``runtime_hours`` once done, the log
        paths and any ``error``. Without an argument: ``runs``, one summary each.
    """
    ws = workspace_root()
    key = (job_or_subject or "").strip()
    # Without a workspace the records sit beside the subjects (see _jobs.registry).
    base = Path(key).parent if key.startswith("/") else Path.cwd()
    reg = _jobs.registry(ws if ws is not None else base)
    states = _jobs.all_states(reg)
    for s in states:
        s.setdefault("job_dir", str(reg / str(s.get("job_id", ""))))

    if not key:
        runs = [
            _summary(s, Path(str(s.get("subject_dir", ""))), str(s.get("job_id")))
            for s in states
            if s.get("subject_dir")
        ]
        return {
            "runs": runs,
            "registry": str(reg),
            "_render": (
                "DISPLAY RULES:\nList each run as one line: subject_id, status, "
                "percent_done (about), elapsed. If there are none, say no recon-all run "
                "is recorded in this workspace.\nNEXT ACTION: Offer details for one run "
                "(recon_all_status with its job_id), or freesurfer_stats for a done one."
            ),
        }

    match: JsonDict | None = None
    subject_dir: Path | None = None
    if key.startswith("/"):
        subject_dir = resolve_inside(key, "job_or_subject", ws)
        match = next((s for s in states if s.get("subject_dir") == str(subject_dir)), None)
        if match is None and not subject_dir.is_dir():
            raise FileNotFoundError(f"No subject folder {subject_dir} and no run recorded for it")
    else:
        match = next((s for s in states if s.get("job_id") == key), None)
        if match is None:
            by_subject = [s for s in states if s.get("subject_id") == key]
            if len(by_subject) > 1:
                return {
                    "runs": [
                        _summary(s, Path(str(s["subject_dir"])), str(s.get("job_id")))
                        for s in by_subject
                    ],
                    "_render": (
                        "DISPLAY RULES:\nSeveral runs have this subject_id. List each "
                        "with its subject_dir and status.\nNEXT ACTION: Ask which one "
                        "the user means (then call with its job_id)."
                    ),
                }
            match = by_subject[0] if by_subject else None
        if match is None:
            raise FileNotFoundError(
                f"No recon-all run {key!r} is recorded in {reg}. Pass the job_id from "
                "run_recon_all or the subject folder's absolute path; call without an "
                "argument to list the runs."
            )
        subject_dir = Path(str(match["subject_dir"]))
    summary = _summary(match, subject_dir, str(match.get("job_id")) if match else None)
    summary["_render"] = _STATUS_RENDER
    return summary


__all__ = [
    "LicenseError",
    "LicenseInvalidError",
    "LicenseMissingError",
    "instructions",
    "prepare_recon_all",
    "recon_all_status",
    "run_recon_all",
]
