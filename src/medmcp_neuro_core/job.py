r"""``medmcp-neuro-core-job``: the standalone entrypoint for hour-long FreeSurfer runs.

Two tool calls run here, never in the MCP server:

- ``run_recon_all``: FreeSurfer's recon-all (classic stream);
- ``segment_brain`` with ``surfaces=True``: FastSurfer's surface pipeline
  (recon-surf), after its segmentation if that is not there yet.

The job needs nothing from the server: it takes the tool call's arguments,
validates them exactly as the tool does, runs the programs and records its state
and log in the workspace (see :mod:`medmcp_neuro_core.tools.freesurfer._jobs`). It
is launched

- by a MedMCP core that runs stack jobs, in a container of its own, from the
  approved tool call (the image's ``org.medmcp.stack`` label names
  ``medmcp-neuro-core-job call`` as its job runner);
- without such a core, by the tool as a detached process inside the stack
  container;
- by hand: ``docker run ... <image> medmcp-neuro-core-job call run_recon_all '{...}'``.

Usage::

    medmcp-neuro-core-job call run_recon_all '<JSON arguments>'
    medmcp-neuro-core-job call segment_brain '<JSON arguments with "surfaces": true>'

Exit code 0 = done, 1 = failed or refused, 2 = bad arguments or already running,
143 = stopped.
"""

import argparse
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from medmcp_neuro_core.tools.freesurfer import _jobs

JsonDict = dict[str, Any]

EXIT_STOPPED = 143
# The tools whose calls a core may run as a job (as the image label declares).
JOB_TOOLS = frozenset({"run_recon_all", "segment_brain"})
LOG_PREFIX = "[medmcp-neuro-core]"
# Set by a MedMCP core that runs stack jobs: a JSON file it shows in its jobs list,
# {"fraction": 0..1, "text": "..."}.
CORE_PROGRESS_ENV = "MEDMCP_JOB_PROGRESS"


def core_progress(state: JsonDict) -> JsonDict | None:
    """Current step, rough percentage and time left, as the core's jobs list shows it."""
    from medmcp_neuro_core.tools.freesurfer.surfaces import KIND, surf_progress

    subject_dir = state.get("request", {}).get("subject_dir")
    if not subject_dir:
        return None
    if state.get("kind") == KIND:
        fs = surf_progress(Path(str(subject_dir)))
        if fs["fs_status"] == "done":
            return {"fraction": 1.0, "text": "done"}
        step = fs["current_step"] or state.get("step") or "starting"
        return {"fraction": 0.0, "text": f"FastSurfer surfaces: {step}"}
    fs = _jobs.fs_progress(Path(str(subject_dir)))
    fraction = float(fs["fraction"])
    parts = [str(fs["current_step"] or "starting")]
    started = state.get("started_at")
    if isinstance(started, (int, float)) and 0.05 < fraction < 1.0:
        elapsed = time.time() - float(started)
        parts.append(f"~{_jobs.fmt_minutes(elapsed / fraction * (1 - fraction))} left")
    return {"fraction": round(fraction, 4), "text": " · ".join(parts)}


def _write_core_progress(state: JsonDict) -> None:
    """Best effort: a job without a core, or an unwritable progress file, still runs."""
    target = os.environ.get(CORE_PROGRESS_ENV, "").strip()
    if not target:
        return
    with contextlib.suppress(Exception):
        progress = core_progress(state)
        if progress is None:
            return
        tmp = Path(target).with_suffix(".tmp")
        tmp.write_text(json.dumps(progress), encoding="utf-8")
        os.replace(tmp, target)


class _Job:
    """Owns one job's state file, heartbeat and stop handling."""

    def __init__(self, jdir: Path, state: JsonDict) -> None:
        self.jdir = jdir
        self.state = state
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.proc: subprocess.Popen[bytes] | None = None

    def update(self, changes: JsonDict) -> None:
        with self.lock:
            self.state.update(changes)
            self.state["heartbeat_at"] = time.time()
            _jobs.write_state(self.jdir, self.state)

    def finish(self, status: str, exit_code: int, **extra: Any) -> None:
        self.finished.set()
        self.update({"status": status, "exit_code": exit_code, "finished_at": time.time(), **extra})

    def heartbeat_loop(self) -> None:
        while not self.finished.wait(_jobs.HEARTBEAT_SEC):
            with contextlib.suppress(OSError):
                self.update({})
            _write_core_progress(self.state)

    def stop(self) -> None:
        """Record 'stopped' and end the running program (its process group) and this process."""
        if not self.finished.is_set():
            with contextlib.suppress(OSError):
                self.finish("stopped", EXIT_STOPPED, error="stopped before the run finished")
        if self.proc is not None:
            with contextlib.suppress(OSError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=10)
            with contextlib.suppress(OSError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_STOPPED)


def _redirect_output(log_path: Path) -> None:
    """Send this process's stdout/stderr (fd 1 and 2) to the job log, appending."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


class _Step:
    """One program the job runs: its name, command line and environment."""

    def __init__(self, name: str, cmd: list[str], env: dict[str, str]) -> None:
        self.name = name
        self.cmd = cmd
        self.env = env


def _steps(request: JsonDict) -> tuple[str, list[_Step]]:
    """The job's kind and the programs it runs, for a validated request."""
    from medmcp_neuro_core.tools.freesurfer._env import freesurfer_env
    from medmcp_neuro_core.tools.freesurfer.surfaces import (
        KIND,
        seg_command,
        segmentation_present,
        surf_command,
    )

    fs_env = freesurfer_env(
        os.environ,
        license_path=str(request["license_path"]),
        subjects_dir=str(request["subjects_dir"]),
        threads=int(request["threads"]),
    )
    if request.get("kind") != KIND:
        from medmcp_neuro_core.tools.freesurfer.recon import recon_all_command

        return "recon-all", [_Step("recon-all", recon_all_command(request), fs_env)]

    from medmcp_neuro_core.tools._neuro import resolve_device
    from medmcp_neuro_core.tools.segmentation import fastsurfer_python, find_fastsurfer

    binary = find_fastsurfer()
    fs_python = fastsurfer_python()
    steps: list[_Step] = []
    if not segmentation_present(Path(str(request["subject_dir"]))):
        device = resolve_device(request["device"])
        cmd = seg_command(request, binary, fs_python, device)
        steps.append(_Step("FastSurfer segmentation", cmd, dict(os.environ)))
    steps.append(_Step("recon-surf", surf_command(request, binary, fs_python), fs_env))
    return KIND, steps


def _outcome(kind: str, request: JsonDict, log_path: Path) -> tuple[bool, str, float | None]:
    """(succeeded, error detail, runtime hours) from the programs' own files."""
    from medmcp_neuro_core.tools.freesurfer.surfaces import KIND, surf_errors, surf_progress

    subject_dir = Path(str(request["subject_dir"]))
    if kind == KIND:
        fs = surf_progress(subject_dir)
        detail = " | ".join(surf_errors(subject_dir))
    else:
        fs = _jobs.fs_progress(subject_dir)
        detail = " | ".join(fs["error_lines"])
    runtime = fs["runtime_hours"] if isinstance(fs["runtime_hours"], float) else None
    return fs["fs_status"] == "done", detail or _jobs.read_tail(log_path, 600).strip(), runtime


def run(request: JsonDict) -> int:
    """Run the job for a validated request. Returns the process exit code."""
    from medmcp_neuro_core.tools.freesurfer.recon import JOB_ID_ENV

    reg = Path(str(request["registry"]))
    job_id = str(request["job_id"])
    jdir = _jobs.job_dir(reg, job_id)
    previous = _jobs.read_state(jdir)
    if (
        previous is not None
        and previous.get("pid") != os.getpid()
        and _jobs.effective_status(previous) in _jobs.ACTIVE
    ):
        print(f"job {job_id} is already running (pid {previous.get('pid')})", file=sys.stderr)
        return 2

    log_path = jdir / _jobs.LOG_FILE
    print(f"{LOG_PREFIX} job {job_id}: logging to {log_path}", flush=True)
    _redirect_output(log_path)

    Path(str(request["subjects_dir"])).mkdir(parents=True, exist_ok=True)
    state: JsonDict = {
        "job_id": job_id,
        "kind": "recon-all",
        "status": "starting",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "subject_id": request["subject_id"],
        "subject_dir": request["subject_dir"],
        "job_dir": str(jdir),
        "core_job_id": os.environ.get(JOB_ID_ENV) or None,
        "step": None,
        "command": None,
        # Paths and settings only; nothing of the licence file's content.
        "request": {k: v for k, v in request.items() if k != "registry"},
    }
    job = _Job(jdir, state)
    job.update({})

    def _on_sigterm(signum: int, frame: object) -> None:
        job.stop()

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)
    threading.Thread(target=job.heartbeat_loop, name="heartbeat", daemon=True).start()

    try:
        kind, steps = _steps(request)
    except (OSError, RuntimeError, ValueError) as exc:
        job.finish("failed", 1, error=str(exc))
        return 1
    job.update({"kind": kind})
    for step in steps:
        print(
            f"{LOG_PREFIX} {time.strftime('%Y-%m-%d %H:%M:%S')} job {job_id}: {step.name} "
            f"(pid {os.getpid()}, host {socket.gethostname()}): {' '.join(step.cmd)}",
            flush=True,
        )
        if shutil.which(step.cmd[0], path=step.env.get("PATH")) is None:
            job.finish(
                "failed",
                1,
                error=f"{step.cmd[0]} is not installed here. The medmcp-neuro-core image has it.",
            )
            return 1
        try:
            job.proc = subprocess.Popen(
                step.cmd,
                cwd=str(request["subjects_dir"]),
                env=step.env,
                stdin=subprocess.DEVNULL,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,  # own process group, so stop takes all of it
            )
        except OSError as exc:
            job.finish("failed", 1, error=f"{step.name} could not be started: {exc}")
            return 1
        job.update({"status": "running", "step": step.name, "command": step.cmd})
        code = job.proc.wait()
        if step is not steps[-1]:
            # The segmentation can exit non-zero on an optional final step (the
            # corpus-callosum shape measures) after writing everything recon-surf needs.
            from medmcp_neuro_core.tools.freesurfer.surfaces import segmentation_present

            if not segmentation_present(Path(str(request["subject_dir"]))):
                detail = _jobs.read_tail(log_path, 600).strip()
                job.finish("failed", 1, error=f"{step.name} exited with code {code}: {detail}")
                return 1
            continue
        _write_core_progress(job.state)
        ok, detail, runtime = _outcome(kind, request, log_path)
        if code == 0 and ok:
            job.finish("done", 0, runtime_hours=runtime)
            print(f"{LOG_PREFIX} job {job_id} done", flush=True)
            return 0
        job.finish("failed", 1, error=f"{step.name} exited with code {code}: {detail}")
        return 1
    return 1


def _prepare(tool: str, kwargs: JsonDict) -> JsonDict:
    """The validated request for an approved call, checked as the tool checks it."""
    from pydantic import validate_call

    if tool == "run_recon_all":
        from medmcp_neuro_core.tools.freesurfer.recon import prepare_recon_all

        return validate_call(prepare_recon_all)(**kwargs)
    from medmcp_neuro_core.tools.segmentation import prepare_segment_surfaces

    return validate_call(prepare_segment_surfaces)(**kwargs)


def call(tool: str, arguments: str) -> int:
    """Run the ``call`` subcommand: a tool call's arguments, validated, then ``run``.

    The arguments are the ones the person approved for the tool, so they go
    through the same checks (and the same type coercion FastMCP applies) as in the
    tool; a call the tool would refuse fails here with the same message.
    """
    from pydantic import ValidationError

    if tool not in JOB_TOOLS:
        print(f"{tool} cannot run as a job", file=sys.stderr)
        return 2
    try:
        parsed: object = json.loads(arguments)
    except json.JSONDecodeError as exc:
        print(f"arguments are not JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(parsed, dict):
        print("arguments must be a JSON object", file=sys.stderr)
        return 2
    kwargs: JsonDict = parsed  # pyright: ignore[reportUnknownVariableType]
    try:
        request = _prepare(tool, kwargs)
    except (ValidationError, TypeError, ValueError, OSError, RuntimeError) as exc:
        print(f"{tool} refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for warning in request.get("warnings", []):
        print(f"{LOG_PREFIX} input warning: {warning}", file=sys.stderr, flush=True)
    return run(request)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medmcp-neuro-core-job", description="medmcp-neuro-core background jobs"
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("call", help="run an approved tool call as a job")
    c.add_argument("tool", choices=sorted(JOB_TOOLS))
    c.add_argument("arguments", help="the tool call's arguments, as one JSON object")
    return p


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point."""
    args = _parser().parse_args(argv)
    code = call(args.tool, args.arguments)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
