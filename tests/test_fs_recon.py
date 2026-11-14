"""run_recon_all / the job entrypoint / recon_all_status, with a stand-in recon-all.

The stand-in writes what recon-all writes that the stack reads (the status log, the
done or error file), so the whole job life cycle runs without FreeSurfer.
"""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from medmcp_neuro_core.tools.freesurfer import _jobs
from medmcp_neuro_core.tools.freesurfer.recon import (
    prepare_recon_all,
    recon_all_command,
    recon_all_status,
    run_recon_all,
)

FAKE_RECON_ALL = """#!/usr/bin/env bash
# Stand-in for recon-all: -s <subject> -i <t1> -sd <subjects_dir> ...
while [ $# -gt 0 ]; do
  case "$1" in -s) s="$2"; shift ;; -sd) sd="$2"; shift ;; esac; shift
done
d="$sd/$s/scripts"; mkdir -p "$d"
echo "$FS_LICENSE" > "$sd/$s/fs_license_path.txt"
echo "$OMP_NUM_THREADS" > "$sd/$s/omp_threads.txt"
echo 'status file for recon-all' > "$d/recon-all-status.log"
printf '#@# MotionCor Sat Nov 14 08:25:33 UTC 2026\\n' >> "$d/recon-all-status.log"
printf '#@# CA Reg Sat Nov 14 08:29:14 UTC 2026\\n' >> "$d/recon-all-status.log"
if [ -n "$FAKE_FAIL" ]; then
  echo "ERROR: talairach_avi failed, see transforms/talairach_avi.log" > "$d/recon-all.log"
  touch "$d/recon-all.error"; exit 1
fi
printf 'SUBJECT %s\\nRUNTIME_HOURS 0.920\\n' "$s" > "$d/recon-all.done"
"""


@pytest.fixture
def fake_freesurfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the stand-in recon-all first on PATH."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = bindir / "recon-all"
    script.write_text(FAKE_RECON_ALL)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir


def _job(arguments: Mapping[str, object], **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "medmcp_neuro_core.job",
            "call",
            "run_recon_all",
            json.dumps(arguments),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
    )


def test_prepare_builds_the_classic_command(workspace: Path, licence: Path, t1: Path) -> None:
    """The request runs recon-all -all -no-v8 -parallel -openmp N on the given scan."""
    req = prepare_recon_all(str(t1), str(workspace / "subjects"), threads=2)
    assert req["subject_id"] == "sub-01_T1w"
    assert req["license_path"] == str(licence)
    assert recon_all_command(req) == [
        "recon-all",
        "-s",
        "sub-01_T1w",
        "-i",
        str(t1),
        "-sd",
        str(workspace / "subjects"),
        "-all",
        "-no-v8",
        "-parallel",
        "-openmp",
        "2",
    ]


def test_without_licence_nothing_starts(workspace: Path, t1: Path) -> None:
    """No licence: status licence_required, instructions, no job record, no folder."""
    result = run_recon_all(str(t1), str(workspace / "subjects"))
    assert result["status"] == "licence_required"
    assert "registration" in (result["error"] or "")
    assert "medmcp_job" not in result
    assert not (workspace / "subjects").exists()
    assert not (workspace / ".medmcp" / "freesurfer").exists()


def test_existing_subject_is_refused(workspace: Path, licence: Path, t1: Path) -> None:
    """recon-all never overwrites a subject folder."""
    (workspace / "subjects" / "sub-01_T1w").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_recon_all(str(t1), str(workspace / "subjects"))


def test_only_classic_pipeline(workspace: Path, licence: Path, t1: Path) -> None:
    """The v8 stream needs models this image does not ship."""
    with pytest.raises(ValueError, match="classic"):
        prepare_recon_all(str(t1), str(workspace / "s"), pipeline="v8")  # pyright: ignore[reportArgumentType]


def test_core_job_request(
    workspace: Path, licence: Path, t1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With MEDMCP_JOBS=1 the tool only validates and asks the core for a job."""
    monkeypatch.setenv("MEDMCP_JOBS", "1")
    result = run_recon_all(str(t1), str(workspace / "subjects"), subject_id="s1")
    assert result["status"] == "requested"
    assert result.get("medmcp_job") == {"name": "recon-all s1"}
    assert "started: true" in result["_render"]
    assert not (workspace / "subjects" / "s1").exists()
    assert not (workspace / ".medmcp" / "freesurfer" / "jobs").exists()


def test_job_runs_recon_all_and_records_done(
    workspace: Path, licence: Path, t1: Path, fake_freesurfer: Path
) -> None:
    """The core's job command line runs recon-all and records the outcome."""
    progress = workspace / ".medmcp" / "jobs" / "abc.json"
    progress.parent.mkdir(parents=True)
    args = {"t1_path": str(t1), "output_dir": str(workspace / "subjects"), "threads": 1}
    proc = _job(args, MEDMCP_JOB_ID="abc123abc123", MEDMCP_JOB_PROGRESS=str(progress))
    assert proc.returncode == 0, proc.stderr + proc.stdout

    subject = workspace / "subjects" / "sub-01_T1w"
    # FreeSurfer got the licence *path*, and the requested thread count.
    assert (subject / "fs_license_path.txt").read_text().strip() == str(licence)
    assert (subject / "omp_threads.txt").read_text().strip() == "1"

    job_id = _jobs.job_id_for(subject)
    state = _jobs.read_state(workspace / ".medmcp" / "freesurfer" / "jobs" / job_id)
    assert state is not None
    assert state["status"] == "done" and state["exit_code"] == 0
    assert state["core_job_id"] == "abc123abc123"
    assert json.loads(progress.read_text())["fraction"] == 1.0

    status = recon_all_status(job_id)
    assert status["status"] == "done"
    assert status["runtime_hours"] == 0.92
    assert status["runs_in"] == "MedMCP background job"
    # Also found by subject folder and by subject id, and listed.
    assert recon_all_status(str(subject))["job_id"] == job_id
    assert recon_all_status("sub-01_T1w")["job_id"] == job_id
    assert [r["job_id"] for r in recon_all_status()["runs"]] == [job_id]


def test_job_records_recon_all_failure(
    workspace: Path, licence: Path, t1: Path, fake_freesurfer: Path
) -> None:
    """A failed recon-all is reported with FreeSurfer's own error line."""
    args = {"t1_path": str(t1), "output_dir": str(workspace / "subjects")}
    proc = _job(args, FAKE_FAIL="1")
    assert proc.returncode == 1
    status = recon_all_status("sub-01_T1w")
    assert status["status"] == "failed"
    assert "talairach_avi failed" in status["error"]


def test_job_refuses_what_the_tool_refuses(workspace: Path, t1: Path) -> None:
    """The job entrypoint validates the approved arguments again (no licence here)."""
    proc = _job({"t1_path": str(t1), "output_dir": str(workspace / "subjects")})
    assert proc.returncode == 1
    assert "LicenseMissingError" in proc.stderr
    bad = _job({"t1_path": str(t1), "output_dir": "/etc"})
    assert bad.returncode == 1 and "outside the workspace" in bad.stderr


def test_local_job_when_core_does_not_run_jobs(
    workspace: Path, licence: Path, t1: Path, fake_freesurfer: Path
) -> None:
    """Without the core's offer, the tool starts the job as a detached process."""
    result = run_recon_all(str(t1), str(workspace / "subjects"), threads=1)
    assert result["status"] in {"running", "done", "starting"}, result
    assert "medmcp_job" not in result
    deadline = time.monotonic() + 30
    while recon_all_status(result["job_id"])["status"] != "done":
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.3)


def test_dead_job_reads_as_interrupted(workspace: Path) -> None:
    """A 'running' record whose heartbeat stopped elsewhere is reported interrupted."""
    subject = workspace / "subjects" / "s9"
    (subject / "scripts").mkdir(parents=True)
    (subject / "scripts" / "recon-all-status.log").write_text(
        "#@# SubCort Seg Sat Nov 14 08:42:22 UTC 2026\n"
    )
    job_id = _jobs.job_id_for(subject)
    jdir = workspace / ".medmcp" / "freesurfer" / "jobs" / job_id
    _jobs.write_state(
        jdir,
        {
            "job_id": job_id,
            "status": "running",
            "hostname": socket.gethostname() + "-elsewhere",
            "pid": 1,
            "started_at": time.time() - 3600,
            "heartbeat_at": time.time() - 600,
            "subject_id": "s9",
            "subject_dir": str(subject),
        },
    )
    status = recon_all_status(job_id)
    assert status["status"] == "interrupted"
    assert status["current_step"] == "SubCort Seg"
    assert status["percent_done"] == 30


def test_status_of_unknown_run(workspace: Path) -> None:
    """An unknown id says how to find runs."""
    with pytest.raises(FileNotFoundError, match="without an"):
        recon_all_status("nothing-here")
