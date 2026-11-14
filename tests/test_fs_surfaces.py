"""segment_brain(surfaces=True): FastSurfer's surface pipeline as a background job.

A stand-in run_fastsurfer.sh writes what FastSurfer writes that the stack reads (the
segmentation files, the stats, ``scripts/recon-surf.done``), so the whole life cycle
runs without FastSurfer or FreeSurfer.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from medmcp_neuro_core.tools.freesurfer import _jobs
from medmcp_neuro_core.tools.freesurfer._env import freesurfer_env
from medmcp_neuro_core.tools.freesurfer.recon import recon_all_status
from medmcp_neuro_core.tools.segmentation import segment_brain

FAKE_FASTSURFER = """#!/usr/bin/env bash
# Stand-in for run_fastsurfer.sh: --sid <s> --sd <sd> (--seg_only | --surf_only) ...
mode=""; lic=""; segf=""; statf=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sid) s="$2"; shift ;; --sd) sd="$2"; shift ;; --fs_license) lic="$2"; shift ;;
    --asegdkt_segfile) segf="$2"; shift ;; --asegdkt_statsfile) statf="$2"; shift ;;
    --seg_only) mode=seg ;; --surf_only) mode=surf ;;
  esac; shift
done
d="$sd/$s"; mkdir -p "$d/mri" "$d/stats" "$d/scripts"
echo "$*" >> "$d/calls.txt"
if [ "$mode" = seg ]; then
  echo seg >> "$d/calls.txt"
  for f in orig.mgz mask.mgz aparc.DKTatlas+aseg.deep.mgz aseg.auto.mgz; do touch "$d/mri/$f"; done
  printf '%s\\n' '# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1200000.5, mm^3' \\
    '# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean' \\
    '  1  10  9000  9123.4  Left-Thalamus   95.0' > "$d/stats/aseg+DKT.VINN.stats"
  if [ -n "$segf" ]; then touch "$segf"; cp "$d/stats/aseg+DKT.VINN.stats" "$statf"; fi
  exit 0
fi
echo surf >> "$d/calls.txt"
echo "$lic" > "$d/fs_license_arg.txt"
echo "$FS_LICENSE" > "$d/fs_license_env.txt"
echo "$FREESURFER_HOME" > "$d/freesurfer_home.txt"
touch "$d/mri/wm.mgz"
echo 'recon-surf log' > "$d/scripts/recon-surf.log"
if [ -n "$FAKE_FAIL" ]; then
  echo "ERROR: mris_make_surfaces failed" >> "$d/scripts/recon-surf.log"; exit 1
fi
printf 'SUBJECT %s\\nRUNTIME_HOURS 0.850\\n' "$s" > "$d/scripts/recon-surf.done"
"""


@pytest.fixture
def fake_fastsurfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in, found through $RUN_FASTSURFER (as by the tool and the job)."""
    script = tmp_path / "fakebin" / "run_fastsurfer.sh"
    script.parent.mkdir()
    script.write_text(FAKE_FASTSURFER)
    script.chmod(0o755)
    monkeypatch.setenv("RUN_FASTSURFER", str(script))
    monkeypatch.delenv("FASTSURFER_PYTHON", raising=False)
    monkeypatch.setenv("MEDMCP_FREESURFER_HOME", str(tmp_path / "fshome"))
    return script


def _job(arguments: Mapping[str, object], **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "medmcp_neuro_core.job",
            "call",
            "segment_brain",
            json.dumps(arguments),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
    )


def _subject(workspace: Path) -> Path:
    return workspace / "out" / "fastsurfer" / "sub-01_T1w"


def test_without_surfaces_nothing_changes(workspace: Path, t1: Path, fake_fastsurfer: Path) -> None:
    """The default call segments only: no subject folder, no job, null surfaces."""
    result = segment_brain(t1, output_dir=workspace / "out", device="cpu")
    assert result["surfaces"] is None and result["medmcp_job"] is None
    assert not (workspace / "out" / "fastsurfer").exists()
    assert Path(result["volumes_path"]).is_file()


def test_without_licence_segmentation_still_runs(
    workspace: Path, t1: Path, fake_fastsurfer: Path
) -> None:
    """No licence: the licence-free segmentation runs, the surfaces say licence_required."""
    result = segment_brain(t1, output_dir=workspace / "out", device="cpu", surfaces=True)
    surfaces = result["surfaces"]
    assert surfaces is not None and surfaces["status"] == "licence_required"
    assert "registration" in surfaces["note"]
    assert result["medmcp_job"] is None
    assert Path(result["seg_path"]).exists()
    assert "licence" in result["_render"]
    assert not (workspace / ".medmcp" / "freesurfer" / "jobs").exists()


def test_core_job_request(
    workspace: Path, licence: Path, t1: Path, fake_fastsurfer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With MEDMCP_JOBS=1: segmentation into the subject folder, then a job request."""
    monkeypatch.setenv("MEDMCP_JOBS", "1")
    result = segment_brain(t1, output_dir=workspace / "out", device="cpu", surfaces=True)
    subject = _subject(workspace)
    assert result["medmcp_job"] == {"name": "FastSurfer surfaces sub-01_T1w"}
    surfaces = result["surfaces"]
    assert surfaces is not None and surfaces["status"] == "requested"
    assert surfaces["subject_dir"] == str(subject)
    assert surfaces["job_id"] == _jobs.job_id_for(subject)
    # The segmentation went to the subject folder; the usual outputs sit beside it.
    assert (subject / "mri" / "aseg.auto.mgz").exists()
    assert Path(result["seg_path"]) == workspace / "out" / "sub-01_T1w_dseg.mgz"
    assert "Left-Thalamus" in Path(result["volumes_path"]).read_text()
    # Only the segmentation ran: recon-surf is the job's.
    assert (subject / "calls.txt").read_text().split().count("surf") == 0


def test_job_runs_recon_surf_with_freesurfer_env(
    workspace: Path, licence: Path, t1: Path, fake_fastsurfer: Path, tmp_path: Path
) -> None:
    """The core's job command runs recon-surf (segmenting first if needed) and records done."""
    args = {
        "input_path": str(t1),
        "output_dir": str(workspace / "out"),
        "device": "cpu",
        "threads": 2,
        "surfaces": True,
    }
    proc = _job(args, MEDMCP_JOB_ID="abc123abc123")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    subject = _subject(workspace)
    calls = (subject / "calls.txt").read_text().split()
    assert calls.count("seg") == 1 and calls.count("surf") == 1
    # FreeSurfer got the licence *path* and its own environment.
    assert (subject / "fs_license_arg.txt").read_text().strip() == str(licence)
    assert (subject / "fs_license_env.txt").read_text().strip() == str(licence)
    assert (subject / "freesurfer_home.txt").read_text().strip() == str(tmp_path / "fshome")

    job_id = _jobs.job_id_for(subject)
    status = recon_all_status(job_id)
    assert status["status"] == "done"
    assert status["pipeline"].startswith("FastSurfer")
    assert status["runtime_hours"] == 0.85
    assert status["runs_in"] == "MedMCP background job"
    assert status["percent_done"] == 100


def test_job_records_failure(
    workspace: Path, licence: Path, t1: Path, fake_fastsurfer: Path
) -> None:
    """A failed recon-surf is reported with FastSurfer's own error line."""
    args = {
        "input_path": str(t1),
        "output_dir": str(workspace / "out"),
        "device": "cpu",
        "surfaces": True,
    }
    proc = _job(args, FAKE_FAIL="1")
    assert proc.returncode == 1
    status = recon_all_status(_jobs.job_id_for(_subject(workspace)))
    assert status["status"] == "failed"
    assert "mris_make_surfaces failed" in status["error"]


def test_job_refuses_what_the_tool_refuses(workspace: Path, licence: Path, t1: Path) -> None:
    """Only surfaces=True runs as a job; the arguments are validated again."""
    plain = _job({"input_path": str(t1), "device": "cpu"})
    assert plain.returncode == 1 and "surfaces=True" in plain.stderr
    licence.unlink()
    nolic = _job({"input_path": str(t1), "device": "cpu", "surfaces": True})
    assert nolic.returncode == 1 and "LicenseMissingError" in nolic.stderr


def test_second_surface_run_is_refused(
    workspace: Path, licence: Path, t1: Path, fake_fastsurfer: Path
) -> None:
    """FastSurfer does not run recon-surf twice over one subject folder."""
    (_subject(workspace) / "mri").mkdir(parents=True)
    (_subject(workspace) / "mri" / "wm.mgz").touch()
    with pytest.raises(FileExistsError, match="already run"):
        segment_brain(t1, output_dir=workspace / "out", device="cpu", surfaces=True)


def test_local_job_when_core_does_not_run_jobs(
    workspace: Path, licence: Path, t1: Path, fake_fastsurfer: Path
) -> None:
    """Without the core's offer, the tool starts the job as a detached process."""
    result = segment_brain(t1, output_dir=workspace / "out", device="cpu", surfaces=True)
    surfaces = result["surfaces"]
    assert surfaces is not None and surfaces["status"] in {"running", "done", "starting"}
    deadline = time.monotonic() + 30
    while recon_all_status(surfaces["job_id"])["status"] != "done":
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.3)


def test_freesurfer_env_is_for_children_only(tmp_path: Path) -> None:
    """FreeSurfer's environment extends a copy; the process's own stays untouched."""
    base = {"PATH": "/usr/bin"}
    env = freesurfer_env(base, license_path="/ws/l.txt", subjects_dir="/ws/s", threads=3)
    assert env["PATH"].endswith("/usr/bin") and env["PATH"].startswith(env["FREESURFER_HOME"])
    assert env["FS_LICENSE"] == "/ws/l.txt" and env["OMP_NUM_THREADS"] == "3"
    assert env["FS_V8_XOPTS"] == "0"
    assert base == {"PATH": "/usr/bin"}
    assert "FREESURFER_HOME" not in os.environ
