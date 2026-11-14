"""FastSurfer's surface pipeline (recon-surf) behind ``segment_brain(surfaces=True)``.

FastSurfer's segmentation needs no licence and takes about a minute; its surface
pipeline (cortical surfaces, thickness, parcellation stats) runs FreeSurfer's
programs, so it needs the user's FreeSurfer licence and about an hour of CPU time.
segment_brain therefore runs the segmentation as always, into a FastSurfer subject
folder, and hands the surfaces to a background job:

1. ``<output_dir>/fastsurfer/<subject>/`` is the subject folder (``--sd`` /
   ``--sid``). The tool runs FastSurfer's segmentation there (``--seg_only``, with
   the corpus-callosum module that recon-surf requires; CerebNet and HypVINN off).
2. The job (``medmcp-neuro-core-job call segment_brain <the same arguments>``)
   validates the call exactly as the tool does, runs the segmentation itself if it
   is not there yet (a job started by hand), then ``run_fastsurfer.sh --surf_only``
   with FreeSurfer's environment (see :mod:`._env`).

FastSurfer 2.5.4 is written against FreeSurfer 7.4.1 and checks the version; the
image has FreeSurfer 8.2.0, so the job passes ``--ignore_fs_version``. The
recon-all steps FastSurfer calls are the classic ones (``FS_V8_XOPTS=0``).
"""

from pathlib import Path
from typing import Any

from medmcp_neuro_core.tools.freesurfer import _jobs
from medmcp_neuro_core.tools.freesurfer._inputs import check_subject_id, default_subject_id
from medmcp_neuro_core.tools.freesurfer._license import find_license
from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root

JsonDict = dict[str, Any]

KIND = "fastsurfer-surfaces"
SUBJECTS_FOLDER = "fastsurfer"

TIME_NOTE = (
    "FastSurfer's surface pipeline takes about an hour per scan on the CPU (the "
    "segmentation before it takes about a minute on a GPU)."
)


def surfaces_started(subject_dir: Path) -> bool:
    """Whether recon-surf has already run (or started) in *subject_dir*.

    FastSurfer refuses to run its surface pipeline over these files (it would mix two
    runs); so does this stack.
    """
    mri = subject_dir / "mri"
    return (mri / "wm.mgz").exists() or (mri / "aparc.DKTatlas+aseg.orig.mgz").exists()


def segmentation_present(subject_dir: Path) -> bool:
    """Whether the segmentation recon-surf builds on is in *subject_dir*.

    recon-surf needs the conformed T1, the brain mask, the DKT+aseg segmentation and
    the aseg with the corpus callosum (written by FastSurfer's CC module).
    """
    mri = subject_dir / "mri"
    needed = ("orig.mgz", "mask.mgz", "aparc.DKTatlas+aseg.deep.mgz", "aseg.auto.mgz")
    return all((mri / name).is_file() for name in needed)


def prepare_surfaces(
    input_path: str | Path,
    output_dir: str | Path | None,
    device: str,
    threads: int,
    force: bool,
    license_path: str | None,
) -> JsonDict:
    """Validate a ``segment_brain(surfaces=True)`` call and return the job request.

    Shared by the tool and the job entrypoint, so the job refuses what the tool
    refuses. The image checks (readable, resolution, contrast) are segment_brain's own
    and run before this.

    Raises:
        LicenseMissingError: no licence file (the message says how to get one).
        FileExistsError: the surface pipeline already ran in this subject folder.
        ValueError: a path outside the workspace, or a bad thread count.
    """
    ws = workspace_root()
    t1 = resolve_inside(input_path, "input_path", ws)
    out = resolve_inside(output_dir, "output_dir", ws) if output_dir is not None else t1.parent
    subjects_dir = out / SUBJECTS_FOLDER
    sid = check_subject_id(default_subject_id(t1))
    subject_dir = subjects_dir / sid
    if threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    licence = find_license(license_path)
    if surfaces_started(subject_dir):
        raise FileExistsError(
            f"FastSurfer's surface pipeline has already run in {subject_dir}. It does not "
            "run twice over one subject folder; use another output_dir (or remove the "
            "old folder yourself)."
        )
    return {
        "kind": KIND,
        "t1_path": str(t1),
        "output_dir": str(out),
        "subjects_dir": str(subjects_dir),
        "subject_id": sid,
        "subject_dir": str(subject_dir),
        "license_path": str(licence),
        "device": device,
        "threads": threads,
        "force": force,
        "job_id": _jobs.job_id_for(subject_dir),
        "registry": str(_jobs.registry(out)),
    }


def seg_command(request: JsonDict, binary: str, fs_python: str | None, device: str) -> list[str]:
    """FastSurfer's segmentation into the subject folder, as recon-surf needs it."""
    cmd = [
        binary,
        "--t1",
        str(request["t1_path"]),
        "--sid",
        str(request["subject_id"]),
        "--sd",
        str(request["subjects_dir"]),
        "--seg_only",
        # Not baked into the image, and not needed by recon-surf. The CC module stays
        # on: recon-surf requires the aseg with the corpus callosum.
        "--no_cereb",
        "--no_hypothal",
        "--device",
        device,
        "--threads",
        str(request["threads"]),
        "--allow_root",
    ]
    if fs_python:
        cmd += ["--py", fs_python]
    return cmd


def surf_command(request: JsonDict, binary: str, fs_python: str | None) -> list[str]:
    """FastSurfer's surface pipeline on the segmented subject (needs the licence)."""
    cmd = [
        binary,
        "--sid",
        str(request["subject_id"]),
        "--sd",
        str(request["subjects_dir"]),
        "--surf_only",
        "--fs_license",
        str(request["license_path"]),
        # FastSurfer 2.5.4 checks for FreeSurfer 7.4.1; the image has 8.2.0.
        "--ignore_fs_version",
        "--threads",
        str(request["threads"]),
        "--allow_root",
    ]
    if fs_python:
        cmd += ["--py", fs_python]
    return cmd


def job_arguments(request: JsonDict) -> JsonDict:
    """The ``call segment_brain`` arguments that reproduce a validated request."""
    return {
        "input_path": request["t1_path"],
        "output_dir": request["output_dir"],
        "device": request["device"],
        "threads": request["threads"],
        "force": request["force"],
        "surfaces": True,
        "license_path": request["license_path"],
    }


def surf_progress(subject_dir: Path) -> JsonDict:
    """What FastSurfer's own files say about a surface run (as ``_jobs.fs_progress``).

    recon-surf calls recon-all for single steps, and each call writes its own
    ``recon-all.done``, so only ``scripts/recon-surf.done`` means the run is
    finished. The current step is the last ``#@#`` line recon-all logged; there is no
    percentage (FastSurfer's step times differ from recon-all's).
    """
    fs = _jobs.fs_progress(subject_dir)
    scripts = subject_dir / "scripts"
    out: JsonDict = {
        "fs_status": "not started",
        "current_step": fs["current_step"],
        "steps_done": fs["steps_done"],
        "fraction": None,
        "runtime_hours": None,
        "error_lines": [],
    }
    done = scripts / "recon-surf.done"
    if done.is_file():
        out["fs_status"] = "done"
        out["fraction"] = 1.0
        try:
            for line in done.read_text(errors="replace").splitlines():
                key, _, value = line.partition(" ")
                if key in {"RUNTIME_HOURS", "RUN_TIME_HOURS"}:
                    out["runtime_hours"] = float(value.split()[0])
        except (OSError, ValueError, IndexError):
            pass
    elif (scripts / "recon-surf.log").is_file():
        out["fs_status"] = "running" if list(scripts.glob("IsRunning.*")) else "stopped"
    return out


def surf_errors(subject_dir: Path, limit: int = 8) -> list[str]:
    """The last ERROR lines of ``scripts/recon-surf.log``."""
    log = subject_dir / "scripts" / "recon-surf.log"
    try:
        lines = log.read_text(errors="replace").splitlines()[-400:]
    except OSError:
        return []
    return [ln.strip() for ln in lines if "ERROR" in ln][-limit:]
