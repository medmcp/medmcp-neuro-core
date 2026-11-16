"""FreeSurfer's environment, for FreeSurfer's child processes only.

The image does not set ``FREESURFER_HOME`` or put FreeSurfer's ``bin/`` on ``PATH``
image-wide: FreeSurfer ships programs whose names other tools in this image also
look for, and FastSurfer's segmentation, ANTs and HD-BET must keep finding what
they found before FreeSurfer was added. recon-all and FastSurfer's recon-surf get
FreeSurfer's environment (``FreeSurferEnv.sh``, written out) from here.
"""

import os
from collections.abc import Mapping
from pathlib import Path

# Where the image installs FreeSurfer; overridable for a host-native install.
HOME_ENV = "MEDMCP_FREESURFER_HOME"
DEFAULT_HOME = Path("/opt/freesurfer")


def freesurfer_home() -> Path:
    """The FreeSurfer installation (``$MEDMCP_FREESURFER_HOME``, else ``/opt/freesurfer``)."""
    override = os.environ.get(HOME_ENV, "").strip()
    return Path(override) if override else DEFAULT_HOME


def freesurfer_env(
    base: Mapping[str, str],
    *,
    license_path: str,
    subjects_dir: str,
    threads: int,
) -> dict[str, str]:
    """*base* plus FreeSurfer's environment, for one FreeSurfer child process.

    Args:
        base: The environment to extend (normally ``os.environ``).
        license_path: The user's licence file (checked by ``_license``); only its path
            goes into the environment.
        subjects_dir: ``SUBJECTS_DIR`` for the run.
        threads: OpenMP threads.
    """
    home = str(freesurfer_home())
    env = dict(base)
    env.update(
        {
            "FREESURFER_HOME": home,
            "FREESURFER": home,
            "FS_OVERRIDE": "0",
            "FSF_OUTPUT_FORMAT": "nii.gz",
            "FUNCTIONALS_DIR": f"{home}/sessions",
            "LOCAL_DIR": f"{home}/local",
            "FREESURFER_HOME_FSPYTHON": home,
            # The classic stream only: FreeSurfer 8's default stream needs models the
            # image does not ship.
            "FS_V8_XOPTS": "0",
            "PATH": f"{home}/bin{os.pathsep}{base.get('PATH', os.defpath)}",
            "FS_LICENSE": license_path,
            "SUBJECTS_DIR": subjects_dir,
            "OMP_NUM_THREADS": str(threads),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads),
        }
    )
    return env
