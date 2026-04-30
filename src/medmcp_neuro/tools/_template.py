"""MNI152 template auto-detection and on-demand download."""

import sys
import tempfile
import urllib.request
from pathlib import Path

_CACHE_DIR = Path.home() / ".medmcp_neuro" / "templates"
_MNI152_FILENAME = "MNI152NLin2009cAsym_res-01_T1w.nii.gz"
# templateflow public S3 — stable, versioned, widely used
_MNI152_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)


def _fsl_mni152() -> Path | None:
    """Return MNI152 1 mm T1w from ``$FSLDIR`` if available."""
    import os

    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        return None
    candidates = [
        Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm.nii.gz",
        Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz",
    ]
    return next((p for p in candidates if p.exists()), None)


def get_mni152_1mm() -> Path:
    """Return the MNI152NLin2009cAsym 1 mm T1w template, downloading on first use.

    Resolution order:
        1. Cached copy in ``~/.medmcp_neuro/templates/``.
        2. FSL standard library (``$FSLDIR/data/standard/``).
        3. Download from templateflow S3 and cache.

    Returns:
        Absolute path to the template NIfTI file.
    """
    cached = _CACHE_DIR / _MNI152_FILENAME
    if cached.exists():
        return cached

    fsl = _fsl_mni152()
    if fsl is not None:
        return fsl

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[medmcp-neuro] Downloading MNI152 template to {cached} …",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.NamedTemporaryFile(dir=_CACHE_DIR, suffix=".tmp", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        urllib.request.urlretrieve(_MNI152_URL, tmp)
        tmp.rename(cached)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return cached
