"""MNI152 template auto-detection and on-demand download."""

import sys
import tempfile
import urllib.request
from pathlib import Path

_CACHE_DIR = Path.home() / ".medmcp_neuro" / "templates"
_MNI152_FILENAME = "MNI152NLin2009cAsym_res-01_T1w.nii.gz"
_MNI152_BRAIN_FILENAME = "MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"
# templateflow public S3 — stable, versioned, widely used
_MNI152_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)
_MNI152_BRAIN_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"
)


def _fsl_mni152(skull_stripped: bool = False) -> Path | None:
    """Return MNI152 1 mm T1w from ``$FSLDIR`` if available."""
    import os

    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        return None
    std = Path(fsldir) / "data" / "standard"
    if skull_stripped:
        candidates = [std / "MNI152_T1_1mm_brain.nii.gz"]
    else:
        candidates = [
            std / "MNI152_T1_1mm.nii.gz",
            std / "MNI152_T1_1mm_brain.nii.gz",
        ]
    return next((p for p in candidates if p.exists()), None)


def get_mni152_1mm(skull_stripped: bool = False) -> Path:
    """Return the MNI152NLin2009cAsym 1 mm T1w template, downloading on first use.

    Resolution order:
        1. Cached copy in ``~/.medmcp_neuro/templates/``.
        2. FSL standard library (``$FSLDIR/data/standard/``).
        3. Download from templateflow S3 and cache.

    Args:
        skull_stripped: When ``True``, return the brain-extracted (``desc-brain``)
            variant. Use this when the input image has already been skull-stripped
            so that skull tissue in the template does not degrade registration quality.

    Returns:
        Absolute path to the template NIfTI file.
    """
    filename = _MNI152_BRAIN_FILENAME if skull_stripped else _MNI152_FILENAME
    url = _MNI152_BRAIN_URL if skull_stripped else _MNI152_URL

    cached = _CACHE_DIR / filename
    if cached.exists():
        return cached

    fsl = _fsl_mni152(skull_stripped)
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
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(cached)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return cached
