"""MNI152 template auto-detection and on-demand download."""

import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

TemplateName = Literal["MNI152NLin2009cAsym", "MNI152NLin6Asym"]

_CACHE_DIR = Path.home() / ".medmcp_neuro_core" / "templates"
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

_MNI152_BRATS_FILENAME = "MNI152NLin6Asym_res-01_T1w.nii.gz"
_MNI152_BRATS_BRAIN_FILENAME = "MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"

_MNI152_BRATS_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/"
    "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz"
)
_MNI152_BRATS_BRAIN_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/"
    "tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"
)

_TEMPLATES: dict[tuple[TemplateName, bool], tuple[str, str]] = {
    ("MNI152NLin2009cAsym", False): (_MNI152_FILENAME, _MNI152_URL),
    ("MNI152NLin2009cAsym", True): (_MNI152_BRAIN_FILENAME, _MNI152_BRAIN_URL),
    ("MNI152NLin6Asym", False): (_MNI152_BRATS_FILENAME, _MNI152_BRATS_URL),
    ("MNI152NLin6Asym", True): (_MNI152_BRATS_BRAIN_FILENAME, _MNI152_BRATS_BRAIN_URL),
}


def get_mni152_1mm(
    skull_stripped: bool = False,
    name: TemplateName = "MNI152NLin2009cAsym",
) -> Path:
    """Return a 1 mm MNI152 T1w template, downloading on first use.

    Resolution order:
        1. Cached copy in ``~/.medmcp_neuro_core/templates/``.
        2. Download from templateflow S3 and cache.

    There is deliberately no ``$FSLDIR`` fallback: even for ``MNI152NLin6Asym``,
    reading only from the explicit templateflow cache guarantees identical voxel
    geometry and provenance across source and container runs.

    Args:
        skull_stripped: When ``True``, return the brain-extracted (``desc-brain``)
            variant. Use this when the input image has already been skull-stripped
            so that skull tissue in the template does not degrade registration quality.
        name: TemplateFlow MNI152 space identifier. Defaults to
            ``"MNI152NLin2009cAsym"``; also supports ``"MNI152NLin6Asym"``.

    Returns:
        Absolute path to the template NIfTI file.

    Raises:
        ValueError: If ``name`` is not a supported template identifier.
    """
    key = (name, skull_stripped)
    if key not in _TEMPLATES:
        raise ValueError(
            f"Unknown template {name!r}; expected one of: MNI152NLin2009cAsym, MNI152NLin6Asym"
        )
    filename, url = _TEMPLATES[key]

    cached = _CACHE_DIR / filename
    if cached.exists():
        return cached

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[medmcp-neuro-core] Downloading {name} template to {cached} …",
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
