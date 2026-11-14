"""Sanity checks on the T1 image and the subject name before a one-hour run.

recon-all validates neither contrast nor resolution: it conforms every input to a
256^3 grid of 1 mm voxels and carries on, so a thick-slice clinical scan or a
FLAIR runs for an hour and produces surfaces that look plausible and are wrong.
These guards catch what the header and the file name can tell:

* **Dimensionality** -- recon-all wants one 3D volume. A 4D series is refused.
* **Resolution** -- the classic stream is built for ~1 mm isotropic T1. Mildly-off
  resolution is a warning; thick slices or strong anisotropy are refused unless
  ``force=True`` (as in neuro-core's FastSurfer guard).
* **Field of view** -- conforming to 256 mm crops anything larger.
* **Contrast** -- NIfTI headers carry no contrast field, so a file *name* that says
  FLAIR/T2/... is the only signal: a warning, never a block.
"""

import re
from pathlib import Path

# Thresholds as in medmcp-neuro-core's FastSurfer guard; recon-all's classic stream
# has the same ~1 mm design point (it conforms to 1 mm unless -cm/-hires is given).
_VOXEL_TARGET_MM: tuple[float, float] = (0.7, 1.3)  # comfortable band; outside -> warn
_VOXEL_BLOCK_MM: float = 2.0  # any voxel dim at/above this -> refuse (thick slices)
_ANISO_WARN: float = 1.5  # max/min zoom ratio above this -> warn
_ANISO_BLOCK: float = 2.0  # ...at/above this -> refuse
_CONFORM_FOV_MM: float = 256.0  # recon-all's conformed field of view

_NON_T1W_TOKENS: frozenset[str] = frozenset(
    {"flair", "t2w", "t2star", "t2", "dwi", "dti", "swi", "pd", "pdw", "bold", "asl", "angio", "ct"}
)
_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".mgz", ".mgh")
_SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")


def image_stem(path: Path) -> str:
    """File name without its image suffix (``sub-01_T1w.nii.gz`` -> ``sub-01_T1w``)."""
    name = path.name
    for suffix in _SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def default_subject_id(t1_path: Path) -> str:
    """A FreeSurfer subject name derived from the T1 file name."""
    stem = re.sub(r"[^A-Za-z0-9_.\-]+", "_", image_stem(t1_path)).strip("._-")
    return (stem or "subject")[:100]


def check_subject_id(subject_id: str) -> str:
    """Return *subject_id* if it is a plain folder name FreeSurfer accepts.

    Raises:
        ValueError: it contains a path separator, spaces or other characters.
    """
    if not _SUBJECT_RE.match(subject_id) or subject_id in {".", ".."}:
        raise ValueError(
            f"subject_id {subject_id!r} must be a plain folder name: letters, digits, "
            "'_', '-' and '.', starting with a letter or digit, at most 100 characters."
        )
    return subject_id


def _header(t1_path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """(shape, voxel sizes in mm) from the image header, without loading the data."""
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage

    img = nib.load(str(t1_path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(img, SpatialImage):
        raise ValueError(f"{t1_path.name} is not a spatial image (no voxel geometry)")
    shape = tuple(int(s) for s in img.shape)
    zooms = tuple(float(z) for z in img.header.get_zooms())
    return shape, zooms


def check_t1(t1_path: Path, force: bool = False) -> list[str]:
    """Check a T1 image before recon-all. Returns non-fatal warnings.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: not a NIfTI/MGZ image, not 3D, or (without ``force``) thick-slice
            or strongly anisotropic.
    """
    if not t1_path.is_file():
        raise FileNotFoundError(f"t1_path {t1_path} does not exist")
    if not t1_path.name.lower().endswith(_SUFFIXES):
        raise ValueError(
            f"{t1_path.name}: expected a NIfTI (.nii, .nii.gz) or MGZ (.mgz) T1 image; "
            "convert DICOM series to NIfTI first."
        )
    warnings: list[str] = []

    fields = {f for f in re.split(r"[^a-z0-9]+", image_stem(t1_path).lower()) if f}
    contrast = sorted(fields & _NON_T1W_TOKENS)
    if contrast:
        warnings.append(
            f"The file name names a non-T1 contrast ({', '.join(contrast)}). recon-all "
            "is built for T1-weighted MRI; on other contrasts its surfaces and "
            "segmentation are unreliable. The header cannot confirm the contrast, so "
            "this is a warning only."
        )

    try:
        shape, zooms = _header(t1_path)
    except Exception as exc:
        raise ValueError(f"{t1_path.name} could not be read as an image: {exc}") from exc

    spatial = shape[:3]
    extra = [s for s in shape[3:] if s != 1]
    if len(spatial) < 3 or extra:
        raise ValueError(
            f"{t1_path.name} has shape {shape}; recon-all needs one 3D volume. Pick a "
            "single 3D T1 (for a 4D series, extract one volume first)."
        )
    zooms = zooms[:3]
    if len(zooms) < 3 or any(z <= 0 for z in zooms):
        warnings.append(f"Unexpected voxel sizes {zooms}; resolution not checked.")
        return warnings

    lo, hi = min(zooms), max(zooms)
    aniso = hi / lo
    res = "x".join(f"{z:.2f}" for z in zooms) + " mm"
    if hi >= _VOXEL_BLOCK_MM or aniso >= _ANISO_BLOCK:
        msg = (
            f"Voxel size {res} (anisotropy {aniso:.1f}x) is far from the ~1 mm isotropic "
            "T1 recon-all is built for: thick-slice or strongly anisotropic scans "
            "(typical 2D clinical acquisitions) give unreliable surfaces and thickness."
        )
        if not force:
            raise ValueError(f"{msg} Pass force=True to run anyway.")
        warnings.append(f"{msg} Running anyway because force=True.")
    elif aniso >= _ANISO_WARN or not (_VOXEL_TARGET_MM[0] <= lo and hi <= _VOXEL_TARGET_MM[1]):
        warnings.append(
            f"Voxel size {res} (anisotropy {aniso:.1f}x) is outside the "
            f"~{_VOXEL_TARGET_MM[0]:g}-{_VOXEL_TARGET_MM[1]:g} mm isotropic range recon-all "
            "is designed for; results may be less accurate. recon-all resamples to 1 mm."
        )

    fov = [n * z for n, z in zip(spatial, zooms, strict=True)]
    if max(fov) > _CONFORM_FOV_MM + 1e-3:
        warnings.append(
            "Field of view "
            + "x".join(f"{f:.0f}" for f in fov)
            + f" mm is larger than the {_CONFORM_FOV_MM:.0f} mm recon-all conforms to; "
            "anything outside the central 256 mm is cropped. Usually only neck or "
            "background is lost, but check the head is not cut off."
        )
    return warnings
