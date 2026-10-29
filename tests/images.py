"""Tiny valid images for the tool tests (the tools refuse an unreadable input)."""

from pathlib import Path

import nibabel as nib
import numpy as np


def write_nifti(path: Path, zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    """Write a tiny valid NIfTI with the given voxel zooms (mm)."""
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))  # pyright: ignore[reportUnknownMemberType]
    return path
