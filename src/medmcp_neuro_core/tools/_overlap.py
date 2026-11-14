"""Lesion/atlas overlap by voxel counting — no statistics, no model.

Every number is a voxel count times the voxel volume taken from the image header
(``prod(spacing)``), or a ratio of two such volumes. Kept free of ANTs/nibabel so it
can be unit-tested on synthetic arrays and imported by the ANTs subprocess.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# A lesion-mask voxel counts as lesion when its value exceeds this. Binary masks
# (0/1, 0/255, label 1..n) are unaffected; soft masks are cut at one half.
LESION_THRESHOLD = 0.5

Row = dict[str, float | int | str | None]


def lesion_voxels(mask: NDArray[Any]) -> NDArray[np.bool_]:
    """Binarise a lesion mask (value > ``LESION_THRESHOLD``)."""
    return np.asarray(mask) > LESION_THRESHOLD


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 4) if whole > 0 else None


def probabilistic_row(
    name: str,
    lesion: NDArray[np.bool_],
    prob: NDArray[Any],
    voxel_mm3: float,
    threshold: float,
) -> Row:
    """Overlap of a binary lesion mask with one probabilistic tract map (same grid).

    Thresholded: the tract is the set of voxels with probability >= ``threshold``.
    Weighted: each voxel counts with its probability, so the tract "volume" is
    ``sum(p) * voxel_mm3`` and the lesion's share is ``sum(p over lesion) * voxel_mm3``.

    Returns:
        One CSV row (volumes in mm³, percentages 0-100; ``None`` when undefined).
    """
    if lesion.shape != prob.shape:
        raise ValueError(f"Shape mismatch: lesion {lesion.shape} vs map {prob.shape}")
    p = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    in_tract = p >= threshold
    tract_vox = int(in_tract.sum())
    hit_vox = int((in_tract & lesion).sum())
    lesion_vox = int(lesion.sum())
    w_tract = float(p.sum(dtype=np.float64))
    w_hit = float(p[lesion].sum(dtype=np.float64))
    return {
        "tract": name,
        "tract_volume_mm3": round(tract_vox * voxel_mm3, 4),
        "lesion_in_tract_mm3": round(hit_vox * voxel_mm3, 4),
        "percent_tract_lesioned": _pct(hit_vox, tract_vox),
        "percent_lesion_in_tract": _pct(hit_vox, lesion_vox),
        "tract_weighted_volume_mm3": round(w_tract * voxel_mm3, 4),
        "lesion_in_tract_weighted_mm3": round(w_hit * voxel_mm3, 4),
        "percent_tract_lesioned_weighted": _pct(w_hit, w_tract),
    }


def label_rows(
    lesion: NDArray[np.bool_],
    labels: NDArray[Any],
    voxel_mm3: float,
    table: list[tuple[int, str]],
) -> list[Row]:
    """Overlap of a binary lesion mask with every region of a label atlas (same grid).

    Args:
        lesion: Binary lesion mask.
        labels: Integer label map (0 = background), already on the lesion grid.
        voxel_mm3: Voxel volume from the header.
        table: ``(label id, region name)`` for every region to report.

    Returns:
        One row per region, then an ``outside atlas`` row for lesion voxels on label 0.
    """
    if lesion.shape != labels.shape:
        raise ValueError(f"Shape mismatch: lesion {lesion.shape} vs labels {labels.shape}")
    lab = np.rint(np.asarray(labels, dtype=np.float64)).astype(np.int64)
    lab[lab < 0] = 0
    size = max(int(lab.max()), max((i for i, _ in table), default=0)) + 1
    region_vox = np.bincount(lab.ravel(), minlength=size)
    hit_vox = np.bincount(lab[lesion].ravel(), minlength=size)
    lesion_vox = int(lesion.sum())
    rows: list[Row] = []
    for idx, name in table:
        r, h = int(region_vox[idx]), int(hit_vox[idx])
        rows.append(
            {
                "label": idx,
                "region": name,
                "region_volume_mm3": round(r * voxel_mm3, 4),
                "lesion_in_region_mm3": round(h * voxel_mm3, 4),
                "percent_region_lesioned": _pct(h, r),
                "percent_lesion_in_region": _pct(h, lesion_vox),
            }
        )
    outside = int(hit_vox[0])
    rows.append(
        {
            "label": 0,
            "region": "outside atlas",
            "region_volume_mm3": None,
            "lesion_in_region_mm3": round(outside * voxel_mm3, 4),
            "percent_region_lesioned": None,
            "percent_lesion_in_region": _pct(outside, lesion_vox),
        }
    )
    return rows


def frequency(masks: list[NDArray[np.bool_]]) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Voxel-wise lesion count and frequency (count / n) over masks on one grid."""
    if not masks:
        raise ValueError("No masks given")
    shape = masks[0].shape
    count = np.zeros(shape, dtype=np.int32)
    for m in masks:
        if m.shape != shape:
            raise ValueError(f"Shape mismatch: {m.shape} vs {shape}")
        count += m
    return count, (count / np.float32(len(masks))).astype(np.float32)
