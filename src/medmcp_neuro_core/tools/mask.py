"""Label-mask selection tool for segmentation and lesion instance maps."""

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from medmcp_neuro_core.tools._neuro import (
    is_same_grid,
    nifti_load,
    nifti_save,
    nii_stem,
)


class SelectMaskResult(TypedDict):
    """Result of selecting label(s) of interest from a segmentation or instance map."""

    selected_mask_path: str
    overlaid_mask_path: str | None
    label_path: str
    overlay_label_path: str | None
    selected_label_ids: list[int]
    voxel_count: int
    _render: str


def _normalize_label_ids(label_ids: list[int] | int) -> list[int]:
    """Validate and deduplicate requested integer label IDs while preserving order."""
    raw = [label_ids] if isinstance(label_ids, int) else list(label_ids)
    if not raw:
        raise ValueError("label_ids must contain at least one positive integer label ID.")
    ids: list[int] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, bool) or item <= 0:
            raise ValueError(
                f"Invalid label ID {item!r}: label_ids must be positive integers (> 0)."
            )
        if item not in seen:
            seen.add(item)
            ids.append(item)
    return ids


def _format_id_tag(ids: list[int]) -> str:
    """Build a compact filename-safe tag for selected label IDs."""
    if len(ids) <= 5:
        return "-".join(str(i) for i in ids)
    return f"{ids[0]}-{ids[1]}-plus-{len(ids) - 2}"


def select_mask_of_interest(
    label_path: Path,
    label_ids: list[int] | int,
    overlay_label_path: Path | None = None,
    output_dir: Path | None = None,
) -> SelectMaskResult:
    """Select one or more label IDs from a segmentation or lesion instance map.

    Filters ``label_path`` to keep only the requested ``label_ids`` while **preserving
    their original integer label IDs** (no binarization or merging of labels).

    How to set ``label_path`` and ``overlay_label_path``:

    * **Leave ``overlay_label_path=None``** whenever you only need labels from a single
      map (do **not** pass the same file to both arguments; ``selected_mask_path``
      already preserves the original label IDs):
        - *FastSurfer brain regions*: ``label_path="sub-01_dseg.mgz"``,
          ``label_ids=[17, 53]``, ``overlay_label_path=None`` -> isolates left & right
          hippocampus while keeping IDs ``17`` and ``53`` distinct.
        - *Single tumor subregion*: ``label_path="sub-01_tumor-dseg.nii.gz"``,
          ``label_ids=3``, ``overlay_label_path=None`` -> isolates enhancing tumor
          across the whole scan.
        - *Single lesion mask only*: ``label_path="sub-01_lesion-map.nii.gz"``,
          ``label_ids=1``, ``overlay_label_path=None`` -> isolates lesion 1 from the
          instance map.
    * **Provide ``overlay_label_path`` ONLY when ``label_path`` and the multi-class
      segmentation are different maps** and you want to overlay the selected region
      onto the multi-class map to get both outputs:
        - *Lesion + tumor subregions*: ``label_path="sub-01_lesion-map.nii.gz"``,
          ``label_ids=1``, ``overlay_label_path="sub-01_tumor-dseg.nii.gz"`` -> returns
          both the filtered lesion instance mask (``selected_mask_path``) and lesion 1
          overlaid onto ``tumor-dseg`` (``overlaid_mask_path``) preserving tumor classes.

    Args:
        label_path: Path to the segmentation or lesion instance map (``.nii``,
            ``.nii.gz``, or ``.mgz``) from which ``label_ids`` are selected.
        label_ids: Single integer label ID or list of positive integer label IDs to
            keep.
        overlay_label_path: Optional path to a **different** multi-class segmentation
            map on the same voxel grid (e.g. ``*_tumor-dseg.nii.gz``) onto which the
            selected mask is overlaid. Omit (``None``) when selecting directly from
            FastSurfer ``dseg``, a single ``tumor-dseg``, or ``lesion-map`` alone.
        output_dir: Directory where output mask(s) are written. Defaults to
            ``label_path.parent``.

    Returns:
        ``SelectMaskResult`` with paths to the generated mask(s) and matched voxel count.

    Raises:
        FileNotFoundError: If ``label_path`` or ``overlay_label_path`` does not exist.
        ValueError: If ``label_ids`` is empty or non-positive, an image is corrupt/non-3D,
            or ``label_path`` and ``overlay_label_path`` do not share the same voxel grid.
    """
    ids = _normalize_label_ids(label_ids)
    label_data, affine, zooms = nifti_load(label_path, role="Label map", dtype=np.int64)

    selected_mask: NDArray[np.bool_] = np.isin(label_data, ids)
    voxel_count = int(selected_mask.sum())

    out_dir = output_dir if output_dir is not None else label_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nii_stem(label_path)
    id_tag = _format_id_tag(ids)

    selected_dtype = np.int32 if int(label_data.max()) > 32767 else np.int16
    selected_arr = np.where(selected_mask, label_data, 0).astype(selected_dtype)
    selected_mask_path = out_dir / f"{stem}_label-{id_tag}_dseg.nii.gz"
    nifti_save(selected_arr, affine, selected_mask_path, zooms=zooms)

    overlaid_mask_path_str: str | None = None
    overlay_label_path_str: str | None = None

    if overlay_label_path is not None:
        overlay_data, overlay_aff, _ = nifti_load(
            overlay_label_path, role="Overlay label map", dtype=np.int64
        )
        if not is_same_grid(label_data.shape, affine, overlay_data.shape, overlay_aff):
            raise ValueError(
                f"Grid mismatch between label_path ({label_data.shape[:3]}) and "
                f"overlay_label_path ({overlay_data.shape[:3]}): both images must share "
                "the same dimensions and affine."
            )

        overlay_dtype = np.int32 if int(overlay_data.max()) > 32767 else np.int16
        overlaid_arr = np.where(selected_mask, overlay_data, 0).astype(overlay_dtype)

        overlaid_mask_path = out_dir / f"{stem}_label-{id_tag}_overlay_dseg.nii.gz"
        nifti_save(overlaid_arr, affine, overlaid_mask_path, zooms=zooms)

        overlaid_mask_path_str = str(overlaid_mask_path)
        overlay_label_path_str = str(overlay_label_path)

    empty_note = (
        "\nNote: None of the requested label IDs were present in label_path; "
        "the generated mask is empty (0 voxels)."
        if voxel_count == 0
        else ""
    )
    overlay_line = (
        f"  Overlaid mask:     {overlaid_mask_path_str}\n"
        if overlaid_mask_path_str is not None
        else ""
    )

    return {
        "selected_mask_path": str(selected_mask_path),
        "overlaid_mask_path": overlaid_mask_path_str,
        "label_path": str(label_path),
        "overlay_label_path": overlay_label_path_str,
        "selected_label_ids": ids,
        "voxel_count": voxel_count,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the selected mask result as a compact key-value list:\n"
            f"  Label path:        {label_path}\n"
            f"  Selected IDs:      {ids}\n"
            f"  Voxel count:       {voxel_count}\n"
            f"  Selected mask:     {selected_mask_path}\n"
            f"{overlay_line}"
            "Omit internal keys."
            f"{empty_note}"
        ),
    }
