"""Atlas-based lesion load and cohort lesion-frequency maps.

Two questions, answered by voxel counting only (no statistics):

* ``lesion_load_by_atlas`` — how much of a lesion mask falls in each atlas region or
  white-matter tract (e.g. the corticospinal tract), and what share of each
  region/tract that is.
* ``lesion_frequency_map`` — for a cohort of masks in template space, the fraction of
  subjects with a lesion at each voxel, plus a figure.

Direction of the warp (lesion load): the **atlas is brought into the lesion mask's
space**, never the mask into the template. The mask is the measurement — resampling
it into MNI would change its volume by the Jacobian of the warp and blur its edges,
so the mm³ reported would no longer be the patient's. Warping the atlas instead keeps
every lesion voxel exactly as segmented, and volumes come from the mask's own header
spacing. Label atlases are resampled nearest-neighbour (labels stay labels);
probabilistic tract maps are resampled linearly (probabilities stay in [0, 1]).
"""

import csv
import sys
from pathlib import Path
from typing import Any, TypedDict

from medmcp_neuro_core.tools._atlas import ATLASES, AtlasName, load_atlas
from medmcp_neuro_core.tools._neuro import is_same_grid, nii_stem
from medmcp_neuro_core.tools._overlap import Row, frequency, lesion_voxels
from medmcp_neuro_core.tools._template import get_mni152_1mm
from medmcp_neuro_core.tools.registration import (
    _check_antspy,  # pyright: ignore[reportPrivateUsage]
    _run_antspy,  # pyright: ignore[reportPrivateUsage]
)

_MNI_SPACE = "MNI152NLin2009cAsym"
_TIMEOUT = 3600
_MAX_LISTED_ROWS = 20

_LABEL_COLUMNS = [
    "label",
    "region",
    "region_volume_mm3",
    "lesion_in_region_mm3",
    "percent_region_lesioned",
    "percent_lesion_in_region",
]
_TRACT_COLUMNS = [
    "tract",
    "tract_volume_mm3",
    "lesion_in_tract_mm3",
    "percent_tract_lesioned",
    "percent_lesion_in_tract",
    "tract_weighted_volume_mm3",
    "lesion_in_tract_weighted_mm3",
    "percent_tract_lesioned_weighted",
]

_WORDING = (
    "WORDING: these are geometric overlaps of the lesion mask with a population "
    "atlas registered to this scan — say 'lesion overlap with the <atlas> "
    "<region/tract> map', never that a tract 'is damaged/disrupted' or that a "
    "deficit follows. The atlas shows where a tract usually runs in healthy adults, "
    "not this patient's anatomy, and accuracy depends on the registration."
)


class AtlasInfo(TypedDict):
    """One bundled atlas."""

    name: str
    kind: str
    title: str
    space: str
    licence: str


class ListAtlasesResult(TypedDict):
    """Bundled atlases."""

    atlases: list[AtlasInfo]
    _render: str


class LesionLoadResult(TypedDict):
    """Per-region / per-tract lesion load."""

    csv_path: str
    lesion_mask_path: str
    atlas: str
    atlas_kind: str
    atlas_title: str
    method: str
    probability_threshold: float | None
    lesion_volume_mm3: float
    voxel_volume_mm3: float
    rows_with_lesion: list[Row]
    corticospinal_rows: list[Row]
    warped_atlas_path: str | None
    attribution: str
    _render: str


class LesionFrequencyResult(TypedDict):
    """Cohort lesion-frequency map."""

    frequency_map_path: str
    count_map_path: str
    figure_path: str
    n_masks: int
    max_frequency: float
    voxels_with_any_lesion: int
    template_path: str
    template_space_masks: list[str]
    attribution: str
    _render: str


# ── list_atlases ───────────────────────────────────────────────────────────────


def list_atlases() -> ListAtlasesResult:
    """List the brain atlases bundled for ``lesion_load_by_atlas``.

    Every atlas listed here is licensed for redistribution and baked into the
    container image (see NOTICE). Tract atlases are probabilistic (one map per
    tract); region atlases are label maps.

    Returns:
        ``ListAtlasesResult`` with name, kind, title, space, and licence per atlas.
    """
    infos: list[AtlasInfo] = [
        {
            "name": s.name,
            "kind": s.kind,
            "title": s.title,
            "space": s.space,
            "licence": s.licence,
        }
        for s in ATLASES.values()
    ]
    return {
        "atlases": infos,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "List each atlas as '<name> — <title> (<kind>, <licence>)'.\n"
            "For corticospinal-tract questions recommend 'pandora_tractseg' (default) and "
            "offer 'hcp1065' as a second opinion. 'schaefer2018_*' are cortical-only "
            "parcellations (no white matter, no deep grey matter)."
        ),
    }


# ── lesion_load_by_atlas ───────────────────────────────────────────────────────


def _write_csv(path: Path, columns: list[str], rows: list[Row]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def _num(row: Row, key: str) -> float:
    v = row.get(key)
    return float(v) if isinstance(v, int | float) else 0.0


def _rows_from(result: dict[str, object]) -> list[Row]:
    raw = result.get("rows")
    if not isinstance(raw, list):
        return []
    return [dict(r) for r in raw if isinstance(r, dict)]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]


def lesion_load_by_atlas(
    lesion_mask_path: Path,
    atlas: AtlasName,
    inverse_transforms: list[str] | None = None,
    inverse_invert_flags: list[bool] | None = None,
    mask_in_template_space: bool = False,
    probability_threshold: float = 0.25,
    output_dir: Path | None = None,
    save_warped_atlas: bool = False,
) -> LesionLoadResult:
    """Report lesion volume per atlas region or white-matter tract (e.g. corticospinal tract).

    Load the ``atlas`` skill before calling this tool.

    Pure voxel counting: the atlas is warped into the lesion mask's space with ANTs
    (nearest-neighbour for label atlases, linear for probabilistic tract maps) and
    overlapped with the mask. Volumes use the mask's header voxel spacing, so they are
    the patient's native mm³. The mask itself is never resampled.

    How the atlas reaches the mask (exactly one of):
        * ``inverse_transforms`` (+ ``inverse_invert_flags``) from a
          ``register_to_template`` run of the image the mask was drawn on (template →
          native). If the mask lives on a different image than the one registered
          (e.g. FLAIR mask, T1w registered), first ``coregister`` the two and prepend
          the FLAIR↔T1w transform accordingly, or warp the mask onto the T1w grid.
        * ``mask_in_template_space=True`` when the mask is already in
          MNI152NLin2009cAsym (the atlas is then only resampled onto the mask grid).

    Voxels with a mask value > 0.5 count as lesion.

    Output columns (CSV ``{stem}_atlas-{atlas}_lesionload.csv``):
        Label atlas: ``region_volume_mm3``, ``lesion_in_region_mm3``,
        ``percent_region_lesioned``, ``percent_lesion_in_region``; plus an
        ``outside atlas`` row.
        Tract atlas: thresholded (tract = voxels with probability >=
        ``probability_threshold``) ``tract_volume_mm3``, ``lesion_in_tract_mm3``,
        ``percent_tract_lesioned``, ``percent_lesion_in_tract``; and
        probability-weighted (each voxel weighted by its probability)
        ``tract_weighted_volume_mm3``, ``lesion_in_tract_weighted_mm3``,
        ``percent_tract_lesioned_weighted``.

    Args:
        lesion_mask_path: Binary lesion mask NIfTI (.nii/.nii.gz).
        atlas: One of ``list_atlases()``: ``pandora_tractseg`` (72 tracts incl.
            CST_left/CST_right; default choice for tracts), ``hcp1065`` (64 tracts,
            second opinion), ``schaefer2018_100|200|400`` (cortical parcels).
        inverse_transforms: Template→native transform list from
            ``register_to_template`` (its ``inverse_transforms``).
        inverse_invert_flags: The matching ``inverse_invert_flags``. If omitted,
            ``.mat`` affines are inverted and warp fields are not (ANTs convention).
        mask_in_template_space: The mask is already in MNI152NLin2009cAsym.
        probability_threshold: Probability (0-1] at or above which a tract-map voxel
            belongs to the tract, for the thresholded columns. Default 0.25.
            Ignored for label atlases.
        output_dir: Where the CSV is written. Defaults to the mask's directory.
        save_warped_atlas: Also write the atlas as warped onto the mask grid (label
            map, or one map per tract in a folder) for visual QC.

    Returns:
        ``LesionLoadResult`` with the CSV path, the rows that overlap the lesion,
        the corticospinal-tract rows (tract atlases), and the atlas attribution.

    Raises:
        FileNotFoundError: If the mask or a transform file does not exist.
        ValueError: On a missing/contradictory space specification or bad threshold.
        RuntimeError: If antspyx is unavailable or ANTs fails.
    """
    _check_antspy()
    if not lesion_mask_path.exists():
        raise FileNotFoundError(f"Lesion mask not found: {lesion_mask_path}")
    if bool(inverse_transforms) == mask_in_template_space:
        raise ValueError(
            "Give exactly one of inverse_transforms (from register_to_template on the "
            "image the mask belongs to) or mask_in_template_space=True. If neither "
            "exists yet, run register_to_template on that image first."
        )
    transforms = list(inverse_transforms or [])
    missing = [t for t in transforms if not Path(t).exists()]
    if missing:
        raise FileNotFoundError(f"Transform file(s) not found: {', '.join(missing)}")
    if inverse_invert_flags is None:
        flags = [t.endswith(".mat") for t in transforms]
    else:
        flags = list(inverse_invert_flags)
        if len(flags) != len(transforms):
            raise ValueError("inverse_invert_flags must have one entry per transform")
    if not 0.0 < probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in (0, 1]")

    prepared = load_atlas(atlas)
    spec = prepared.spec
    out_dir = output_dir if output_dir is not None else lesion_mask_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nii_stem(lesion_mask_path)
    csv_path = out_dir / f"{stem}_atlas-{atlas}_lesionload.csv"

    warped_path: Path | None = None
    if save_warped_atlas:
        if spec.kind == "labels":
            warped_path = out_dir / f"{stem}_atlas-{atlas}_dseg.nii.gz"
        else:
            warped_path = out_dir / f"{stem}_atlas-{atlas}_probseg"
            warped_path.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = [
        {
            "index": e.index,
            "name": e.name,
            "path": str(e.path) if e.path is not None else "",
            "file_stem": nii_stem(e.path) if e.path is not None else "",
        }
        for e in prepared.entries
    ]
    payload: dict[str, object] = {
        "operation": "atlas_overlap",
        "lesion_path": str(lesion_mask_path),
        "kind": spec.kind,
        "dseg_path": str(prepared.dseg_path) if prepared.dseg_path else "",
        "entries": entries,
        "transforms": transforms,
        "invert_flags": flags if transforms else None,
        "threshold": probability_threshold,
        "save_dir": str(warped_path) if warped_path is not None else None,
    }
    interp = "nearest-neighbour" if spec.kind == "labels" else "linear"
    print(
        f"[medmcp-neuro-core] lesion_load_by_atlas: {atlas} → {lesion_mask_path.name} "
        f"({len(entries)} {'regions' if spec.kind == 'labels' else 'tracts'}) …",
        file=sys.stderr,
        flush=True,
    )
    run = _run_antspy(payload, _TIMEOUT)
    rows = _rows_from(run)
    voxel_mm3 = float(run.get("voxel_mm3", 0.0))  # pyright: ignore[reportArgumentType]
    lesion_mm3 = float(run.get("lesion_volume_mm3", 0.0))  # pyright: ignore[reportArgumentType]

    if spec.kind == "labels":
        _write_csv(csv_path, _LABEL_COLUMNS, rows)
        key, name_key = "lesion_in_region_mm3", "region"
        hit = [r for r in rows if _num(r, key) > 0]
    else:
        _write_csv(csv_path, _TRACT_COLUMNS, rows)
        key, name_key = "lesion_in_tract_weighted_mm3", "tract"
        hit = [r for r in rows if _num(r, key) > 0 or _num(r, "lesion_in_tract_mm3") > 0]
    hit.sort(key=lambda r: _num(r, key), reverse=True)
    cst = [r for r in rows if "corticospinal" in str(r.get(name_key, "")).lower()]

    if mask_in_template_space:
        method = f"atlas resampled onto the mask grid (identity, {interp}); mask in {_MNI_SPACE}"
    else:
        method = f"atlas warped template→mask space with ANTs ({interp}); mask not resampled"
    if spec.space != _MNI_SPACE:
        method += f"; atlas space {spec.space} used in MNI world coordinates as {_MNI_SPACE}"

    thr_text = (
        f"Tract = voxels with atlas probability >= {probability_threshold}. "
        "Weighted columns count each voxel by its probability. "
        if spec.kind == "probabilistic"
        else ""
    )
    result: LesionLoadResult = {
        "csv_path": str(csv_path),
        "lesion_mask_path": str(lesion_mask_path),
        "atlas": atlas,
        "atlas_kind": spec.kind,
        "atlas_title": spec.title,
        "method": method,
        "probability_threshold": probability_threshold if spec.kind == "probabilistic" else None,
        "lesion_volume_mm3": round(lesion_mm3, 4),
        "voxel_volume_mm3": voxel_mm3,
        "rows_with_lesion": hit[:_MAX_LISTED_ROWS],
        "corticospinal_rows": cst,
        "warped_atlas_path": str(warped_path) if warped_path is not None else None,
        "attribution": spec.attribution,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Header: '<atlas_title>' and total lesion volume <lesion_volume_mm3> mm³.\n"
            "If the user asked about the corticospinal tract, show corticospinal_rows "
            "first (left and right). Then a table of rows_with_lesion (already sorted, "
            "at most 20; the full table is at csv_path). For tracts show: tract, "
            "lesion_in_tract_mm3, percent_tract_lesioned, lesion_in_tract_weighted_mm3, "
            "percent_tract_lesioned_weighted. For regions show: region, "
            "lesion_in_region_mm3, percent_region_lesioned, percent_lesion_in_region.\n"
            f"{thr_text}Volumes are native mm³ from the mask header. A tract can share "
            "voxels with other tracts, so tract rows do not add up to the lesion volume.\n"
            f"State the method line: <method>.\n{_WORDING}\n"
            "Always end with the atlas attribution (the 'attribution' field, verbatim).\n"
            "NEXT ACTION: offer the CSV path, a second-opinion run with the other tract "
            "atlas, or save_warped_atlas=True to inspect the warped atlas over the scan."
        ),
    }
    return result


# ── lesion_frequency_map ───────────────────────────────────────────────────────


def _load(path: Path) -> tuple[Any, Any]:
    """Return (data array, affine) of a NIfTI via nibabel."""
    import nibabel as nib
    import numpy as np
    from nibabel.spatialimages import SpatialImage

    img = nib.load(str(path))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(img, SpatialImage):
        raise ValueError(f"{path} is not a spatial image")
    data: Any = np.asarray(img.dataobj)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    affine: Any = img.affine
    return data, affine


def _plot(freq_path: Path, template: Path, fig_path: Path, n: int) -> None:
    """Render the frequency map over the template (nilearn, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    from nilearn import plotting  # pyright: ignore[reportMissingTypeStubs]

    display = plotting.plot_stat_map(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        str(freq_path),
        bg_img=str(template),
        display_mode="z",
        cut_coords=8,
        threshold=1e-6,
        vmin=0.0,
        vmax=1.0,
        cmap="hot",
        symmetric_cbar=False,
        colorbar=True,
        title=f"Lesion frequency (n={n}), {_MNI_SPACE}",
    )
    display.savefig(str(fig_path), dpi=150)  # pyright: ignore[reportUnknownMemberType]
    display.close()  # pyright: ignore[reportUnknownMemberType]


def lesion_frequency_map(
    lesion_mask_paths: list[Path],
    output_dir: Path,
    forward_transforms_list: list[list[str]] | None = None,
    output_prefix: str = "cohort",
) -> LesionFrequencyResult:
    """Build a cohort lesion-frequency map in MNI152NLin2009cAsym and a figure.

    Load the ``atlas`` skill before calling this tool.

    Counting only: each voxel's value is (number of masks with a lesion there) / n.
    Voxels with a mask value > 0.5 count as lesion.

    Masks must be in, or be brought into, MNI152NLin2009cAsym:
        * already in template space → pass them alone. Masks on a different grid
          than the 1 mm template (e.g. 2 mm) are resampled onto it (identity,
          nearest-neighbour).
        * native space → pass ``forward_transforms_list`` (one ``forward_transforms``
          list from ``register_to_template`` per mask, same order); each mask is
          warped nearest-neighbour onto the template grid first.

    Output files in ``output_dir``:
        - ``{prefix}_space-MNI152NLin2009cAsym_lesionfreq.nii.gz`` — frequency (0-1).
        - ``{prefix}_space-MNI152NLin2009cAsym_lesioncount.nii.gz`` — counts.
        - ``{prefix}_lesionfreq.png`` — axial slices over the MNI template (nilearn).
        - ``{mask_stem}_space-MNI152NLin2009cAsym.nii.gz`` — each mask that had to be
          warped/resampled.

    Args:
        lesion_mask_paths: Binary lesion masks, one per subject.
        output_dir: Directory for the outputs.
        forward_transforms_list: Per-mask native→template transforms, or None when the
            masks are already in template space.
        output_prefix: Filename prefix for the map and figure. Default ``cohort``.

    Returns:
        ``LesionFrequencyResult`` with the map, count map and figure paths.

    Raises:
        FileNotFoundError: If a mask or transform is missing.
        ValueError: If no masks are given or the transform list length mismatches.
    """
    import nibabel as nib
    import numpy as np

    if not lesion_mask_paths:
        raise ValueError("lesion_mask_paths must not be empty")
    missing = [str(p) for p in lesion_mask_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Lesion mask(s) not found: {', '.join(missing)}")
    if forward_transforms_list is not None:
        if len(forward_transforms_list) != len(lesion_mask_paths):
            raise ValueError("forward_transforms_list needs one transform list per mask")
        gone = [t for ts in forward_transforms_list for t in ts if not Path(t).exists()]
        if gone:
            raise FileNotFoundError(f"Transform file(s) not found: {', '.join(gone)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    template = get_mni152_1mm(skull_stripped=True)
    tmpl_data, tmpl_aff = _load(template)
    tmpl_shape: tuple[int, ...] = tuple(tmpl_data.shape)

    # Decide per mask whether it is usable as is or must go through ANTs.
    final_paths: list[Path] = []
    items: list[dict[str, object]] = []
    for i, p in enumerate(lesion_mask_paths):
        transforms = forward_transforms_list[i] if forward_transforms_list is not None else []
        if not transforms:
            data, aff = _load(p)
            if is_same_grid(tuple(data.shape), aff, tmpl_shape, tmpl_aff):
                final_paths.append(p)
                continue
        out = output_dir / f"{nii_stem(p)}_space-{_MNI_SPACE}.nii.gz"
        items.append({"input_path": str(p), "out_path": str(out), "transforms": transforms})
        final_paths.append(out)
    if items:
        _check_antspy()
        print(
            f"[medmcp-neuro-core] lesion_frequency_map: bringing {len(items)} mask(s) "
            f"onto the {_MNI_SPACE} grid …",
            file=sys.stderr,
            flush=True,
        )
        _run_antspy(
            {"operation": "masks_to_template", "reference_path": str(template), "items": items},
            _TIMEOUT,
        )

    masks = [lesion_voxels(_load(p)[0]) for p in final_paths]
    count, freq = frequency(masks)

    freq_path = output_dir / f"{output_prefix}_space-{_MNI_SPACE}_lesionfreq.nii.gz"
    count_path = output_dir / f"{output_prefix}_space-{_MNI_SPACE}_lesioncount.nii.gz"
    fig_path = output_dir / f"{output_prefix}_lesionfreq.png"
    for arr, path in ((freq, freq_path), (count.astype(np.int16), count_path)):
        img = nib.Nifti1Image(arr, tmpl_aff)
        img.header.set_xyzt_units("mm")  # pyright: ignore[reportUnknownMemberType]
        nib.save(img, str(path))  # pyright: ignore[reportUnknownMemberType]
    _plot(freq_path, template, fig_path, len(masks))

    n = len(masks)
    result: LesionFrequencyResult = {
        "frequency_map_path": str(freq_path),
        "count_map_path": str(count_path),
        "figure_path": str(fig_path),
        "n_masks": n,
        "max_frequency": float(freq.max()),
        "voxels_with_any_lesion": int((count > 0).sum()),
        "template_path": str(template),
        "template_space_masks": [str(p) for p in final_paths],
        "attribution": (
            "Background: MNI ICBM152 2009c nonlinear asymmetric template (Fonov et al., "
            "NeuroImage 2011), via TemplateFlow. Figure: nilearn (BSD-3)."
        ),
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            f"Report: n = {n} masks; maximum voxel frequency <max_frequency> "
            "(fraction of subjects); <voxels_with_any_lesion> voxels lesioned in at least "
            "one subject. Show the figure at figure_path and give frequency_map_path.\n"
            "This is a descriptive count map (voxel count / n), not a statistical test — "
            "do not call areas 'significant'. For per-tract numbers run "
            "lesion_load_by_atlas on each mask.\n"
            "End with the attribution field verbatim."
        ),
    }
    return result
