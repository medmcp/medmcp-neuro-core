---
name: segmentation
description: Workflow for whole-brain segmentation and structure volumetry (e.g. thalamic volume) using FastSurfer
---

# Brain segmentation & volumetry workflow

`segment_brain` runs FastSurfer's deep-learning segmentation (FastSurferVINN,
seg-only) to label cortical and subcortical structures and produce a per-structure
**volume CSV** — no FreeSurfer license required, GPU-accelerated.

## When to use

- The user wants a **volume** for a specific structure (e.g. *thalamic volume*,
  hippocampus, ventricles) or a whole-brain label map.
- The user wants subcortical/cortical morphometry on a T1w scan.

## Steps

1. **Confirm the input is T1w.** FastSurfer is trained on T1-weighted MRI (MPRAGE
   is the canonical sequence, but any T1w — MP2RAGE, SPGR, … — is fine; do **not**
   restrict to MPRAGE). `segment_brain` also checks this itself: a filename naming a
   non-T1w contrast (`FLAIR`, `T2w`, `dwi`, …) comes back in the `warnings` field.
   Note the header alone *cannot* confirm contrast (NIfTI has no sequence field), so
   this is a filename heuristic — relay the warning and offer to coregister a T1w
   instead. Do **not** silently run on non-T1w.
2. **Do not skull-strip first.** FastSurfer accepts full-head *and* skull-stripped
   input and does its own brain extraction. Running `skull_strip` beforehand is
   unnecessary (and harmless, but don't add the step on the user's behalf).
3. **Run `segment_brain`** with `device="auto"` (uses the GPU when available; falls
   back to CPU, which is much slower — tell the user if it resolves to CPU). If the
   image is strongly anisotropic or thick-slice (e.g. 1×1×5 mm 2D clinical scans),
   the call raises rather than producing a garbage segmentation; relay the error and
   only re-run with `force=True` if the user explicitly accepts degraded quality.
4. **Report the requested structure.** When the user asked for a specific volume,
   read the CSV at `volumes_path` and report the matching row(s) — e.g. for the
   thalamus, the `Left-Thalamus` and `Right-Thalamus` rows (values in **mm³**).
   Match the structure names verbatim (see below). Offer the full CSV for other
   structures rather than dumping every row.

## Getting a specific structure's volume

The volumes CSV has columns `structure,volume_mm3`. The `structure` column holds
FastSurfer's **exact StructNames** (case-sensitive, hyphenated): non-cortical
structures follow the aseg convention (`Left-Thalamus`, `Right-Hippocampus`, `CSF`,
`Brain-Stem`, …) and cortical parcels are `ctx-lh-<stem>` / `ctx-rh-<stem>`
(Desikan-Killiany-Tourville, e.g. `ctx-lh-superiorfrontal`). Match them verbatim — a
lowercased/spaced guess like `left thalamus` will not be found. Call
`list_brain_segmentation_labels()` to see the exact names before searching the CSV.

## Gotchas

- **Comparing volumes across subjects** — raw mm³ volumes scale with head size, so
  for cross-subject or cohort comparisons normalise by head size. The CSV's final
  `BrainSegVol` row (total brain-segmentation volume, mm³) is the normaliser to use:
  report each structure as a fraction of `BrainSegVol`. Note this is brain-segmentation
  volume, **not** eTIV/ICV — true eTIV needs a Talairach registration that requires a
  FreeSurfer license, which this license-free seg-only pipeline deliberately avoids;
  `BrainSegVol` is the license-free equivalent for head-size normalisation.
- **Output is a FreeSurfer `.mgz` label map** (`*_dseg.mgz`) — the workspace viewer
  renders MGZ natively, and you can overlay it on the input by dragging it onto the
  image. To warp it into template space, use `apply_transform` with
  `interpolation="NearestNeighbor"` (it's an integer label map).
- **Runtime** — segmentation is substantially faster on GPU than on CPU. The
  result reports which `device` was used.
- **T1w only** — FastSurfer is not contrast-agnostic. For heterogeneous/clinical
  multi-contrast cohorts, flag that a contrast-robust method may be more appropriate.
- **Resolution** — FastSurfer conforms to ~1 mm internally and handles ~0.7–1.0 mm
  native, so ordinary 1 mm variation is fine. `segment_brain` warns when resolution
  drifts outside ~0.7–1.3 mm or is mildly anisotropic, and **refuses** strongly
  anisotropic / thick-slice data (≥2 mm voxels or ≥2× anisotropy) unless `force=True`,
  because conforming such scans yields a meaningless segmentation. Always read back
  the `warnings` field and pass it on to the user.
