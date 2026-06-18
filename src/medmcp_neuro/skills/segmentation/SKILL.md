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

1. **Confirm the input is T1w.** FastSurfer is trained on T1-weighted MRI. If the
   image looks like another contrast (FLAIR/T2w/DWI — e.g. the filename contains
   `FLAIR`, `T2w`, `dwi`), warn the user that results may be unreliable and offer to
   proceed anyway or coregister a T1w instead. Do **not** silently run on non-T1w.
2. **Do not skull-strip first.** FastSurfer accepts full-head *and* skull-stripped
   input and does its own brain extraction. Running `skull_strip` beforehand is
   unnecessary (and harmless, but don't add the step on the user's behalf).
3. **Run `segment_brain`** with `device="auto"` (uses the GPU when available; falls
   back to CPU, which is much slower — tell the user if it resolves to CPU).
4. **Report the requested structure.** When the user asked for a specific volume,
   read the CSV at `volumes_path` and report the matching row(s) — e.g. for the
   thalamus, the `left thalamus` and `right thalamus` rows (values in **mm³**).
   Offer the full CSV for other structures rather than dumping every row.

## Getting a specific structure's volume

The volumes CSV has columns `structure,volume_mm3`. Subcortical names follow the
aseg convention (`left thalamus`, `right hippocampus`, …); cortical parcels are
`ctx-lh-<name>` / `ctx-rh-<name>` (Desikan-Killiany-Tourville). Call
`list_brain_segmentation_labels()` to see the exact names before searching the CSV.

## Gotchas

- **Comparing volumes across subjects** — raw mm³ volumes scale with head size. If
  the user is comparing subjects or cohorts, suggest normalising by `total
  intracranial` (ICV), which is included in the CSV.
- **Output is a FreeSurfer `.mgz` label map** (`*_dseg.mgz`) — the workspace viewer
  renders MGZ natively, and you can overlay it on the input by dragging it onto the
  image. To warp it into template space, use `apply_transform` with
  `interpolation="NearestNeighbor"` (it's an integer label map).
- **Runtime** — on GPU a subject is minutes; on CPU it can be much longer. The
  result reports which `device` was used.
- **T1w only** — FastSurfer is not contrast-agnostic. For heterogeneous/clinical
  multi-contrast cohorts, flag that a contrast-robust method may be more appropriate.
