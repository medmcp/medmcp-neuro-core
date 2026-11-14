---
name: atlas
description: Workflow for lesion load per brain region / white-matter tract (e.g. corticospinal tract) with a bundled atlas, and cohort lesion-frequency heatmaps
---

# Atlas lesion-load workflow

`lesion_load_by_atlas` counts how much of a lesion mask overlaps each region or
tract of a population atlas. `lesion_frequency_map` counts, voxel by voxel, how many
subjects of a cohort have a lesion there. Both are **counting only** — no statistics,
no model, no prediction.

## When to use

- "How much of the lesion is in the corticospinal tract?", "lesion load per tract",
  "which tracts does the lesion overlap?" → `lesion_load_by_atlas` with a tract atlas.
- "Which cortical regions are lesioned?" → `lesion_load_by_atlas` with a Schaefer atlas.
- "Where do lesions cluster across these patients?", "lesion heatmap" →
  `lesion_frequency_map`.
- Not for: lesion segmentation itself (needs a mask first), structure volumes
  (`segment_brain`), or any group comparison / significance test.

## Choosing the atlas (`list_atlases`)

| Atlas | Use for | Notes |
|---|---|---|
| `pandora_tractseg` | tracts — **default**, incl. `Corticospinal tract left/right` | 72 probabilistic bundles, MNI152NLin2009cAsym, CC BY 4.0 |
| `hcp1065` | tracts — second opinion | 64 tracts (`Corticospinal Tract left/right`, also corticobulbar, reticulospinal); ICBM 2009a space used as 2009c (small approximation) |
| `schaefer2018_100/200/400` | cortical parcels | cortex only — no white matter, no deep grey matter |

Offer the second tract atlas when the answer matters: two independently built CST
maps agreeing is more reassuring than one.

## Steps — single patient

1. **Have a lesion mask and know what image it was drawn on.** (Often a FLAIR.)
2. **Get template→native transforms** for that image, unless the mask is already in
   MNI152NLin2009cAsym (then pass `mask_in_template_space=True` and skip to 3):
   - Mask drawn on the T1w: follow the `registration` skill to run
     `register_to_template` on the T1w (the user chooses the transform type; a
     deformable `syn` type is appropriate for tract overlap). Use its
     `inverse_transforms` and `inverse_invert_flags`.
   - Mask drawn on the FLAIR (or another contrast): `coregister` the FLAIR to the
     T1w (`fixed_path`=T1w, `moving_paths`=[FLAIR]), `register_to_template` the T1w,
     then chain **FLAIR-coregistration inverse first, template inverse second**:
     `inverse_transforms = coreg.inverse_transforms_list[0] + reg.inverse_transforms`,
     `inverse_invert_flags = coreg.inverse_invert_flags_list[0] + reg.inverse_invert_flags`.
   - Registering the FLAIR straight to the T1w MNI template also works but is less
     accurate (cross-contrast); prefer the chain.
3. **Call `lesion_load_by_atlas`** with the mask, the atlas, the transforms (or
   `mask_in_template_space=True`) and `probability_threshold` (default 0.25 — say so;
   change it only if the user asks). The atlas is warped onto the mask — the mask is
   never resampled, so volumes are the patient's native mm³.
4. **Report** per the `_render` rules: CST rows first if asked, then the overlapping
   tracts/regions, the method line, and the attribution string verbatim.
5. Offer `save_warped_atlas=True` so the user can check the warped atlas over the
   scan — a misregistration shows up immediately.

## Steps — cohort heatmap

1. Collect the lesion masks. If they are already in MNI152NLin2009cAsym, pass them
   alone. If native, register each subject (as above) and pass
   `forward_transforms_list` = one `forward_transforms` per mask, same order (for a
   FLAIR mask: `reg.forward_transforms + coreg.forward_transforms_list[0]` — template
   warp first, coregistration second).
2. Call `lesion_frequency_map`. Show the figure; give the NIfTI path.

## How to word results

- Say **"lesion overlap with the Pandora CST map"**, "X mm³ of the lesion lies within
  the atlas's left corticospinal tract (probability ≥ 0.25)", "Y % of the tract map".
- Do **not** say the CST "is damaged/disrupted/transected", do not predict motor
  deficits or outcome, and do not call heatmap areas "significant". The atlas is
  where a tract usually runs in healthy adults, not this patient's tract; accuracy
  depends on registration and on the lesion mask.
- Tracts overlap each other: tract rows do not sum to the total lesion volume.
- Thresholded vs weighted: thresholded treats the tract as a binary region
  (probability ≥ threshold); weighted counts each voxel by how many atlas subjects
  had the tract there. Report both when they differ markedly.
- For heatmaps: "fraction of the n patients with a lesion at this voxel".

## Gotchas

- Voxels with mask value > 0.5 count as lesion; a multi-label mask is treated as one
  lesion class.
- Masks with thick slices (e.g. 1×1×5 mm) still give correct native volumes, but the
  tract overlap is coarse.
- `inverse_invert_flags` omitted → `.mat` affines are inverted and warp fields are
  not (the ANTs convention `register_to_template` follows).
- A large `outside atlas` share for a Schaefer atlas is expected for white-matter
  lesions — Schaefer covers cortex only.
