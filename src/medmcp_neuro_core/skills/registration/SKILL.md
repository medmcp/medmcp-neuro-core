---
name: registration
description: Workflow for registering NIfTI images to a standard template or within-subject coregistration using ANTsPy
---

# Registration workflow

## Steps

1. **Confirm the template** — before calling `register_to_template`, infer both the template type and skull-stripped variant, then confirm them with the user in one message:

   **i) Template type (`template_name`)**

   | BIDS template name | Use case |
   |---|---|
   | `MNI152NLin2009cAsym` | Default (general neuroimaging, multiple sclerosis analysis) |
   | `MNI152NLin6Asym` | Brain tumor workflows (when required by a glioma, metastasis, or meningioma tool/skill) |

   - If a downstream tool/skill dictates the space (e.g. a brain tumor workflow requiring `MNI152NLin6Asym`, or `lesion_load_by_atlas` / `lesion_frequency_map` using `MNI152NLin2009cAsym`), select it automatically.
   - Otherwise, infer `MNI152NLin2009cAsym` as the default and offer `MNI152NLin6Asym` as the alternative in the confirmation below (if ambiguous, ask with brief domain-relevant suggestions).

   **ii) Skull-stripped variant (`skull_stripped`)** — determine whether to use the skull-stripped (brain-only) or full-head MNI template:

   - If the image was produced by `skull_strip` in this session **or** the filename
     contains `brain` (e.g. `_desc-brain_`, `_brain.nii.gz`) → infer skull-stripped.
   - Otherwise → infer full-head (not skull-stripped).

   Always tell the user which template and variant will be used and offer the alternatives:
   > "Your image appears to be skull-stripped, so I'll register to the **brain-only
   > `MNI152NLin2009cAsym`** template (`desc-brain`, default). Reply if you prefer
   > **full-head** or **`MNI152NLin6Asym`** (brain-tumor/FSL space)."
   > — or —
   > "Your image appears to be a full-head scan, so I'll register to the **full-head
   > `MNI152NLin2009cAsym`** template (default). Reply if you prefer **brain-only**
   > or **`MNI152NLin6Asym`** (brain-tumor/FSL space)."

   Wait for the user to confirm or correct before proceeding.
   Using the wrong template or variant degrades registration quality.

2. **Ask the user which transform type to use** — before calling any registration tool,
   present all available transform types with their descriptions (as listed in the tool's
   docstring) and wait for the user to choose. Never pick a type yourself.
3. **Run registration** — call `register_to_template` or `coregister` with the confirmed
   `transform_type`, `skull_stripped` flag, and `template_name`.
4. **Apply to additional images** — use `apply_transform` with the transform lists from
   the registration result to warp images into the same space without re-running registration:
   - native → template space: pass `forward_transforms` directly
   - template → native space: pass `inverse_transforms` with `invert_flags=inverse_invert_flags`

### Getting other MRI contrasts into template space

When a T1w has been registered to the template and the user wants an additional contrast
(FLAIR, T2w, DWI b0, etc.) in template space, use a two-step approach:

1. `coregister` the contrast to the T1w. Present all transform types with their descriptions
   and let the user choose. Note that for DWI/fMRI b0, EPI distortion is a factor the user
   may want to consider when choosing.
2. `apply_transform` with the T1w→template `forward_transforms` to carry the
   coregistered contrast into template space, using the registered T1w as the reference.

Do **not** re-register the contrast directly to the template — this wastes time, produces
inconsistent deformations, and makes native-space analysis harder.

## Gotchas

- For label images (brain masks, atlas parcellations), use `interpolation="NearestNeighbor"`
  in `apply_transform` to avoid interpolation artifacts on integer labels.
- When applying transforms where the reference is the MNI template, pass
  `output_space="MNI152NLin2009cAsym"` explicitly so the BIDS output filename is correct.
- Transforms from `register_to_template` and `coregister` can be composed: concatenate
  the transform lists in application order when calling `apply_transform`.
