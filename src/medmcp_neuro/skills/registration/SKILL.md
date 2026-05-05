---
name: registration
description: Workflow for registering NIfTI images to a standard template or within-subject coregistration using ANTsPy
---

# Registration workflow

## Steps

1. **Ask the user which transform type to use** — before calling any registration tool,
   present the available transform types (listed in each tool's docstring) and ask the
   user to confirm their choice.
2. **Run registration** — call `register_to_template` or `coregister` with the confirmed
   `transform_type`.
3. **Apply to additional images** — use `apply_transform` with the transform lists from
   the registration result to warp masks, parcellations, or lesion maps into the same
   space without re-running registration:
   - native → template space: pass `forward_transforms` directly
   - template → native space: pass `inverse_transforms` with `invert_flags=inverse_invert_flags`

## Gotchas

- **Always ask the user before calling** — never pick a transform type without asking.
- For DWI/fMRI b0-to-T1w coregistration, suggest `synquick` to correct EPI distortion.
- For label images (brain masks, atlas parcellations), use `interpolation="NearestNeighbor"`
  in `apply_transform` to avoid interpolation artefacts on integer labels.
- When applying transforms where the reference is the MNI template, pass
  `output_space="MNI152NLin2009cAsym"` explicitly so the BIDS output filename is correct.
- Transforms from `register_to_template` and `coregister` can be composed: concatenate
  the transform lists in application order when calling `apply_transform`.
