---
name: lesion-segmentation
description: Workflow for segmenting white-matter (MS) lesions from paired T1w + FLAIR images using LST-AI, including the recommended pre-skull-stripping path and warping lesion maps into template space
---

# Lesion segmentation workflow

`segment_ms_lesions` runs LST-AI, which needs **both** a T1w and a FLAIR image of the
same subject. LST-AI is run out of process via its `lst` console script, which
must be installed in its own virtualenv (it is not a dependency of this package —
it pins an old HD-BET that would conflict). Point `$MEDMCP_LST_AI_BIN` at that
`lst`, or have it on PATH.

## Steps

1. **Confirm both inputs.** You need a T1w and a FLAIR for the same session.
   If only one contrast is available, stop and tell the user LST-AI cannot run.

2. **Decide on skull stripping** — this matters for both quality and reliability:

   - **Recommended:** run this package's `skull_strip` tool on **both** the T1w and
     the FLAIR first, then call `segment_ms_lesions(..., skull_stripped=True)`. This
     reuses the HD-BET already in this stack and skips LST-AI's own pinned HD-BET.
   - If the user prefers, pass raw (non-stripped) images and let LST-AI strip them
     internally (`skull_stripped=False`, the default).

   Always tell the user which path you are taking and why. Never set
   `skull_stripped=True` unless **both** inputs are actually brain-extracted —
   doing so on full-head images silently corrupts the result.

3. **Ask which compute device to use** — present the options and wait:
   - `"cpu"` — always available, slow (often more than an hour).
   - `"cuda"` — NVIDIA GPU (uses `gpu_id`, default 0), much faster.
   LST-AI has no Apple-Silicon (MPS) path; do not offer `mps`.

4. **Run `segment_ms_lesions`** with the confirmed `device` and `skull_stripped` flag.
   Report the lesion mask path (and the region-annotated map, if produced).

5. **Optional — warp the lesion map into template space.** If a T1w→template
   registration already exists for this subject (from `register_to_template`), carry
   the lesion mask into MNI with `apply_transform`:
   - pass the registration's `forward_transforms`,
   - use `interpolation="NearestNeighbor"` (the mask is a label image),
   - set `output_space="MNI152NLin2009cAsym"` so the BIDS filename is correct.
   See the `registration` skill for the transform-composition details.

## Gotchas

- LST-AI requires same-subject T1w **and** FLAIR; it is not a single-contrast tool.
- If `lst` is not installed the tool raises with venv install guidance — relay
  that to the user rather than guessing at a fix.
- The lesion mask is a binary label image — always use `NearestNeighbor`
  interpolation when resampling or warping it.
- The `greedy` binary is downloaded on first use (cached under
  `~/.medmcp_neuro/bin/`); the first run may pause while that happens.
- On CPU the run can take well over an hour. Set expectations before starting.
