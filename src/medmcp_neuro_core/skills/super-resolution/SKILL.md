---
name: super-resolution
description: Workflow for super-resolving anisotropic, thick-slice (2D-acquired) MRI to isotropic resolution with ECLARE or SMORE
---

# Super-resolution workflow

`super_resolve` turns a thick-slice MRI volume (e.g. 0.9 × 0.9 × 5 mm) into an
isotropic one at the in-plane resolution (0.9 × 0.9 × 0.9 mm) using **ECLARE** or
**SMORE**. Both are **self-supervised**: there are no pretrained weights and no
training data. Every call trains a fresh network on the image itself (from its
high-resolution in-plane slices), then applies it to the through-plane axis. This
makes them contrast- and scanner-agnostic (T1w, T2w, FLAIR, PD, …), and also slow.

## When to use

- The scan was acquired in **2D with thick slices**: one voxel dimension clearly
  coarser than the other two (typical clinical T2w/FLAIR, e.g. 1 × 1 × 4–6 mm).
- The user wants to **view** it in other planes without staircase artefacts, or to
  feed it to a tool that needs near-isotropic input. `segment_brain` refuses ≥ 2 mm or
  ≥ 2× anisotropic input; super-resolving first is one way to get there (with the
  caveat below).

Do **not** use it on 3D/isotropic acquisitions. The tool detects (near-)isotropic input
(anisotropy < 1.1×), skips it, and writes nothing. Do not use it on 4-D series either:
it handles one 3-D volume at a time.

## Choosing the method

| | `eclare` (default) | `smore` |
|---|---|---|
| What | ECLARE (2026), the successor to SMORE from the same group | SMORE v4 (2021/2023) |
| Slice profile | Estimated from the image (ESPRESO) | RF-pulse profile, thickness = slice spacing |
| In-plane | May be anisotropic | **Must be isotropic** (else error) |
| Runtime on a GPU (default settings) | ~30 min per brain volume | ~1.5 h per brain volume |

Recommend **ECLARE** unless the user asks for SMORE or wants to compare the two.
Measured in this stack's container on an NVIDIA GB10 (DGX Spark) with a 193 × 229 × 48
(1 × 1 × 4 mm) brain volume:

- ECLARE trained at ~550 patches/s. The default 1,000,000 patches plus slice-profile
  estimation and inference took **32 min**.
- SMORE trained at ~170 patches/s. A 100,000-patch run took 10 min, so the default
  832,000 patches comes to **~85 min**.

Training time scales with the number of patches, not the volume size, so these
numbers carry over to other brain volumes. Slower GPUs take longer.

## Steps

1. **Check the geometry.** Confirm the input has one thick through-plane axis. If the
   user doesn't know, just call the tool: it reads the header and skips isotropic
   input.
2. **Set expectations before running.** Tell the user it trains a network on their
   image and takes roughly 30 min (ECLARE) or 1.5 h (SMORE) on the GPU, and ask them
   to confirm. Do not start a run without saying so.
3. **Slice thickness (optional).** If the user knows the *acquired* slice thickness
   and it differs from the spacing in the header (slice gap, e.g. 5 mm slices with a
   1 mm gap), pass `slice_thickness_mm`. Otherwise omit it: ECLARE estimates the
   profile itself and SMORE assumes thickness = spacing.
4. **Run `super_resolve`** with `device="auto"`. It needs a CUDA GPU. `"auto"` raises
   instead of silently falling back to CPU, because CPU training takes many hours and
   can outlive the 2-hour tool timeout. Only pass `device="cpu"` if the user explicitly
   accepts that. Apple MPS is not supported upstream.
5. **Report.** Relay any `warnings` first, then the input and output voxel sizes and
   shapes, the device, and the runtime, as the result's `_render` asks.

`n_patches` lowers the number of training patches for a quick, lower-quality preview
(e.g. 100,000 patches: ~4 min for ECLARE, ~10 min for SMORE). Use it only when the user
asks for speed over quality, and say the result is a preview.

## Output

- `{stem}_eclare.nii.gz` or `{stem}_smore.nii.gz` in `output_dir` (default: next to
  the input), e.g. `sub-01_FLAIR.nii.gz` → `sub-01_FLAIR_eclare.nii.gz`. A `.nii` input
  gives a `.nii` output.
- The through-plane spacing becomes the finest in-plane spacing. In-plane size and
  spacing are unchanged. The affine is updated, so the output stays aligned with
  the input in world space and overlays on it directly.
- Intermediates (training weights, ECLARE's slice-profile estimate, logs) are
  discarded. Only the super-resolved volume is kept.

## Gotchas

- **Synthesized detail.** The network fills in the through-plane detail. It is learned
  from the same image's in-plane content and is usually faithful, but it is not
  measured data. It is good for viewing and for isotropic downstream processing. Treat
  quantitative results (e.g. volumetry after `segment_brain`) on super-resolved data as
  research-grade, and say so.
- **Stochastic.** Training is seeded, but GPU kernels and the slice-profile estimate
  can make two runs differ slightly.
- **Strongly anisotropic in-plane** (e.g. 0.5 × 1.0 × 5 mm): SMORE refuses. ECLARE runs
  and super-resolves the through-plane axis to the finer in-plane spacing. The in-plane
  axes are left as they are, and the tool warns.
- **Two equally coarse axes** (e.g. 1 × 4 × 4 mm) are not a 2D acquisition. Both methods
  refuse.
- **Timeouts.** One call has a 2-hour budget. SMORE at its default setting (~85 min on
  a GB10) gets close to it, and a slower GPU can run past it. Prefer ECLARE, or lower
  `n_patches` for SMORE.
