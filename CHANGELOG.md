# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `register_to_template` / `coregister` with `synquick` failed with "SyNQuick does not exist" (antspyx
  0.6.3 has no such name); it now runs ANTsPy's `antsRegistrationSyNQuick[s]`.
- A failing ANTs subprocess now reports its own error message instead of only the exit code.

### Added

- `average_repeats`: averages repeated scans of one session (e.g. mpr-1..mpr-4, run-01/02) into one image, with a check that the scans agree and a warning for a moved scan. Before, there was no tool for this.
- FreeSurfer 8.2.0 in the image (`/opt/freesurfer`), built in its own Dockerfile stages: amd64 from the official Ubuntu 24 package (SHA-256 pinned), arm64 from the v8.2.0 source with ITK 5.4.0 (no official arm64 build exists); both pruned to what recon-all needs and given `talairach_avi`. A modified FreeSurfer, marked as such (`/opt/freesurfer/MEDMCP-MODIFIED.txt`, version string, image label, NOTICE); the licence text ships with MGH's preface. FreeSurfer's environment is given only to the processes that run it. No licence key is ever part of the image.
- `run_recon_all`: validates one T1 scan and the user's FreeSurfer licence (by path, never read), then runs `recon-all -all -no-v8 -parallel -openmp <threads>` as a background job (about an hour). Without a licence nothing starts and the result explains the free registration.
- `recon_all_status`: progress of a recon-all or FastSurfer-surface run, or a list of all runs, from files in the workspace only.
- `freesurfer_stats`: subcortical volumes, thickness/area/volume per Desikan-Killiany region and the global measures (eTIV, mean thickness, ...) as CSV, verbatim from FreeSurfer's stats files, plus a QC image.
- `segment_brain(surfaces=True)`: FastSurfer's surface pipeline (recon-surf: surfaces, cortical thickness, eTIV) on top of the segmentation, as a background job; needs the FreeSurfer licence (without one the segmentation still runs and the result says `licence_required`). The FastSurfer corpus-callosum checkpoints are now baked, as recon-surf requires them.
- Background jobs: the image label declares `medmcp-neuro-core-job call` for `run_recon_all` and `segment_brain`; a core that runs stack jobs starts them in their own container, an older core gets a detached process in the stack container.
- The `segmentation` skill decides between the FastSurfer segmentation, FastSurfer surfaces and FreeSurfer recon-all (time, licence, how to follow a run).
- `lesion_load_by_atlas`: lesion volume per atlas region or white-matter tract (e.g. the corticospinal tract) by voxel counting. The atlas is warped into the lesion mask's space with ANTs (nearest-neighbour for label atlases, linear for probabilistic tract maps), so the mask is never resampled and volumes are native mm³ from its header. Tract atlases report thresholded (`probability_threshold`, default 0.25) and probability-weighted volumes and percentages; writes `<stem>_atlas-<atlas>_lesionload.csv` and returns the atlas attribution.
- `lesion_frequency_map`: cohort lesion-frequency map (voxel count / n) in MNI152NLin2009cAsym plus a nilearn figure over the template; masks in template space, or native masks with their `forward_transforms`.
- `list_atlases` and the bundled atlases, baked into the image from pinned, SHA-256-checked sources with attribution: Pandora TractSeg set (CC BY 4.0), HCP1065 (CC BY-SA 4.0, unmodified), Schaefer 2018 100/200/400 parcels (MIT). New `atlas` skill.
- `nilearn[plotting]` dependency (BSD-3) for the frequency-map figure.
- `NOTICE` is now copied into the container image.
- `segment_brain` writes `<stem>_labels.csv` (`label,structure`) beside the label map, so the workspace viewer names the structures in its legend.
- `super_resolve` tool: through-plane super-resolution of anisotropic, thick-slice MRI to isotropic, with `method="eclare"` (ECLARE 1.0.1, default) or `method="smore"` (SMORE v4.0.5). Both are self-supervised (a network is trained on each image, so there are no weights to bake) and run as their upstream CLIs in a subprocess. They need a CUDA GPU: `device="auto"` refuses to fall back to CPU, and there is no MPS. (Near-)isotropic input is skipped with a warning. The result reports input/output voxel sizes and shapes, the device, and the runtime. Output: `{stem}_eclare.nii.gz` / `{stem}_smore.nii.gz`.
- `super-resolution` SKILL.md: when super-resolution applies, ECLARE vs SMORE, runtime expectations, slice thickness, and output naming.

### Changed

- The image's `ENTRYPOINT` is `tini --` with `medmcp-neuro-core` as the default `CMD`, so the same image runs the job entrypoint.
- Dependencies: `eclare` (PyPI) and `smore` (git, gitlab.com/iacl/smore tag v4.0.5) added. `[tool.uv] override-dependencies` relaxes three stale upstream pins that cannot install on Python 3.12 or would downgrade the stack. ECLARE's `scipy==1.11.4` and degrade's `numpy==1.23.5` become `>=` floors. degrade's `sigpy` is dropped in favour of `sigpy-lite` (the same code with numba optional), which ECLARE already brings and which provides the `sigpy` module degrade imports. The locked numpy/scipy are unchanged. The container build imports both CLIs to verify them.

### Fixed

- Every tool refuses a truncated or corrupt input image instead of processing it; `skull_strip` returned an empty brain from a truncated `.nii.gz`.

## [0.1.0]

First public release. `medmcp-neuro-core` is the neuroimaging tool stack for
MedMCP — an MCP server exposing brain extraction, registration, and whole-brain
segmentation as tools an agent calls by name. Model weights and the template are
baked into the container image, so every tool runs with networking denied.

**Not licensed for clinical use.**

### Added

- `skull_strip` persistent HD-BET worker: loads torch + CUDA + the model once and reuses it across calls (opt-in via `MEDMCP_HDBET_PERSIST`, or once pre-loaded by the new `warmup` tool), reclaiming the per-call model-load cost on warm calls. Falls back to the per-call subprocess if the worker is unavailable, so behaviour is unchanged by default.
- `warmup` tool to pre-load the HD-BET model so the first `skull_strip` is already warm (the hook the workspace pre-warm pool calls on activation).
- Shared device convention for GPU-capable tools: `device="auto"` (the default) resolves to cuda > mps > cpu via `resolve_device()` in `_neuro.py`, and the resolved device is reported in the result (with a CPU-fallback note). `skull_strip`, `warmup`, and `segment_brain` all default to `auto`; the CPU-only ANTs registration tools take no device parameter.
- Container image: `Dockerfile` (`FROM medmcp-base`; torch pinned to the cu128 build, version `2.7.1`, so it runs on any host driver >= R570; baked HD-BET weights + the FastSurferVINN checkpoint; `segment_brain` runs in-container via FastSurfer) + `.dockerignore`; `org.medmcp.stack` label for one-click install; `.devcontainer`; CI publishes to the private `ghcr.io/medmcp/neuro-core`. Image-size optimized: HD-BET, antspyx, and FastSurfer share one torch/CUDA venv (torch pinned to FastSurfer's `2.7.1` so the ~8 GB CUDA stack isn't installed twice), only the asegdkt checkpoint is baked (`--vinn`, not `--all`), and bytecode/static-archive cruft is stripped in-layer.

- `segment_brain` and `list_brain_segmentation_labels` tools backed by FastSurfer (FastSurferVINN, seg-only — **no FreeSurfer license required**); whole-brain segmentation of 95 cortical/subcortical classes (33 subcortical/global + 31 DKT cortical parcels per hemisphere). The per-structure volume CSV uses FastSurfer's exact StructNames, omits the always-empty corpus-callosum rows (CC is only populated by the surface pipeline, which seg-only does not run), and appends a `BrainSegVol` row (license-free total brain-segmentation volume) for head-size normalization. T1w only; input contrast and resolution are sanity-checked (non-fatal warnings, with strongly anisotropic/thick-slice data blocked unless `force=True`)
- `register_to_template`, `coregister`, and `apply_transform` tools backed by ANTsPy
- MNI152NLin2009cAsym 1 mm template baked into the image at build time, both
  variants (whole-head and skull-stripped), so `register_to_template` resolves it
  from the local cache and runs with networking denied; the brain-extracted variant
  is selected via the `skull_stripped` parameter. Running from source it is
  downloaded from templateflow on first use and cached.
- `registration` SKILL.md with step-by-step workflow for template normalization and within-subject coregistration
- `skull_strip` writes to a configurable `output_dir` (consistent with the other tools)
- Initial template scaffold: pyproject + uv, ruff + pyright strict, pytest, just, pre-commit
- GitHub Actions CI workflow (lint, format-check, pyright, pytest on py3.12 / 3.13)
- Contributor docs: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Issue and PR templates with medical-context PHI warnings
- [all-contributors](https://allcontributors.org) setup (`.all-contributorsrc` + README section) to credit all contribution types
- Documented Semantic Versioning policy (tool name/parameters/result shape as the public contract) in CONTRIBUTING
- Shared-file contract with [medmcp-template](https://github.com/medmcp/medmcp-template):
  `scripts/shared-files.txt` lists the files every stack takes from the template,
  `scripts/sync-from-template.sh` pulls them in, and a **Template drift** workflow
  reports when one diverges.
- `NOTICE` crediting the bundled tools, models, and templates (HD-BET, ANTs/ANTsPy, FastSurfer, MNI152NLin2009cAsym) with licenses and citations; README "Citation" section
