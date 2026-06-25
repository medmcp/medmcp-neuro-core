# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `skull_strip` persistent HD-BET worker: loads torch + CUDA + the model once and reuses it across calls (opt-in via `MEDMCP_HDBET_PERSIST`, or once pre-loaded by the new `warmup` tool), reclaiming ~10 s of model-load per warm call (~23 s → ~11 s on GPU). Falls back to the per-call subprocess if the worker is unavailable, so behaviour is unchanged by default.
- `warmup` tool to pre-load the HD-BET model so the first `skull_strip` is already warm (the hook the workspace pre-warm pool calls on activation).
- Shared device convention for GPU-capable tools: `device="auto"` (now the default) resolves to cuda > mps > cpu via `resolve_device()` in `_neuro.py`, and the resolved device is reported in the result (with a CPU-fallback note). `skull_strip` and `warmup` now default to `auto` (were `cpu`/`cuda`), matching `segment_brain`; the CPU-only ANTs registration tools keep no device parameter.
- Container image: `Dockerfile` (`FROM medmcp-base`; torch pinned to the cu128 build, version `2.7.1`, so it runs on any host driver >= R570; baked HD-BET weights + the FastSurferVINN checkpoint; `segment_brain` runs in-container via FastSurfer) + `.dockerignore`; `org.medmcp.stack` label for one-click install; `.devcontainer`; CI publishes to the private `ghcr.io/medmcp/neuro`. Image-size optimized: HD-BET, antspyx, and FastSurfer share one torch/CUDA venv (torch pinned to FastSurfer's `2.7.1` so the ~8 GB CUDA stack isn't installed twice), only the asegdkt checkpoint is baked (`--vinn`, not `--all`), and bytecode/static-archive cruft is stripped in-layer.

- `segment_brain` and `list_brain_segmentation_labels` tools backed by FastSurfer (FastSurferVINN, seg-only — **no FreeSurfer license required**); whole-brain segmentation of 95 cortical/subcortical classes (33 subcortical/global + 31 DKT cortical parcels per hemisphere). The per-structure volume CSV uses FastSurfer's exact StructNames, omits the always-empty corpus-callosum rows (CC is only populated by the surface pipeline, which seg-only does not run), and appends a `BrainSegVol` row (license-free total brain-segmentation volume) for head-size normalization. T1w only; input contrast and resolution are sanity-checked (non-fatal warnings, with strongly anisotropic/thick-slice data blocked unless `force=True`)
- `register_to_template`, `coregister`, and `apply_transform` tools backed by ANTsPy
- MNI152NLin2009cAsym 1 mm template auto-download with local cache; brain-extracted variant selected via `skull_stripped` parameter
- `registration` SKILL.md with step-by-step workflow for template normalization and within-subject coregistration
- `skull_strip` gains `output_dir` parameter (consistent with other tools)
- Initial template scaffold: pyproject + uv, ruff + pyright strict, pytest, just, pre-commit
- GitHub Actions CI workflow (lint, format-check, pyright, pytest on py3.12 / 3.13)
- Contributor docs: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Issue and PR templates with medical-context PHI warnings
- Rename helper script for one-shot placeholder replacement
