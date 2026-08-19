# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

First public release. `medmcp-neuro-core` is the neuroimaging tool stack for
MedMCP — an MCP server exposing brain extraction, registration, and whole-brain
segmentation as tools an agent calls by name. Model weights and the template are
baked into the container image, so every tool runs with networking denied.

**Not licensed for clinical use.**

### Changed

- Tracks the files shared with [medmcp-template](https://github.com/medmcp/medmcp-template):
  `scripts/shared-files.txt` lists them, `scripts/sync-from-template.sh` pulls them
  in, and a **Template drift** workflow reports when one diverges. This first sync
  picked up a CI action bump that had already landed in the template.
- Dependabot ignores `torch`/`torchvision`. Both are pinned exactly to the cu128
  build so the GPU image runs on any driver >= R570; bumping them moves the CUDA
  build and with it the minimum driver, which is a decision about the fleet rather
  than routine maintenance. The two open bump PRs could not resolve anyway and are
  closed.
- `CODEOWNERS` removed. It was entirely commented out behind a "replace before the
  repo goes public" note, so it assigned no ownership and requested no reviews.
- References to the core repo use its current name, `medmcp`, not the pre-rename
  `medmcp-dev` — including one in `pyproject.toml` that pointed at a design
  document which exists only on a maintainer's machine.

### Fixed

- `register_to_template` now works in the container. The MNI152 template was the one asset still fetched at runtime rather than baked into the image, so once stack containers began running with networking denied, every template registration failed at the download with a name-resolution error. Both template variants (whole-head and skull-stripped) are now baked at build time, like the HD-BET and FastSurfer weights already were.

### Changed

- Image builds reuse the previous build's layers, and install dependencies before copying the source. The arm64 leg compiles antspyx from its sdist (no aarch64 wheel is published past 0.4.2), which measured 2158s of a 2376s build — and that layer sat behind the source copy, so editing a single line of Python recompiled ITK and ANTs from scratch. A change that leaves `uv.lock` alone now pulls the layer instead of rebuilding it: roughly 40 minutes down to roughly 4. A dependency change still pays the full compile. The image contents are unchanged.
- Only one image build runs per branch at a time. A superseded push is cancelled instead of spending most of an hour producing an image nobody will pull; pushes to `main` still always finish, since both architectures have to complete for the multi-arch manifest to be assembled.

### Added

- `skull_strip` persistent HD-BET worker: loads torch + CUDA + the model once and reuses it across calls (opt-in via `MEDMCP_HDBET_PERSIST`, or once pre-loaded by the new `warmup` tool), reclaiming the per-call model-load cost on warm calls. Falls back to the per-call subprocess if the worker is unavailable, so behaviour is unchanged by default.
- `warmup` tool to pre-load the HD-BET model so the first `skull_strip` is already warm (the hook the workspace pre-warm pool calls on activation).
- Shared device convention for GPU-capable tools: `device="auto"` (the default) resolves to cuda > mps > cpu via `resolve_device()` in `_neuro.py`, and the resolved device is reported in the result (with a CPU-fallback note). `skull_strip`, `warmup`, and `segment_brain` all default to `auto`; the CPU-only ANTs registration tools take no device parameter.
- Container image: `Dockerfile` (`FROM medmcp-base`; torch pinned to the cu128 build, version `2.7.1`, so it runs on any host driver >= R570; baked HD-BET weights + the FastSurferVINN checkpoint; `segment_brain` runs in-container via FastSurfer) + `.dockerignore`; `org.medmcp.stack` label for one-click install; `.devcontainer`; CI publishes to the private `ghcr.io/medmcp/neuro-core`. Image-size optimized: HD-BET, antspyx, and FastSurfer share one torch/CUDA venv (torch pinned to FastSurfer's `2.7.1` so the ~8 GB CUDA stack isn't installed twice), only the asegdkt checkpoint is baked (`--vinn`, not `--all`), and bytecode/static-archive cruft is stripped in-layer.

- `segment_brain` and `list_brain_segmentation_labels` tools backed by FastSurfer (FastSurferVINN, seg-only — **no FreeSurfer license required**); whole-brain segmentation of 95 cortical/subcortical classes (33 subcortical/global + 31 DKT cortical parcels per hemisphere). The per-structure volume CSV uses FastSurfer's exact StructNames, omits the always-empty corpus-callosum rows (CC is only populated by the surface pipeline, which seg-only does not run), and appends a `BrainSegVol` row (license-free total brain-segmentation volume) for head-size normalization. T1w only; input contrast and resolution are sanity-checked (non-fatal warnings, with strongly anisotropic/thick-slice data blocked unless `force=True`)
- `register_to_template`, `coregister`, and `apply_transform` tools backed by ANTsPy
- MNI152NLin2009cAsym 1 mm template auto-download with local cache; brain-extracted variant selected via `skull_stripped` parameter
- `registration` SKILL.md with step-by-step workflow for template normalization and within-subject coregistration
- `skull_strip` writes to a configurable `output_dir` (consistent with the other tools)
- Initial template scaffold: pyproject + uv, ruff + pyright strict, pytest, just, pre-commit
- GitHub Actions CI workflow (lint, format-check, pyright, pytest on py3.12 / 3.13)
- Contributor docs: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Issue and PR templates with medical-context PHI warnings
- [all-contributors](https://allcontributors.org) setup (`.all-contributorsrc` + README section) to credit all contribution types
- Documented Semantic Versioning policy (tool name/parameters/result shape as the public contract) in CONTRIBUTING
- `NOTICE` crediting the bundled tools, models, and templates (HD-BET, ANTs/ANTsPy, FastSurfer, MNI152NLin2009cAsym) with licenses and citations; README "Citation" section
