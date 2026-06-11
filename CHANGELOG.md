# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `segment_lesions` tool: white-matter (MS) lesion segmentation from paired T1w + FLAIR via LST-AI, run out of process through a native `lst` venv (preferred) or the `jqmcginnis/lst-ai` Docker image (fallback); backend auto-selects and is overridable via `backend=` or `$MEDMCP_LST_AI_BACKEND`. The `greedy` binary is downloaded on first native use.
- `lesion-segmentation` SKILL.md covering the pre-skull-stripping path (`skull_strip` both inputs → `skull_stripped=True`), device choice, backend selection, and warping lesion masks into MNI
- `register_to_template`, `coregister`, and `apply_transform` tools backed by ANTsPy
- MNI152NLin2009cAsym 1 mm template auto-download with local cache; brain-extracted variant selected via `skull_stripped` parameter
- `registration` SKILL.md with step-by-step workflow for template normalisation and within-subject coregistration
- `skull_strip` gains `output_dir` parameter (consistent with other tools)
- Initial template scaffold: pyproject + uv, ruff + pyright strict, pytest, just, pre-commit
- GitHub Actions CI workflow (lint, format-check, pyright, pytest on py3.12 / 3.13)
- Contributor docs: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Issue and PR templates with medical-context PHI warnings
- Rename helper script for one-shot placeholder replacement
