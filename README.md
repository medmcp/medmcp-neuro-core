# medmcp-neuro-core

Neuroimaging tools for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to perform brain extraction, image registration, whole-brain segmentation, super-resolution of thick-slice MRI, and related neuroimaging operations.

<p align="center">
  <a href="https://medmcp.ai"><b>medmcp.ai</b></a> ·
  <a href="https://github.com/medmcp/medmcp">Main repository</a>
</p>

> [!NOTE]
> **This repository is for developers** who build, extend, or run the neuro stack from source. **If you just want to use MedMCP, you don't need this repo** — install the MedMCP app and add this stack through the workspace UI (one-click install). See [medmcp.ai](https://medmcp.ai) or the [main repository](https://github.com/medmcp/medmcp) to get started.

> [!WARNING]
> MedMCP and its ecosystem are research software under active development and are **not licensed for clinical use**.

---

## Tool inventory

| Tool name | Description | Inputs | Outputs |
|---|---|---|---|
| `skull_strip` | Brain extraction using HD-BET. `device` follows the shared convention — `auto` (default) picks cuda > mps > cpu, or force `cuda`/`mps`/`cpu`; the resolved device is reported | `input_path: Path`, `output_dir: Path \| None = None`, `device: str = "auto"` | `{"brain_path": "...", "input_path": "...", "device": "...", "_render": "..."}` |
| `register_to_template` | Normalize a structural image to MNI152NLin2009cAsym (or a custom template); the template is baked into the container image, and downloaded on first use when running from source. Selects the brain-extracted template variant when `skull_stripped=True`. | `input_path: Path`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `skull_stripped: bool`, `output_dir: Path \| None = None`, `template_path: Path \| None = None` | `{"registered_path": "...", "transform_prefix": "...", "forward_transforms": [...], "inverse_transforms": [...], "inverse_invert_flags": [...], "template_path": "...", "transform_type": "...", "_render": "..."}` |
| `coregister` | Align multiple same-subject images to a common reference (e.g. FLAIR, T2w, b0 → T1w). Ask the user which transform type to use before calling | `fixed_path: Path`, `moving_paths: list[Path]`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `output_dir: Path \| None = None` | `{"registered_paths": [...], "transform_prefixes": [...], "forward_transforms_list": [[...]], "inverse_transforms_list": [[...]], "inverse_invert_flags_list": [[...]], "fixed_path": "...", "transform_type": "...", "_render": "..."}` |
| `apply_transform` | Apply a pre-computed ANTs transform to additional images (brain masks, lesion maps, parcellations) without re-running registration | `input_path: Path`, `reference_path: Path`, `transforms: list[str]`, `output_dir: Path \| None = None`, `interpolation: "Linear"\|"NearestNeighbor"\|"BSpline" = "Linear"`, `output_space: str \| None = None`, `invert_flags: list[bool] \| None = None` | `{"output_path": "...", "_render": "..."}` |
| `segment_brain` | Whole-brain segmentation using FastSurfer (FastSurferVINN, seg-only — **no FreeSurfer license**). T1w only; accepts full-head or skull-stripped input. Outputs a `.mgz` label map, a `label,structure` CSV naming its ids, and a per-structure volume CSV. Sanity-checks contrast/resolution (returns warnings; blocks strongly anisotropic/thick-slice data unless `force=True`). | `input_path: Path`, `output_dir: Path \| None = None`, `device: str = "auto"`, `threads: int = 4`, `force: bool = False` | `{"seg_path": "...", "labels_path": "...", "volumes_path": "...", "input_path": "...", "device": "...", "warnings": [...], "_render": "..."}` |
| `list_brain_segmentation_labels` | List all brain structures measured by `segment_brain`. Returns the exact structure names used in the volume CSV for a given `parc` setting. | `parc: bool = True` | `{"parc": ..., "total_structures": ..., "subcortical_and_global": [...], "cortical_parcels": [...], "_render": "..."}` |
| `super_resolve` | Super-resolve an anisotropic, thick-slice (2D-acquired) MRI volume to isotropic at the in-plane resolution with **ECLARE** (default) or **SMORE** v4. Self-supervised: no pretrained weights, a network is trained on each image (minutes to about an hour per volume on a GPU). Writes `{stem}_eclare.nii.gz` / `{stem}_smore.nii.gz`; (near-)isotropic input is skipped with a warning. Needs a CUDA GPU (`auto` refuses to fall back to CPU, `cpu` must be explicit; no MPS). | `input_path: Path`, `method: "eclare"\|"smore" = "eclare"`, `output_dir: Path \| None = None`, `device: "auto"\|"cuda"\|"cpu" = "auto"`, `slice_thickness_mm: float \| None = None`, `n_patches: int \| None = None` | `{"output_path": "..." \| null, "input_path": "...", "method": "...", "device": "...", "skipped": ..., "input_voxel_size_mm": [...], "output_voxel_size_mm": [...], "input_shape": [...], "output_shape": [...], "through_plane_axis": ..., "n_patches": ..., "runtime_sec": ..., "warnings": [...], "_render": "..."}` |

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_neuro_core/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `registration` | Workflow for template normalization and within-subject coregistration. Instructs the agent to present all transform options and wait for the user to choose; covers native↔template warping, the two-step multi-contrast workflow (coregister→apply_transform), transform composition, and label interpolation. |
| `segmentation` | Workflow for whole-brain segmentation and structure volumetry (e.g. thalamic volume) with FastSurfer. Covers confirming T1w input, not skull-stripping first, reading a specific structure's volume out of the CSV, and head-size normalization (via the license-free `BrainSegVol`, not ICV/eTIV) when comparing subjects. |
| `super-resolution` | Workflow for super-resolving thick-slice MRI with ECLARE or SMORE. Covers when it applies (one coarse through-plane axis, 2D acquisitions), choosing the method, setting runtime expectations before starting (per-image training), the optional acquired slice thickness, output naming, and the caveat that through-plane detail is synthesized. |

---

### Bundled tools, models & templates

`medmcp-neuro-core` wraps established third-party neuroimaging tools and redistributes some of their weights/checkpoints (baked into the container image). Each is used under its own license:

| Tool / template | Used by | Source | License |
|---|---|---|---|
| HD-BET (nnU-Net-based brain extraction) | `skull_strip` | weights downloaded via `hd-bet` to `~/hd-bet_params/` (baked into the image) | [Apache 2.0](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE) |
| ANTs / ANTsPy | `register_to_template`, `coregister`, `apply_transform` | [`antspyx`](https://github.com/ANTsX/ANTsPy) package dependency | [Apache 2.0](https://github.com/ANTsX/ANTsPy/blob/master/LICENSE) |
| FastSurfer (FastSurferVINN, seg-only) | `segment_brain` | checkpoint downloaded from the [FastSurfer](https://github.com/Deep-MI/FastSurfer) repo (baked into the image) | [Apache 2.0](https://github.com/Deep-MI/FastSurfer/blob/stable/LICENSE) |
| ECLARE | `super_resolve` (`method="eclare"`) | [`eclare`](https://pypi.org/project/eclare/) 1.0.1 package dependency ([source](https://github.com/sremedios/ECLARE)); self-supervised, no weights | [GPL-3.0](https://github.com/sremedios/ECLARE/blob/main/LICENSE). Runs as its own program (`run-eclare` subprocess) and is never imported |
| SMORE v4 | `super_resolve` (`method="smore"`) | [`smore`](https://gitlab.com/iacl/smore) v4.0.5 git dependency, with its [`resize`](https://gitlab.com/iacl/resize) and [`degrade`](https://gitlab.com/iacl/degrade) helpers; self-supervised, no weights | Apache 2.0 (declared in [`setup.py`](https://gitlab.com/iacl/smore/-/blob/v4.0.5/setup.py); the repo ships no LICENSE file); `resize`/`degrade` [MIT](https://gitlab.com/iacl/degrade/-/blob/v0.4.0/LICENSE.txt) |
| MNI152NLin2009cAsym template | `register_to_template` | retrieved from [TemplateFlow](https://github.com/templateflow/tpl-MNI152NLin2009cAsym) (baked into the image) | [template license](https://github.com/templateflow/tpl-MNI152NLin2009cAsym/blob/master/LICENSE) |

### Citation

These are third-party scientific methods. **If you use `medmcp-neuro-core` results in research, please cite the underlying tools:**

- **HD-BET** — Isensee F, et al. Automated brain extraction of multi-sequence MRI using artificial neural networks. *Human Brain Mapping* (2019). [doi:10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)
- **ANTs / ANTsPy** — Tustison NJ, et al. The ANTsX ecosystem for quantitative biological and medical imaging. *Scientific Reports* 11, 9068 (2021). [doi:10.1038/s41598-021-87564-6](https://doi.org/10.1038/s41598-021-87564-6)
- **FastSurfer** — Henschel L, et al. FastSurfer – A fast and accurate deep learning based neuroimaging pipeline. *NeuroImage* 219, 117012 (2020). [doi:10.1016/j.neuroimage.2020.117012](https://doi.org/10.1016/j.neuroimage.2020.117012). FastSurferVINN — Henschel L, et al. *NeuroImage* 251, 118933 (2022). [doi:10.1016/j.neuroimage.2022.118933](https://doi.org/10.1016/j.neuroimage.2022.118933)
- **ECLARE** — Remedios SW, et al. ECLARE: efficient cross-planar learning for anisotropic resolution enhancement. *Journal of Medical Imaging* 13(2), 024001 (2026). [doi:10.1117/1.JMI.13.2.024001](https://doi.org/10.1117/1.JMI.13.2.024001)
- **SMORE** — Zhao C, et al. SMORE: A Self-Supervised Anti-Aliasing and Super-Resolution Algorithm for MRI Using Deep Learning. *IEEE Transactions on Medical Imaging* 40(3), 805–817 (2021). [doi:10.1109/TMI.2020.3037187](https://doi.org/10.1109/TMI.2020.3037187). SMORE v4 (the bundled version) — Remedios SW, et al. Self-supervised super-resolution for anisotropic MR images with and without slice gap. *SASHIMI 2023*, LNCS, 118–128 (2023). [doi:10.1007/978-3-031-44689-4_12](https://doi.org/10.1007/978-3-031-44689-4_12)
- **MNI152NLin2009cAsym template** — Fonov V, et al. Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage* 54(1), 313–327 (2011). [doi:10.1016/j.neuroimage.2010.07.033](https://doi.org/10.1016/j.neuroimage.2010.07.033). Retrieved via TemplateFlow — Ciric R, et al. *Nature Methods* 19, 1568–1571 (2022). [doi:10.1038/s41592-022-01681-2](https://doi.org/10.1038/s41592-022-01681-2)

Full third-party attribution is in [`NOTICE`](NOTICE).

### Hardware requirements

`skull_strip` supports CPU (TTA disabled) and GPU (`device="cuda"` for NVIDIA, `device="mps"` for Apple Silicon), and is substantially faster on GPU. `register_to_template`, `coregister`, and `apply_transform` call ANTsPy, which is installed automatically as a package dependency and is CPU-only; registration time varies with image size and hardware — `syn` is significantly slower than the other transform types. `segment_brain` (FastSurfer) auto-detects the device (`device="auto"`) and is substantially faster on a GPU (`cuda`/`mps`) than on CPU. `super_resolve` (ECLARE/SMORE) trains a network per image and in practice needs an NVIDIA GPU (see the `super-resolution` skill for measured runtimes). CPU has to be requested explicitly and takes many hours. Apple MPS is not supported upstream.

---

## Development

### Develop in the dev container (recommended)

This repo ships a dev container (`.devcontainer/`) with the full toolchain
(Python 3.12 + uv, `just`, git, Docker CLI). It derives from the shared
`medmcp-base` image, so build that once from the core repo first (`just docker-base`
in a `medmcp` checkout). Then open the repo with the **Dev Container** action in PyCharm
(2024.2+) or **Reopen in Container** in VS Code — `uv sync` runs on first start.
The dev container requests no GPU by default (so it starts anywhere); to run GPU
code from inside add `--device nvidia.com/gpu=all`, or build and run the image. See
the core repo's [CONTRIBUTING](https://github.com/medmcp/medmcp/blob/main/CONTRIBUTING.md)
for IDE specifics.

### Local install (alternative)

```bash
just setup     # install uv, sync dev environment, register pre-commit hooks
just check     # lint + format-check + typecheck + tests
just fix       # auto-fix lint and format
just test      # pytest only
```

For local agent use, install the stack into its own uv tool environment:

```bash
uv tool install --editable .
```

The package registers itself via the `[medmcp.stacks]` entry point. The local agent autodiscovers it on the next session — no manual config needed.

### Container image (deployment)

This stack also ships as a GPU container with a fixed environment:

```bash
just docker-build           # build medmcp-neuro-core:dev (FROM medmcp-base)
```

It is a stdio MCP server (`ENTRYPOINT ["tini", "--", "medmcp-neuro-core"]`). The medmcp
**core** launches it on demand via a `stacks.d/medmcp-neuro-core.toml` manifest
(`docker run -i --device nvidia.com/gpu=all …`, CDI/rootless), so deployment nodes
need no host Python install. torch is pinned to the **cu128 (CUDA 12.8)** build, so
the image runs on any host with NVIDIA driver **≥ R570** (Turing→Blackwell; newer
drivers via CUDA backward-compatibility). All tools — `skull_strip`, the registration
tools, `segment_brain`, and `super_resolve` — work in-container; `segment_brain` is backed by
**FastSurfer** (seg-only, no FreeSurfer license), whose seg dependencies share the
single application venv (torch is pinned to FastSurfer's `2.7.1`, so HD-BET, antspyx,
and FastSurfer all use one CUDA stack), with the FastSurferVINN checkpoint baked at build.

### Staying in sync with the template

Files shared with [medmcp-template](https://github.com/medmcp/medmcp-template) are
listed in `scripts/shared-files.txt`. The **Template drift** workflow reports when
one of them diverges; `./scripts/sync-from-template.sh` pulls them back. A change
that belongs in every stack goes in the template, not here.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: fork, `just setup`, `just check`, open a PR against `main`.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://jqmcginnis.github.io/"><img src="https://avatars.githubusercontent.com/u/33037028?v=4?s=100" width="100px;" alt="Julian McGinnis"/><br /><sub><b>Julian McGinnis</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-neuro-core/commits?author=jqmcginnis" title="Code">💻</a> <a href="https://github.com/medmcp/medmcp-neuro-core/commits?author=jqmcginnis" title="Documentation">📖</a> <a href="https://github.com/medmcp/medmcp-neuro-core/pulls?q=is%3Apr+reviewed-by%3Ajqmcginnis" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://pfriedri.github.io"><img src="https://avatars.githubusercontent.com/u/101359393?v=4?s=100" width="100px;" alt="Paul Friedrich"/><br /><sub><b>Paul Friedrich</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-neuro-core/commits?author=pfriedri" title="Code">💻</a> <a href="https://github.com/medmcp/medmcp-neuro-core/commits?author=pfriedri" title="Documentation">📖</a> <a href="https://github.com/medmcp/medmcp-neuro-core/pulls?q=is%3Apr+reviewed-by%3Apfriedri" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification — contributions of any kind are welcome!

## License

[Apache 2.0](LICENSE). Third-party tools, model weights, and templates bundled by this stack retain their own licenses and are attributed in [`NOTICE`](NOTICE).
