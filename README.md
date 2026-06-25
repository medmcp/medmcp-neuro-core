# medmcp-neuro

Neuroimaging tools for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to perform brain extraction, image registration, whole-brain segmentation, and related neuroimaging operations.

> [!WARNING]
> MedMCP and its ecosystem are research software under active development and are **not licensed for clinical use**.

---

## Tool inventory

| Tool name | Description | Inputs | Outputs |
|---|---|---|---|
| `skull_strip` | Brain extraction using HD-BET. Ask the user which device to use (`cpu`/`cuda`/`mps`) before calling; defaults to `cpu` | `input_path: Path`, `output_dir: Path \| None = None`, `device: str = "cpu"` | `{"brain_path": "...", "input_path": "...", "device": "...", "_render": "..."}` |
| `register_to_template` | Normalize a structural image to MNI152NLin2009cAsym (or a custom template); template downloaded on first use. Selects the brain-extracted template variant when `skull_stripped=True`. | `input_path: Path`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `skull_stripped: bool`, `output_dir: Path \| None = None`, `template_path: Path \| None = None` | `{"registered_path": "...", "forward_transforms": [...], "inverse_transforms": [...], "inverse_invert_flags": [...], "template_path": "...", "transform_type": "...", "_render": "..."}` |
| `coregister` | Align multiple same-subject images to a common reference (e.g. FLAIR, T2w, b0 → T1w). Ask the user which transform type to use before calling | `fixed_path: Path`, `moving_paths: list[Path]`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `output_dir: Path \| None = None` | `{"registered_paths": [...], "transform_prefixes": [...], "forward_transforms_list": [[...]], "inverse_transforms_list": [[...]], "inverse_invert_flags_list": [[...]], "_render": "..."}` |
| `apply_transform` | Apply a pre-computed ANTs transform to additional images (brain masks, lesion maps, parcellations) without re-running registration | `input_path: Path`, `reference_path: Path`, `transforms: list[str]`, `output_dir: Path \| None = None`, `interpolation: "Linear"\|"NearestNeighbor"\|"BSpline" = "Linear"`, `output_space: str \| None = None`, `invert_flags: list[bool] \| None = None` | `{"output_path": "...", "_render": "..."}` |
| `segment_brain` | Whole-brain segmentation using FastSurfer (FastSurferVINN, seg-only — **no FreeSurfer license**). T1w only; accepts full-head or skull-stripped input. Outputs a `.mgz` label map + per-structure volume CSV. Sanity-checks contrast/resolution (returns warnings; blocks strongly anisotropic/thick-slice data unless `force=True`). | `input_path: Path`, `output_dir: Path \| None = None`, `device: str = "auto"`, `threads: int = 4`, `force: bool = False` | `{"seg_path": "...", "volumes_path": "...", "input_path": "...", "device": "...", "warnings": [...], "_render": "..."}` |
| `list_brain_segmentation_labels` | List all brain structures measured by `segment_brain`. Returns exact CSV column names for a given `parc` setting. | `parc: bool = True` | `{"parc": ..., "total_structures": ..., "subcortical_and_global": [...], "cortical_parcels": [...], "_render": "..."}` |

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_neuro/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `registration` | Workflow for template normalization and within-subject coregistration. Instructs the agent to present all transform options and wait for the user to choose; covers native↔template warping, the two-step multi-contrast workflow (coregister→apply_transform), transform composition, and label interpolation. |
| `segmentation` | Workflow for whole-brain segmentation and structure volumetry (e.g. thalamic volume) with FastSurfer. Covers confirming T1w input, not skull-stripping first, reading a specific structure's volume out of the CSV, and ICV normalization when comparing subjects. |

---

### Model / weights provenance

| Tool | Model | Source | License |
|---|---|---|---|
| `skull_strip` | HD-BET (nnU-Net-based brain extraction) | Downloaded automatically on first run via `hd-bet` to `~/.hd_bet_data/` | [Apache 2.0](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE) |
| `segment_brain` | FastSurfer (FastSurferVINN whole-brain segmentation) | Checkpoints downloaded from the [FastSurfer](https://github.com/Deep-MI/FastSurfer) repo on first use (baked into the container image at build) | [Apache 2.0](https://github.com/Deep-MI/FastSurfer/blob/stable/LICENSE) |

### Runtime dependencies

| Tool | Installed via | Action required |
|---|---|---|
| `skull_strip` | `hd-bet` Python package | None — installed automatically with `pip install medmcp-neuro` |
| `register_to_template`, `coregister`, `apply_transform` | `antspyx` Python package | None — installed automatically |
| `segment_brain` | FastSurfer (`run_fastsurfer.sh`) | **Manual install required outside the container** — clone [FastSurfer](https://github.com/Deep-MI/FastSurfer) and set `$FASTSURFER_HOME` (or `$RUN_FASTSURFER`). Pre-installed in the container image. No FreeSurfer license needed (seg-only pipeline). |

### Hardware requirements

`skull_strip` supports CPU (~2–3 min per volume, TTA disabled) and GPU (`device="cuda"` for NVIDIA, `device="mps"` for Apple Silicon, ~30 s). `register_to_template`, `coregister`, and `apply_transform` call ANTsPy, which is installed automatically as a package dependency and is CPU-only; registration time varies with image size and hardware — `syn` is significantly slower than the other transform types. `segment_brain` (FastSurfer) auto-detects the device (`device="auto"`): minutes per subject on a GPU (`cuda`/`mps`), substantially slower on CPU.

---

## Development

### Develop in the dev container (recommended)

This repo ships a dev container (`.devcontainer/`) with the full toolchain
(Python 3.12 + uv, `just`, git, Docker CLI). It derives from the shared
`medmcp-base` image, so build that once from the core repo first (`just docker-base`
in `medmcp-dev`). Then open the repo with the **Dev Container** action in PyCharm
(2024.2+) or **Reopen in Container** in VS Code — `uv sync` runs on first start.
The dev container requests no GPU by default (so it starts anywhere); to run GPU
code from inside add `--device nvidia.com/gpu=all`, or build and run the image. See
the core repo's [CONTRIBUTING](https://github.com/medmcp/medmcp-dev/blob/main/CONTRIBUTING.md)
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
just docker-build           # build medmcp-neuro:dev (FROM medmcp-base)
```

It is a stdio MCP server (`ENTRYPOINT ["tini", "--", "medmcp-neuro"]`). The medmcp
**core** launches it on demand via a `stacks.d/medmcp-neuro.toml` manifest
(`docker run -i --device nvidia.com/gpu=all …`, CDI/rootless), so deployment nodes
need no host Python install. torch is pinned to the **cu128 (CUDA 12.8)** build, so
the image runs on any host with NVIDIA driver **≥ R570** (Turing→Blackwell; newer
drivers via CUDA backward-compatibility). All tools — `skull_strip`, the registration
tools, and `segment_brain` — work in-container; `segment_brain` is backed by
**FastSurfer** (seg-only, no FreeSurfer license), whose seg dependencies share the
single application venv (torch is pinned to FastSurfer's `2.7.1`, so HD-BET, antspyx,
and FastSurfer all use one CUDA stack), with the FastSurferVINN checkpoint baked at build.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: fork, `just setup`, `just check`, open a PR against `main`.

## License

[Apache 2.0](LICENSE)
