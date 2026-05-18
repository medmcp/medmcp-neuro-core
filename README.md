# medmcp-neuro

Neuroimaging tools for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to perform brain extraction, image registration, and related neuroimaging operations.

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
| `segment_brain` | Segment brain structures using SynthSeg (FreeSurfer). Contrast-agnostic deep learning segmentation; works on T1w, T2w, FLAIR, DWI with or without prior skull stripping. | `input_path: Path`, `output_dir: Path \| None = None`, `parc: bool = True`, `robust: bool = False` | `{"seg_path": "...", "volumes_path": "...", "input_path": "...", "_render": "..."}` |
| `list_brain_segmentation_labels` | List all brain structures measured by `segment_brain`. Returns exact CSV column names for a given `parc` setting. | `parc: bool = True` | `{"parc": ..., "total_structures": ..., "subcortical_and_global": [...], "cortical_parcels": [...], "_render": "..."}` |

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_neuro/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `registration` | Workflow for template normalization and within-subject coregistration. Instructs the agent to present all transform options and wait for the user to choose; covers native↔template warping, the two-step multi-contrast workflow (coregister→apply_transform), transform composition, and label interpolation. |

---

### Model / weights provenance

| Tool | Model | Source | License |
|---|---|---|---|
| `skull_strip` | HD-BET (nnU-Net-based brain extraction) | Downloaded automatically on first run via `hd-bet` to `~/.hd_bet_data/` | [Apache 2.0](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE) |
| `segment_brain` | SynthSeg (contrast-agnostic deep learning segmentation) | Bundled with [FreeSurfer ≥ 7.3](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) — **must be installed separately**, not a Python package | [FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense) (free for research) |

### Runtime dependencies

| Tool | Installed via | Action required |
|---|---|---|
| `skull_strip` | `hd-bet` Python package | None — installed automatically with `pip install medmcp-neuro` |
| `register_to_template`, `coregister`, `apply_transform` | `antspyx` Python package | None — installed automatically |
| `segment_brain` | FreeSurfer (`mri_synthseg` binary) | **Manual install required** — see [FreeSurfer installation guide](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). Set `$FREESURFER_HOME` or put `mri_synthseg` on `$PATH`. |

### Hardware requirements

`skull_strip` supports CPU (~2–3 min per volume, TTA disabled) and GPU (`device="cuda"` for NVIDIA, `device="mps"` for Apple Silicon, ~30 s). `register_to_template`, `coregister`, and `apply_transform` call ANTsPy, which is installed automatically as a package dependency and is CPU-only; registration time varies with image size and hardware — `syn` is significantly slower than the other transform types. `segment` (SynthSeg) runs on CPU or GPU and takes ~1–2 min per volume on CPU.

---

## Development

```bash
just setup     # install uv, sync dev environment, register pre-commit hooks
just check     # lint + format-check + typecheck + tests
just fix       # auto-fix lint and format
just test      # pytest only
```

### Install for local agent use

```bash
uv tool install --editable .
```

The package registers itself via the `[medmcp.stacks]` entry point. The local agent autodiscovers it on the next session — no manual config needed.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: fork, `just setup`, `just check`, open a PR against `main`.

## License

[Apache 2.0](LICENSE)
