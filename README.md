# medmcp-neuro

Neuroimaging tools for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to perform brain extraction, image registration, and related neuroimaging operations.

> [!WARNING]
> MedMCP and its ecosystem are research software under active development and are **not licensed for clinical use**.

---

## Tool inventory

| Tool name | Description | Inputs | Outputs |
|---|---|---|---|
| `skull_strip` | Brain extraction using HD-BET. Ask the user which device to use (`cpu`/`cuda`/`mps`) before calling; defaults to `cpu` | `input_path: Path`, `output_dir: Path \| None = None`, `device: str = "cpu"` | `{"brain_path": "...", "input_path": "...", "device": "...", "_render": "..."}` |
| `register_to_template` | Normalise a structural image to MNI152NLin2009cAsym (or a custom template); template downloaded on first use. Selects the brain-extracted template variant when `skull_stripped=True`. | `input_path: Path`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `output_dir: Path \| None = None`, `template_path: Path \| None = None`, `skull_stripped: bool = False` | `{"registered_path": "...", "forward_transforms": [...], "inverse_transforms": [...], "inverse_invert_flags": [...], "template_path": "...", "transform_type": "...", "_render": "..."}` |
| `coregister` | Align multiple same-subject images to a common reference (e.g. FLAIR, T2w, b0 → T1w). Ask the user which transform type to use before calling | `fixed_path: Path`, `moving_paths: list[Path]`, `transform_type: "rigid"\|"similarity"\|"affine"\|"synquick"\|"syn"`, `output_dir: Path \| None = None` | `{"registered_paths": [...], "transform_prefixes": [...], "forward_transforms_list": [[...]], "inverse_transforms_list": [[...]], "inverse_invert_flags_list": [[...]], "_render": "..."}` |
| `apply_transform` | Apply a pre-computed ANTs transform to additional images (brain masks, lesion maps, parcellations) without re-running registration | `input_path: Path`, `reference_path: Path`, `transforms: list[str]`, `output_dir: Path \| None = None`, `interpolation: "Linear"\|"NearestNeighbor"\|"BSpline" = "Linear"`, `output_space: str \| None = None`, `invert_flags: list[bool] \| None = None` | `{"output_path": "...", "_render": "..."}` |
| `segment_ms_lesions` | White-matter (MS) lesion segmentation from paired T1w + FLAIR using LST-AI. Run out of process via the `lst` console script from a dedicated LST-AI virtualenv (`$MEDMCP_LST_AI_BIN` / PATH). Ask the user which device to use (`cpu`/`cuda`); pass `skull_stripped=True` only when both inputs were already brain-extracted. | `t1_path: Path`, `flair_path: Path`, `output_dir: Path \| None = None`, `device: str = "cpu"`, `gpu_id: int = 0`, `skull_stripped: bool = False` | `{"lesion_mask_path": "...", "annotated_path": "..."\|null, "output_files": [...], "stats_files": [...], "device": "...", "_render": "..."}` |

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_neuro/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `registration` | Workflow for template normalisation and within-subject coregistration. Instructs the agent to present all transform options and wait for the user to choose; covers native↔template warping, the two-step multi-contrast workflow (coregister→apply_transform), transform composition, and label interpolation. |
| `lesion-segmentation` | Workflow for `segment_ms_lesions`. Covers the paired T1w + FLAIR requirement, the recommended pre-skull-stripping path (run `skull_strip` on both, then `skull_stripped=True`), device choice, and warping the resulting lesion mask into MNI with `apply_transform` (NearestNeighbor). |

---

### Model / weights provenance

| Tool | Model | Source | License |
|---|---|---|---|
| `skull_strip` | HD-BET (nnU-Net-based brain extraction) | Downloaded automatically on first run via `hd-bet` to `~/.hd_bet_data/` | [Apache 2.0](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE) |
| `segment_ms_lesions` | LST-AI lesion-segmentation ensemble | Weights bundled with the LST-AI install; the `greedy` registration binary is downloaded once to `~/.medmcp_neuro/bin/` if not on PATH | [LST-AI](https://github.com/CompImg/LST-AI) (see upstream repo) |

### Hardware requirements

`skull_strip` supports CPU (~2–3 min per volume, TTA disabled) and GPU (`device="cuda"` for NVIDIA, `device="mps"` for Apple Silicon, ~30 s). `register_to_template`, `coregister`, and `apply_transform` call ANTsPy, which is installed automatically as a package dependency and is CPU-only; registration time varies with image size and hardware — `syn` is significantly slower than the other transform types.

`segment_ms_lesions` runs LST-AI out of process (it is **not** a package dependency — it conflicts with this stack's torch/HD-BET versions). Install it once in its own virtualenv and point `$MEDMCP_LST_AI_BIN` at the `lst` console script:

```bash
python3 -m venv ~/.medmcp_neuro/lst-ai-venv
~/.medmcp_neuro/lst-ai-venv/bin/pip install git+https://github.com/CompImg/LST-AI
export MEDMCP_LST_AI_BIN=~/.medmcp_neuro/lst-ai-venv/bin/lst
```

It supports CPU (often >1 h per case) and NVIDIA GPU (`device="cuda"`); there is no Apple-Silicon path. Relevant env vars: `MEDMCP_LST_AI_BIN` (path to `lst`), `MEDMCP_GREEDY_BIN` (path to `greedy`, otherwise auto-downloaded).

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
