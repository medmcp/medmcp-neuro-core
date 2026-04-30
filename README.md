# medmcp-neuro

Neuroimaging tools for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio so an LLM agent can call neuroimaging operations by name.

> [!WARNING]
> Research software under active development — **not licensed for clinical use**.

---

## Tool inventory

| Tool | Description | Key inputs | Outputs |
|---|---|---|---|
| `skull_strip` | Brain extraction using HD-BET | `input_path`, `device` | `brain_path` |
| `register_to_template` | Normalise a structural image to a standard-space template (default: MNI152NLin2009cAsym); template downloaded on first use | `input_path`, `output_dir`, `transform_type` (`rigid`/`affine`/`synquick`/`syn`) | `registered_path`, `forward_transforms`, `inverse_transforms` |
| `coregister` | Align multiple same-subject images to a common reference (e.g. FLAIR, T2w, b0 → T1w) | `fixed_path`, `moving_paths`, `output_dir`, `transform_type` (`rigid`/`affine`) | `registered_paths`, `transform_prefixes` |
| `apply_transform` | Apply a pre-computed ANTs transform to additional images (masks, parcellations, lesion maps) | `input_path`, `reference_path`, `transforms`, `output_dir`, `interpolation` | `output_path` |

---

## Model / weights provenance

| Tool | Model | Source | License |
|---|---|---|---|
| `skull_strip` | HD-BET (nnU-Net-based brain extraction) | Downloaded automatically on first run via `hd-bet` to `~/.hd_bet_data/` | [Apache 2.0](https://github.com/MIC-DKFZ/HD-BET/blob/master/LICENSE) |

---

## Hardware requirements

| Tool | CPU | GPU |
|---|---|---|
| `skull_strip` | Supported (~2–3 min per volume, TTA disabled automatically) | `device="cuda"` or `device="mps"` (~30 s, TTA enabled) |

---

## What's in the box

| Area | Files | Notes |
|---|---|---|
| Build / deps | `pyproject.toml`, `.python-version` | uv-managed, Python ≥3.12, `mcp>=1.0`, `hd-bet` |
| MCP server | `src/medmcp_neuro/server.py` | FastMCP over stdio; `server_config()` enables autodiscovery |
| Tools | `src/medmcp_neuro/tools/` | One file per tool; shared helpers in `_neuro.py` |
| Dev workflow | `justfile`, `.pre-commit-config.yaml` | `just setup`, `just check`, `just fix` |
| CI | `.github/workflows/ci.yml` | Lint, format-check, pyright (strict), pytest on py3.12 / 3.13 |

---

## Development

```bash
just setup     # install uv, sync dev environment, register pre-commit hooks
just check     # lint + format-check + typecheck + tests
just fix       # auto-fix lint and format
just test      # pytest only
```

## Install and activate

```bash
uv tool install .        # local dev
# or
uv tool install medmcp-neuro   # from PyPI once published
```

The package registers itself via the `[medmcp.stacks]` entry point. The local agent discovers it automatically on the next session — no manual config needed.

## License

[Apache 2.0](LICENSE)
