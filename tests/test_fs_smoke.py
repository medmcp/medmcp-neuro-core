"""The FreeSurfer tools are registered with the signatures the skill and the core rely on."""

from pathlib import Path
from typing import Any

import pytest

from medmcp_neuro_core.server import mcp


@pytest.mark.asyncio
async def test_freesurfer_tools_registered() -> None:
    """The server exposes the three FreeSurfer tools beside the others."""
    names = {t.name for t in await mcp.list_tools()}
    assert {"run_recon_all", "recon_all_status", "freesurfer_stats", "segment_brain"} <= names


@pytest.mark.asyncio
async def test_run_recon_all_schema() -> None:
    """The tool signature the skill and the core's job runner rely on."""
    tool = next(t for t in await mcp.list_tools() if t.name == "run_recon_all")
    props = tool.inputSchema["properties"]
    assert set(tool.inputSchema["required"]) == {"t1_path", "output_dir"}
    for name in ("subject_id", "license_path", "threads", "pipeline", "force"):
        assert name in props, name
    assert props["threads"]["default"] == 8
    assert props["pipeline"]["default"] == "classic"


@pytest.mark.asyncio
async def test_segment_brain_surfaces_schema() -> None:
    """segment_brain keeps its signature and gains surfaces / license_path, off by default."""
    tool = next(t for t in await mcp.list_tools() if t.name == "segment_brain")
    props = tool.inputSchema["properties"]
    assert props["surfaces"]["default"] is False
    assert "license_path" in props
    assert set(tool.inputSchema["required"]) == {"input_path"}


@pytest.mark.asyncio
async def test_structured_output_keeps_render(workspace: Path, t1: Path) -> None:
    """Through FastMCP, results keep _render and carry no null medmcp_job.

    A TypedDict return type loses keys starting with "_" from the structured output
    and fills an absent NotRequired key with null, which fails output validation.
    """
    # No licence: the licence_required result, which has no medmcp_job.
    out = await mcp.call_tool(
        "run_recon_all", {"t1_path": str(t1), "output_dir": str(workspace / "fs")}
    )
    assert isinstance(out, tuple)
    structured: Any = out[1]
    assert isinstance(structured, dict)
    assert structured["status"] == "licence_required"
    assert "_render" in structured and "medmcp_job" not in structured


def test_job_entrypoint_declared() -> None:
    """The console script the image label's job entry names exists in the package."""
    from importlib.metadata import entry_points

    scripts = {e.name: e.value for e in entry_points(group="console_scripts")}
    assert scripts.get("medmcp-neuro-core-job") == "medmcp_neuro_core.job:main"
