"""MCP server entrypoint for medmcp-neuro-core."""

from importlib.resources import files as _pkg_files

from mcp.server.fastmcp import FastMCP

from medmcp_neuro_core.effects import tool_effect
from medmcp_neuro_core.tools.atlas import lesion_frequency_map, lesion_load_by_atlas, list_atlases
from medmcp_neuro_core.tools.freesurfer.recon import recon_all_status, run_recon_all
from medmcp_neuro_core.tools.freesurfer.stats import freesurfer_stats
from medmcp_neuro_core.tools.mask import select_mask_of_interest
from medmcp_neuro_core.tools.registration import apply_transform, coregister, register_to_template
from medmcp_neuro_core.tools.segmentation import list_brain_segmentation_labels, segment_brain
from medmcp_neuro_core.tools.skull_strip import skull_strip, warmup
from medmcp_neuro_core.tools.super_resolution import super_resolve

mcp = FastMCP("medmcp-neuro-core")

mcp.add_tool(skull_strip, **tool_effect("writes_new"))
mcp.add_tool(warmup, **tool_effect("read_only"))
mcp.add_tool(register_to_template, **tool_effect("writes_new"))
mcp.add_tool(coregister, **tool_effect("writes_new"))
mcp.add_tool(apply_transform, **tool_effect("writes_new"))
mcp.add_tool(segment_brain, **tool_effect("writes_new"))
mcp.add_tool(list_brain_segmentation_labels, **tool_effect("read_only"))
mcp.add_tool(select_mask_of_interest, **tool_effect("writes_new"))
mcp.add_tool(super_resolve, **tool_effect("writes_new"))
mcp.add_tool(list_atlases, **tool_effect("read_only"))
mcp.add_tool(lesion_load_by_atlas, **tool_effect("writes_new"))
mcp.add_tool(lesion_frequency_map, **tool_effect("writes_new"))
# FreeSurfer 8.2.0 (needs the user's licence): recon-all as a background job, the
# status of recon-all and FastSurfer-surface runs, and their stats.
mcp.add_tool(run_recon_all, **tool_effect("writes_new"))
mcp.add_tool(recon_all_status, **tool_effect("read_only"))
mcp.add_tool(freesurfer_stats, **tool_effect("writes_new"))


def server_config() -> dict[str, object]:
    """Return MCP server metadata for autodiscovery by the local agent."""
    return {
        "name": "medmcp-neuro-core",
        "command": "medmcp-neuro-core",
        "tool_timeout_sec": 7200.0,
        "skills_path": str(_pkg_files("medmcp_neuro_core") / "skills"),
    }


def main() -> None:
    """Launch the MCP server over stdio (JSON-RPC)."""
    mcp.run(transport="stdio")
