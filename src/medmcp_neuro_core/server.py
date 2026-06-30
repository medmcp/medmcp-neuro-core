"""MCP server entrypoint for medmcp-neuro-core."""

from importlib.resources import files as _pkg_files

from mcp.server.fastmcp import FastMCP

from medmcp_neuro_core.tools.registration import apply_transform, coregister, register_to_template
from medmcp_neuro_core.tools.segmentation import list_brain_segmentation_labels, segment_brain
from medmcp_neuro_core.tools.skull_strip import skull_strip, warmup

mcp = FastMCP("medmcp-neuro-core")

mcp.add_tool(skull_strip)
mcp.add_tool(warmup)
mcp.add_tool(register_to_template)
mcp.add_tool(coregister)
mcp.add_tool(apply_transform)
mcp.add_tool(segment_brain)
mcp.add_tool(list_brain_segmentation_labels)


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
