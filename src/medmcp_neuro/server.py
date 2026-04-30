"""MCP server entrypoint for medmcp-neuro."""

from importlib.resources import files as _pkg_files

from mcp.server.fastmcp import FastMCP

from medmcp_neuro.tools.registration import apply_transform, coregister, register_to_template
from medmcp_neuro.tools.skull_strip import skull_strip

mcp = FastMCP("medmcp-neuro")

mcp.add_tool(skull_strip)
mcp.add_tool(register_to_template)
mcp.add_tool(coregister)
mcp.add_tool(apply_transform)


def server_config() -> dict[str, object]:
    """Return MCP server metadata for autodiscovery by the local agent."""
    return {
        "name": "medmcp-neuro",
        "command": "medmcp-neuro",
        "tool_timeout_sec": 7200.0,
        "skills_path": str(_pkg_files("medmcp_neuro") / "skills"),
    }


def main() -> None:
    """Launch the MCP server over stdio (JSON-RPC)."""
    mcp.run(transport="stdio")
