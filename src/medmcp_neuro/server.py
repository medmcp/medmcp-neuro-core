"""MCP server entrypoint for medmcp-neuro."""

from mcp.server.fastmcp import FastMCP

from medmcp_neuro.tools.skull_strip import skull_strip

mcp = FastMCP("medmcp-neuro")

mcp.add_tool(skull_strip)


def server_config() -> dict[str, object]:
    """Return MCP server metadata for autodiscovery by the local agent."""
    return {
        "name": "medmcp-neuro",
        "command": "medmcp-neuro",
        "tool_timeout_sec": 1800.0,
    }


def main() -> None:
    """Launch the MCP server over stdio (JSON-RPC)."""
    mcp.run(transport="stdio")
