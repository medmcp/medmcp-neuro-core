"""Smoke tests — the template must pass CI out of the box."""

import pytest

from medmcp_template import __version__
from medmcp_template.server import mcp


def test_version_is_string() -> None:
    """The package exposes a ``__version__`` string."""
    assert isinstance(__version__, str)
    assert __version__


@pytest.mark.asyncio
async def test_server_has_tools() -> None:
    """The MCP server exposes at least one registered tool."""
    tools = await mcp.list_tools()
    assert len(tools) > 0, "No tools registered on the MCP server"
