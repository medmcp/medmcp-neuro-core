"""segment_brain / run_recon_all through a real MCP client session (in memory).

The client validates each structured result against the tool's advertised output
schema, as MedMCP's core does; a result that does not fit would fail the call. The
cases are the ones this stack's job-slot and licence handling adds.
"""

from pathlib import Path
from typing import Any

import jsonschema
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from medmcp_neuro_core.server import mcp
from tests.conftest import write_running_job
from tests.test_fs_surfaces import (
    fake_fastsurfer,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)


async def _call(tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """(structured result, the text content the model reads)."""
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        tools = {t.name: t for t in (await client.list_tools()).tools}
        schema = tools[tool].outputSchema
        assert schema is not None, f"{tool} advertises no output schema"
        # call_tool validates structuredContent against the schema listed above.
        result = await client.call_tool(tool, arguments)
        assert not result.isError, result.content
        structured = result.structuredContent
        assert isinstance(structured, dict)
        jsonschema.validate(structured, schema)
        text = "".join(getattr(c, "text", "") for c in result.content)
        return structured, text


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_fastsurfer")
async def test_segment_brain_waiting_for_job_slot(
    workspace: Path, licence: Path, t1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Busy slot: a valid result with the segmentation and surfaces waiting."""
    monkeypatch.setenv("MEDMCP_JOBS", "1")
    busy = write_running_job(workspace)
    out, text = await _call(
        "segment_brain",
        {
            "input_path": str(t1),
            "output_dir": str(workspace / "out"),
            "device": "cpu",
            "surfaces": True,
            "license_path": str(licence),
        },
    )
    assert out["surfaces"]["status"] == "waiting_for_job_slot"
    assert out["surfaces"]["running_job"]["job_id"] == busy
    assert out["medmcp_job"] is None
    # The render rules reach the model in the text content.
    assert "waiting_for_job_slot" in text and "NOT a licence problem" in text


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_fastsurfer")
async def test_segment_brain_job_request(
    workspace: Path, licence: Path, t1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free slot: a valid result carrying the job request for the core."""
    monkeypatch.setenv("MEDMCP_JOBS", "1")
    out, _ = await _call(
        "segment_brain",
        {
            "input_path": str(t1),
            "output_dir": str(workspace / "out"),
            "device": "cpu",
            "surfaces": True,
        },
    )
    assert out["surfaces"]["status"] == "requested"
    assert out["medmcp_job"] == {"name": "FastSurfer surfaces sub-01_T1w"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_fastsurfer")
async def test_segment_brain_licence_invalid(workspace: Path, t1: Path) -> None:
    """Malformed licence: a valid result, licence_invalid, segmentation still done."""
    bad = workspace / "keys" / "license.txt"
    bad.parent.mkdir()
    bad.write_text("nope\n")
    out, _ = await _call(
        "segment_brain",
        {
            "input_path": str(t1),
            "output_dir": str(workspace / "out"),
            "device": "cpu",
            "surfaces": True,
            "license_path": str(bad),
        },
    )
    assert out["surfaces"]["status"] == "licence_invalid"
    assert Path(out["seg_path"]).exists()


@pytest.mark.asyncio
async def test_run_recon_all_waiting_for_job_slot(
    workspace: Path, licence: Path, t1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Busy slot: a valid result, no medmcp_job, the running job and the retry call."""
    monkeypatch.setenv("MEDMCP_JOBS", "1")
    busy = write_running_job(workspace)
    out, text = await _call(
        "run_recon_all", {"t1_path": str(t1), "output_dir": str(workspace / "fs")}
    )
    assert out["status"] == "waiting_for_job_slot"
    assert out["running_job"]["job_id"] == busy
    assert out["retry_call"]["tool"] == "run_recon_all"
    assert "medmcp_job" not in out and "_render" in out
    assert "NOT a licence problem" in text


@pytest.mark.asyncio
async def test_run_recon_all_licence_invalid(workspace: Path, t1: Path, tmp_path: Path) -> None:
    """A licence outside the workspace: a valid licence_invalid result, nothing started."""
    outside = tmp_path / "license.txt"
    outside.write_text("someone@example.org\n1\n *K\n")
    out, _ = await _call(
        "run_recon_all",
        {
            "t1_path": str(t1),
            "output_dir": str(workspace / "fs"),
            "license_path": str(outside),
        },
    )
    assert out["status"] == "licence_invalid"
    assert "outside the workspace" in out["error"]
    assert "medmcp_job" not in out
