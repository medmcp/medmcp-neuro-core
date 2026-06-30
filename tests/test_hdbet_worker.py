"""Tests for the persistent HD-BET worker client and skull_strip integration.

Drive a real worker subprocess (the torch-free ``fake_hdbet_worker``) so the IPC
protocol, reuse, fallback, and failure paths are exercised without HD-BET.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from medmcp_neuro_core.tools import _hdbet_worker, skull_strip
from medmcp_neuro_core.tools._hdbet_worker import (
    HdbetWorker,
    WorkerError,
    get_worker,
    shutdown_worker,
)

_FAKE = Path(__file__).parent / "fake_hdbet_worker.py"
_FAKE_CMD = [sys.executable, str(_FAKE)]


@pytest.fixture(autouse=True)
def _reset_worker() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse fixture
    """Ensure the module-level worker singleton never leaks between tests."""
    yield
    shutdown_worker()


def test_worker_start_run_close(tmp_path: Path) -> None:
    """A worker starts, serves a request (writing brain_path), then closes."""
    worker = HdbetWorker("cpu", use_tta=False, command=_FAKE_CMD)
    worker.start()
    try:
        assert worker.alive
        brain = tmp_path / "out.nii.gz"
        worker.run(str(tmp_path / "in.nii.gz"), "in", str(brain))
        assert brain.read_bytes() == b"brain"
    finally:
        worker.aclose()
    assert not worker.alive


def test_worker_start_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load failure surfaces as WorkerError from start()."""
    monkeypatch.setenv("FAKE_FAIL_READY", "1")
    worker = HdbetWorker("cpu", use_tta=False, command=_FAKE_CMD)
    with pytest.raises(WorkerError):
        worker.start()


def test_worker_run_failure_raises_runtimeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A genuine inference failure raises RuntimeError (not a fallback signal)."""
    monkeypatch.setenv("FAKE_FAIL_RUN", "1")
    worker = HdbetWorker("cpu", use_tta=False, command=_FAKE_CMD)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="HD-BET failed"):
            worker.run(str(tmp_path / "in.nii.gz"), "in", str(tmp_path / "out.nii.gz"))
    finally:
        worker.aclose()


def test_dead_worker_run_raises_workererror(tmp_path: Path) -> None:
    """If the worker died, run() raises WorkerError so the caller can fall back."""
    worker = HdbetWorker("cpu", use_tta=False, command=_FAKE_CMD)
    worker.start()
    worker.aclose()  # simulate death
    with pytest.raises(WorkerError):
        worker.run(str(tmp_path / "in.nii.gz"), "in", str(tmp_path / "out.nii.gz"))


def test_get_worker_reuses_then_switches_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_worker reuses a matching worker and replaces one with a different config."""
    monkeypatch.setattr(_hdbet_worker, "_default_worker_command", lambda: _FAKE_CMD)
    first = get_worker("cpu", use_tta=False, start=True)
    assert first is not None
    again = get_worker("cpu", use_tta=False, start=True)
    assert again is first  # reused
    switched = get_worker("cuda", use_tta=True, start=True)
    assert switched is not None and switched is not first
    assert not first.alive  # the old one was torn down


def test_get_worker_no_start_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """With start=False and no warm worker, get_worker returns None."""
    monkeypatch.setattr(_hdbet_worker, "_default_worker_command", lambda: _FAKE_CMD)
    assert get_worker("cpu", use_tta=False, start=False) is None


def test_skull_strip_uses_warm_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """skull_strip uses the warm worker (not the subprocess) when one is available."""
    monkeypatch.setattr(_hdbet_worker, "_default_worker_command", lambda: _FAKE_CMD)
    monkeypatch.setenv("MEDMCP_HDBET_PERSIST", "1")
    inp = tmp_path / "t1.nii.gz"
    inp.touch()
    with patch(
        "medmcp_neuro_core.tools.skull_strip.subprocess.run",
        side_effect=AssertionError("should have used the worker, not the subprocess"),
    ):
        result = skull_strip.skull_strip(inp, device="cpu")
    assert Path(result["brain_path"]).exists()


def test_skull_strip_falls_back_when_worker_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker that fails mid-run falls back to the subprocess path."""
    monkeypatch.setattr(_hdbet_worker, "_default_worker_command", lambda: _FAKE_CMD)
    monkeypatch.setenv("MEDMCP_HDBET_PERSIST", "1")

    def _boom(self: HdbetWorker, *_args: str) -> None:
        raise WorkerError("died mid-run")

    monkeypatch.setattr(HdbetWorker, "run", _boom)

    called: dict[str, bool] = {"subprocess": False}

    def _fake_subprocess(*_args: object, **_kwargs: object) -> object:
        called["subprocess"] = True
        raise RuntimeError("subprocess fallback reached")  # enough to prove the path

    inp = tmp_path / "t1.nii.gz"
    inp.touch()
    with (
        patch("medmcp_neuro_core.tools.skull_strip.subprocess.run", side_effect=_fake_subprocess),
        pytest.raises(RuntimeError, match="subprocess fallback reached"),
    ):
        skull_strip.skull_strip(inp, device="cpu")
    assert called["subprocess"] is True


def test_warmup_starts_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warmup pre-starts a worker so a later skull_strip reuses it."""
    monkeypatch.setattr(_hdbet_worker, "_default_worker_command", lambda: _FAKE_CMD)
    out = skull_strip.warmup(device="cpu")
    assert out == {"ok": True, "device": "cpu", "warmed": True}
    assert get_worker("cpu", use_tta=False, start=False) is not None
