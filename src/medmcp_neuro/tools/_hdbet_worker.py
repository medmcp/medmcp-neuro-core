"""Client for the persistent HD-BET worker (see ``_hdbet_serve``).

Manages one long-lived worker subprocess keyed by ``(device, use_tta)``: starts it,
forwards skull-strip requests over its stdin, reads a status line from its dedicated
response pipe, and reuses it across calls so the model loads once. A dead/unavailable
worker raises :class:`WorkerError` so the caller (``skull_strip``) can fall back to
the per-call subprocess; a genuine inference failure raises ``RuntimeError``.

The worker is opt-in: ``skull_strip`` only auto-starts one when ``MEDMCP_HDBET_PERSIST``
is set, but it always *reuses* a worker already started by the ``warmup`` tool (the
workspace pre-warm hook). Default behaviour is unchanged — the subprocess path.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from typing import cast

# First-ever run may download model parameters; subsequent loads are ~10 s.
_START_TIMEOUT_SEC: float = 600.0
# Inference upper bound (mirrors the per-call subprocess timeout).
_RUN_TIMEOUT_SEC: float = 3600.0
_WORKER_MODULE: str = "medmcp_neuro.tools._hdbet_serve"


class WorkerError(RuntimeError):
    """The persistent worker is unavailable (failed to start, died, or timed out)."""


def _default_worker_command() -> list[str]:
    """Command that launches the worker process (overridable in tests)."""
    return [sys.executable, "-m", _WORKER_MODULE]


def _parse(line: str) -> dict[str, object]:
    """Parse one status line into a dict (empty dict on garbage)."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, object]", obj) if isinstance(obj, dict) else {}


class HdbetWorker:
    """A persistent HD-BET worker subprocess, keyed by ``(device, use_tta)``."""

    def __init__(self, device: str, use_tta: bool, command: list[str] | None = None) -> None:
        """Create (but do not start) a worker for *device*/*use_tta*."""
        self.device = device
        self.use_tta = use_tta
        self._command = command if command is not None else _default_worker_command()
        self._proc: subprocess.Popen[str] | None = None
        self._resp_fd: int | None = None
        self._log_path: str | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Spawn the worker and block until it reports the model is loaded.

        Raises:
            WorkerError: the worker failed to start or load the model.
        """
        resp_r, resp_w = os.pipe()
        log_fd, self._log_path = tempfile.mkstemp(prefix="hdbet-worker-", suffix=".log")
        env = {
            **os.environ,
            "MEDMCP_HDBET_RESP_FD": str(resp_w),
            "MEDMCP_HDBET_DEVICE": self.device,
            "MEDMCP_HDBET_TTA": "1" if self.use_tta else "0",
        }
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                pass_fds=(resp_w,),
            )
        except BaseException:
            os.close(resp_r)
            raise
        finally:
            os.close(resp_w)  # parent keeps only the read end (so EOF reaches the worker)
            os.close(log_fd)
        self._resp_fd = resp_r

        msg = _parse(self._read_line(_START_TIMEOUT_SEC))
        if not msg.get("ready"):
            self.aclose()
            err = msg.get("error", "no ready signal")
            raise WorkerError(f"hdbet worker failed to start: {err}")

    def run(self, input_path: str, stem: str, brain_path: str) -> None:
        """Run one skull strip on the warm predictor; writes *brain_path*.

        Raises:
            WorkerError: the worker died, timed out, or could not be written to.
            RuntimeError: the inference itself failed (a real error, not a fallback).
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise WorkerError("hdbet worker is not running")
            request = json.dumps({"input_path": input_path, "stem": stem, "brain_path": brain_path})
            try:
                proc.stdin.write(request + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise WorkerError(f"hdbet worker write failed: {exc}") from exc
            msg = _parse(self._read_line(_RUN_TIMEOUT_SEC))
            if not msg.get("ok"):
                raise RuntimeError(f"HD-BET failed: {msg.get('error', 'unknown')}")

    @property
    def alive(self) -> bool:
        """Whether the worker process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def aclose(self) -> None:
        """Stop the worker (closing stdin makes it exit), idempotent."""
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.stdin is not None:
                with suppress(Exception):
                    proc.stdin.close()  # EOF → worker's request loop ends
            with suppress(Exception):
                proc.wait(timeout=10)
            if proc.poll() is None:
                with suppress(Exception):
                    proc.kill()
        if self._resp_fd is not None:
            with suppress(OSError):
                os.close(self._resp_fd)
            self._resp_fd = None

    def _read_line(self, timeout: float) -> str:
        """Read one newline-terminated status line from the response pipe, bounded."""
        fd = self._resp_fd
        if fd is None:
            raise WorkerError("hdbet worker has no response channel")
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while not buf.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("hdbet worker timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise WorkerError("hdbet worker closed the response channel (died)")
            buf.extend(chunk)
        return buf.decode()


# ── module-level singleton (one warm worker reused across calls) ───────────────

_worker: HdbetWorker | None = None
_worker_lock = threading.Lock()


def persist_enabled() -> bool:
    """Whether ``skull_strip`` may auto-start a persistent worker (default off)."""
    return os.environ.get("MEDMCP_HDBET_PERSIST", "").strip().lower() in {"1", "true", "yes", "on"}


def get_worker(device: str, use_tta: bool, *, start: bool = False) -> HdbetWorker | None:
    """Return the warm worker for *device*/*use_tta*, starting one if *start*.

    Reuses a live worker that matches the config; a mismatched or dead one is torn
    down first. Returns ``None`` if no worker exists and *start* is false, or if a
    requested start failed (the caller then falls back to the subprocess path).
    """
    global _worker
    with _worker_lock:
        current = _worker
        if current is not None:
            if current.alive and current.device == device and current.use_tta == use_tta:
                return current
            current.aclose()  # dead, or a different device/tta → replace
            _worker = None
        if not start:
            return None
        worker = HdbetWorker(device, use_tta)
        try:
            worker.start()
        except WorkerError:
            worker.aclose()
            return None
        _worker = worker
        return worker


def shutdown_worker() -> None:
    """Tear down the warm worker, if any (registered at exit; freeing the model)."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.aclose()
            _worker = None


atexit.register(shutdown_worker)
