"""Fake HD-BET worker for tests: speaks the _hdbet_serve protocol without torch.

Honours two env knobs: FAKE_FAIL_READY=1 reports a load failure; FAKE_FAIL_RUN=1
reports a per-request failure. Otherwise it touches each request's brain_path and
replies ok.
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    """Reply ready, then echo a status per stdin request until EOF."""
    resp = os.fdopen(int(os.environ["MEDMCP_HDBET_RESP_FD"]), "w")

    def reply(obj: dict[str, object]) -> None:
        resp.write(json.dumps(obj) + "\n")
        resp.flush()

    if os.environ.get("FAKE_FAIL_READY") == "1":
        reply({"ready": False, "error": "boom"})
        return
    reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if os.environ.get("FAKE_FAIL_RUN") == "1":
            reply({"ok": False, "error": "inference boom"})
            continue
        Path(req["brain_path"]).write_bytes(b"brain")
        reply({"ok": True})


if __name__ == "__main__":
    main()
