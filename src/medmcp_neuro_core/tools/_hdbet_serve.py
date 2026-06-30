"""Persistent HD-BET worker process (subprocess entry point; excluded from pyright).

Loads the nnU-Net predictor once and serves skull-strip requests in a loop, so the
~10 s of torch import + CUDA init + model load is paid a single time instead of on
every call. The isolation that the per-call subprocess provided is preserved: this
process's stdout/stderr (nnU-Net's noisy output) are redirected to a log by the
parent, requests arrive as JSON lines on stdin, and one small status JSON per
request is written to the dedicated response fd named by ``MEDMCP_HDBET_RESP_FD`` —
so nnU-Net output never reaches the MCP stdio channel.

Protocol (one JSON object per line):
    parent -> stdin:     {"input_path", "stem", "brain_path"}
    worker -> resp fd:   {"ready": true} once loaded (or {"ready": false, "error"})
                         then {"ok": true} | {"ok": false, "error"} per request
The worker exits when stdin reaches EOF (the parent died), freeing the model/VRAM.
"""

import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Load the predictor once, then serve skull-strip requests until stdin EOF."""
    resp = os.fdopen(int(os.environ["MEDMCP_HDBET_RESP_FD"]), "w")
    device = os.environ.get("MEDMCP_HDBET_DEVICE", "cpu")
    use_tta = os.environ.get("MEDMCP_HDBET_TTA", "0") == "1"

    def reply(obj: dict[str, object]) -> None:
        """Write one status object to the dedicated response fd."""
        resp.write(json.dumps(obj) + "\n")
        resp.flush()

    try:
        import torch
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor

        maybe_download_parameters()
        predictor = get_hdbet_predictor(use_tta=use_tta, device=torch.device(device), verbose=False)
    except Exception as exc:  # model load failed — report and exit
        reply({"ready": False, "error": repr(exc)})
        return

    reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            with tempfile.TemporaryDirectory() as tmp_dir:
                mask_truncated = str(Path(tmp_dir) / f"{req['stem']}_mask")
                predictor.predict_from_files(
                    [[req["input_path"]]],
                    [mask_truncated],
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )
                apply_bet(req["input_path"], mask_truncated + ".nii.gz", req["brain_path"])
            reply({"ok": True})
        except Exception as exc:
            reply({"ok": False, "error": repr(exc)})


if __name__ == "__main__":
    main()
