"""HD-BET inference subprocess entry point, isolated from MCP stdio pipes.

Invoked by skull_strip via subprocess.run with capture_output=True so that
none of the MCP file descriptors are inherited by nnU-Net's multiprocessing
workers.  The result dict is written to a tempfile path supplied via stdin
rather than stdout, to avoid contamination from nnU-Net's own output.
"""

import json
import sys
import tempfile
from pathlib import Path


def _run(device: str, use_tta: bool, input_path: str, stem: str, brain_path: str) -> None:
    import torch
    from HD_BET.checkpoint_download import (  # type: ignore[import-untyped]
        maybe_download_parameters,
    )
    from HD_BET.hd_bet_prediction import (  # type: ignore[import-untyped]
        apply_bet,  # pyright: ignore[reportUnknownVariableType]
        get_hdbet_predictor,
    )

    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=use_tta, device=torch.device(device), verbose=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_truncated = str(Path(tmp_dir) / f"{stem}_mask")
        mask_path = mask_truncated + ".nii.gz"
        predictor.predict_from_files(
            [[input_path]],
            [mask_truncated],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,  # type: ignore[arg-type]
            num_parts=1,
            part_id=0,
        )
        apply_bet(input_path, mask_path, brain_path)


if __name__ == "__main__":
    _args: dict[str, object] = json.loads(sys.stdin.read())
    _result_path = str(_args["result_path"])

    try:
        _run(
            device=str(_args["device"]),
            use_tta=bool(_args["use_tta"]),
            input_path=str(_args["input_path"]),
            stem=str(_args["stem"]),
            brain_path=str(_args["brain_path"]),
        )
        _result: dict[str, object] = {"ok": True}
    except Exception as exc:
        _result = {"ok": False, "error": str(exc)}

    with open(_result_path, "w") as f:
        json.dump(_result, f)

    if not _result.get("ok"):
        sys.exit(1)
