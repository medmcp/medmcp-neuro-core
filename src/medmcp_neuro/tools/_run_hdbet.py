"""HD-BET inference subprocess entry point, isolated from MCP stdio pipes.

Invoked by skull_strip via subprocess.run with stdin=DEVNULL so that none of
the MCP file descriptors are inherited by nnU-Net's multiprocessing workers.
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

    # nnU-Net uses print() for progress, which would corrupt the JSON result that
    # skull_strip.py reads from stdout. Redirect Python stdout → stderr for the
    # duration of inference so all nnU-Net output goes to stderr; restore before
    # printing the result.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
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
    finally:
        sys.stdout = _real_stdout

    print(json.dumps(_result))
    if not _result.get("ok"):
        sys.exit(1)
