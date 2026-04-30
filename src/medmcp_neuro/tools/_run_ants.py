"""ANTsPy subprocess entry point, isolated from MCP stdio pipes.

Invoked by registration tools via subprocess.run so that ANTsPy's ITK output
does not pollute the MCP JSON-RPC stdout pipe.

Supported operations (dispatched from the ``operation`` key in the JSON payload
read from stdin):
    - ``register``: run ants.registration and write the warped image to disk.
    - ``apply_transforms``: run ants.apply_transforms and write the output image.
"""

import json
import sys


def _register(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]

    fixed = ants.image_read(str(args["fixed_path"]))
    moving = ants.image_read(str(args["moving_path"]))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=str(args["type_of_transform"]),
        outprefix=str(args["outprefix"]),
    )

    ants.image_write(reg["warpedmovout"], str(args["out_registered"]))

    fwd: list[str] = [str(t) for t in reg["fwdtransforms"]]
    inv: list[str] = [str(t) for t in reg["invtransforms"]]
    # Affines (.mat) in invtransforms must be inverted when applied; displacement
    # fields (.nii.gz) are the actual inverse field and must not be re-inverted.
    inv_flags: list[bool] = [str(t).endswith(".mat") for t in inv]

    return {
        "ok": True,
        "fwdtransforms": fwd,
        "invtransforms": inv,
        "inverse_invert_flags": inv_flags,
    }


def _apply_transforms(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]

    ref = ants.image_read(str(args["reference_path"]))
    moving = ants.image_read(str(args["input_path"]))
    transforms: list[str] = [str(t) for t in args["transforms"]]  # type: ignore[union-attr]

    raw_flags = args.get("invert_flags")
    invert_flags: list[bool] | None = (
        [bool(f) for f in raw_flags]  # type: ignore[union-attr]
        if raw_flags is not None
        else None
    )

    _interp_map = {"Linear": "linear", "NearestNeighbor": "nearestNeighbor", "BSpline": "bSpline"}
    interpolator = _interp_map.get(str(args["interpolator"]), "linear")

    result = ants.apply_transforms(
        fixed=ref,
        moving=moving,
        transformlist=transforms,
        interpolator=interpolator,
        whichtoinvert=invert_flags,
    )

    ants.image_write(result, str(args["out_path"]))
    return {"ok": True}


if __name__ == "__main__":
    _args: dict[str, object] = json.loads(sys.stdin.read())

    # ANTsPy/ITK uses print() for progress output which would corrupt the JSON
    # result read by registration.py. Redirect Python stdout → stderr for the
    # duration of the call so all ITK output goes to stderr; restore before
    # printing the result.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        _op = str(_args["operation"])
        if _op == "register":
            _result = _register(_args)
        elif _op == "apply_transforms":
            _result = _apply_transforms(_args)
        else:
            _result = {"ok": False, "error": f"Unknown operation: {_op}"}
    except Exception as exc:
        _result = {"ok": False, "error": str(exc)}
    finally:
        sys.stdout = _real_stdout

    print(json.dumps(_result))
    if not _result.get("ok"):
        sys.exit(1)
