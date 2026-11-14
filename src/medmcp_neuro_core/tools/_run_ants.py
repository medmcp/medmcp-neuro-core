"""ANTsPy subprocess entry point, isolated from MCP stdio pipes.

Invoked by registration tools via subprocess.run so that ANTsPy's ITK output
does not pollute the MCP JSON-RPC stdout pipe. The result dict is written to a
tempfile path supplied via the stdin payload rather than stdout, to avoid
contamination from ITK's own output.

Supported operations (dispatched from the ``operation`` key in the JSON payload
read from stdin):
    - ``register``: run ants.registration and write the warped image to disk.
    - ``apply_transforms``: run ants.apply_transforms and write the output image.
    - ``atlas_overlap``: bring atlas maps onto a lesion mask's grid (``transforms`` from
      template to mask space, or an identity resample when none are given) and return
      per-region / per-tract overlap rows computed by ``_overlap``.
    - ``masks_to_template``: bring lesion masks onto the template grid (nearest neighbour).
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


def _to_grid(ants, moving, fixed, transforms, invert_flags, interpolator):  # type: ignore[no-untyped-def]
    """Resample ``moving`` onto ``fixed``: through ``transforms``, or identity if none."""
    if transforms:
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transforms,
            interpolator=interpolator,
            whichtoinvert=invert_flags,
        )
    return ants.resample_image_to_target(moving, fixed, interp_type=interpolator)


def _atlas_overlap(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]
    import numpy as np

    from medmcp_neuro_core.tools._overlap import label_rows, lesion_voxels, probabilistic_row

    lesion_img = ants.image_read(str(args["lesion_path"]))
    if lesion_img.dimension != 3:
        raise ValueError(f"Lesion mask must be 3-D, got {lesion_img.dimension}-D")
    lesion = lesion_voxels(lesion_img.numpy())
    voxel_mm3 = float(np.prod(lesion_img.spacing))
    transforms = [str(t) for t in args.get("transforms") or []]  # type: ignore[union-attr]
    raw_flags = args.get("invert_flags")
    flags = [bool(f) for f in raw_flags] if raw_flags else None  # type: ignore[union-attr]
    save_dir = args.get("save_dir")
    entries: list[dict[str, object]] = list(args["entries"])  # type: ignore[call-overload]

    if args["kind"] == "labels":
        dseg = ants.image_read(str(args["dseg_path"]))
        warped = _to_grid(ants, dseg, lesion_img, transforms, flags, "nearestNeighbor")
        if save_dir:
            ants.image_write(warped, str(save_dir))
        table = [(int(e["index"]), str(e["name"])) for e in entries]  # type: ignore[call-overload]
        rows = label_rows(lesion, warped.numpy(), voxel_mm3, table)
    else:
        threshold = float(args["threshold"])  # type: ignore[arg-type]
        rows = []
        for e in entries:
            tract = ants.image_read(str(e["path"]))
            warped = _to_grid(ants, tract, lesion_img, transforms, flags, "linear")
            if save_dir:
                ants.image_write(warped, str(save_dir) + f"/{e['file_stem']}.nii.gz")
            row = probabilistic_row(str(e["name"]), lesion, warped.numpy(), voxel_mm3, threshold)
            rows.append(row)

    return {
        "ok": True,
        "rows": rows,
        "voxel_mm3": voxel_mm3,
        "lesion_volume_mm3": float(lesion.sum()) * voxel_mm3,
    }


def _masks_to_template(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]

    ref = ants.image_read(str(args["reference_path"]))
    for item in args["items"]:  # type: ignore[union-attr]
        moving = ants.image_read(str(item["input_path"]))
        transforms = [str(t) for t in item.get("transforms") or []]
        warped = _to_grid(ants, moving, ref, transforms, None, "nearestNeighbor")
        ants.image_write(warped, str(item["out_path"]))
    return {"ok": True}


if __name__ == "__main__":
    _args: dict[str, object] = json.loads(sys.stdin.read())
    _result_path = str(_args["result_path"])

    try:
        _op = str(_args["operation"])
        if _op == "register":
            _result = _register(_args)
        elif _op == "apply_transforms":
            _result = _apply_transforms(_args)
        elif _op == "atlas_overlap":
            _result = _atlas_overlap(_args)
        elif _op == "masks_to_template":
            _result = _masks_to_template(_args)
        else:
            _result = {"ok": False, "error": f"Unknown operation: {_op}"}
    except Exception as exc:
        _result = {"ok": False, "error": str(exc)}

    with open(_result_path, "w") as f:
        json.dump(_result, f)

    if not _result.get("ok"):
        sys.exit(1)
