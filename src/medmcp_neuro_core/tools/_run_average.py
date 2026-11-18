# pyright: basic
"""ANTsPy subprocess entry point for ``average_repeats``, isolated from MCP stdio pipes.

Same protocol as ``_run_ants``: a JSON payload on stdin, the result dict written to
the ``result_path`` it names (never to stdout, which ITK may write to).

Operations:
    - ``quick_rigid``: a fast, coarse rigid registration of every scan to the first,
      for the motion estimate. Returns each scan's 4x4 transform (ANTs physical
      space, mapping points of the first scan to the scan) and the first scan's
      centre of mass in the same coordinates.
    - ``rigid_to_first``: full rigid registration of every other scan to the first,
      resampled (linear) onto the first scan's grid and written to ``out_dir``; the
      transforms are copied to the given paths.
"""

import json
import shutil
import sys
import tempfile


def _matrix(ants, path: str) -> list[list[float]]:  # type: ignore[no-untyped-def]
    """The 4x4 matrix of an ANTs affine transform file (y = A(x - c) + c + t)."""
    import numpy as np

    tx = ants.read_transform(path)
    params = np.asarray(tx.parameters, dtype=float)
    centre = np.asarray(tx.fixed_parameters, dtype=float)
    a = params[:9].reshape(3, 3)
    t = params[9:12]
    m = np.eye(4)
    m[:3, :3] = a
    m[:3, 3] = t + centre - a @ centre
    return m.tolist()


def _quick_rigid(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]
    import numpy as np

    paths = [str(p) for p in args["paths"]]  # type: ignore[union-attr]
    fixed = ants.image_read(paths[0])
    matrices: list[list[list[float]]] = [np.eye(4).tolist()]
    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(paths[1:], start=1):
            moving = ants.image_read(path)
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="Rigid",
                aff_iterations=(100, 50, 20),
                aff_shrink_factors=(4, 2, 1),
                aff_smoothing_sigmas=(2, 1, 0),
                outprefix=f"{tmp}/scan{i}_",
                random_seed=1,
            )
            matrices.append(_matrix(ants, str(reg["fwdtransforms"][0])))
    centre = [float(v) for v in ants.get_center_of_mass(fixed)]
    return {"ok": True, "matrices": matrices, "centre": centre}


def _rigid_to_first(args: dict[str, object]) -> dict[str, object]:
    import ants  # type: ignore[import-untyped]

    paths = [str(p) for p in args["paths"]]  # type: ignore[union-attr]
    warped_paths = [str(p) for p in args["warped_paths"]]  # type: ignore[union-attr]
    transform_paths = [str(p) for p in args["transform_paths"]]  # type: ignore[union-attr]
    fixed = ants.image_read(paths[0])
    ants.image_write(fixed, warped_paths[0])
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, len(paths)):
            moving = ants.image_read(paths[i])
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="Rigid",
                outprefix=f"{tmp}/scan{i}_",
                random_seed=1,
            )
            ants.image_write(reg["warpedmovout"], warped_paths[i])
            shutil.copyfile(str(reg["fwdtransforms"][0]), transform_paths[i])
    return {"ok": True}


if __name__ == "__main__":
    _args: dict[str, object] = json.loads(sys.stdin.read())
    _result_path = str(_args["result_path"])
    try:
        _op = str(_args["operation"])
        if _op == "quick_rigid":
            _result = _quick_rigid(_args)
        elif _op == "rigid_to_first":
            _result = _rigid_to_first(_args)
        else:
            _result = {"ok": False, "error": f"Unknown operation: {_op}"}
    except Exception as exc:
        _result = {"ok": False, "error": str(exc)}
    with open(_result_path, "w") as f:
        json.dump(_result, f)
    if not _result.get("ok"):
        sys.exit(1)
