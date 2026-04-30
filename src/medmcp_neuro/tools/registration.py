"""Image registration tools using ANTsPy."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

from medmcp_neuro.tools._template import get_mni152_1mm

_MNI_SPACE = "MNI152NLin2009cAsym"
_TIMEOUT_FAST = 3600  # rigid / similarity / affine / synquick (≤1 h)
_TIMEOUT_SYN = 7200  # full SyN                               (≤2 h)
_TRANSFORM_MAP: dict[str, str] = {
    "rigid": "Rigid",
    "similarity": "Similarity",
    "affine": "Affine",
    "synquick": "SyNQuick",
    "syn": "SyN",
}
_TRANSFORM_TYPE = Literal["rigid", "similarity", "affine", "synquick", "syn"]


# ── TypedDicts ─────────────────────────────────────────────────────────────────


class RegisterToTemplateChoiceResult(TypedDict):
    """Returned when the caller must confirm transform type before running."""

    available_transforms: list[str]
    recommended_transform: str
    _render: str


class RegisterToTemplateResult(TypedDict):
    """Successful template-space normalisation result."""

    registered_path: str
    transform_prefix: str
    forward_transforms: list[str]
    inverse_transforms: list[str]
    inverse_invert_flags: list[bool]
    template_path: str
    transform_type: str
    _render: str


class CoregisterChoiceResult(TypedDict):
    """Returned when the caller must confirm transform type before running."""

    available_transforms: list[str]
    recommended_transform: str
    _render: str


class CoregisterResult(TypedDict):
    """Successful within-subject coregistration result."""

    registered_paths: list[str]
    transform_prefixes: list[str]
    forward_transforms_list: list[list[str]]
    inverse_transforms_list: list[list[str]]
    inverse_invert_flags_list: list[list[bool]]
    fixed_path: str
    transform_type: str
    _render: str


class ApplyTransformResult(TypedDict):
    """Successful transform-application result."""

    output_path: str
    _render: str


# ── helpers ────────────────────────────────────────────────────────────────────


def _nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _space_label(path: Path) -> str:
    """Extract the last BIDS entity from a NIfTI filename as a space label."""
    stem = _nii_stem(path)
    return stem.rsplit("_", 1)[-1]


def _check_antspy() -> None:
    """Raise RuntimeError with install instructions if antspyx is not importable."""
    if importlib.util.find_spec("ants") is None:
        raise RuntimeError(
            "antspyx is not installed. Install it with:\n"
            "  pip install antspyx\n"
            "or in a uv environment:\n"
            "  uv add antspyx"
        )


def _run_antspy(payload: dict[str, object], timeout: int) -> dict[str, object]:
    """Call _run_ants.py in an isolated subprocess and return the parsed result dict."""
    proc = subprocess.run(
        [sys.executable, "-m", "medmcp_neuro.tools._run_ants"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"ANTsPy failed (exit {proc.returncode}): {proc.stderr.strip()}")
    stdout = proc.stdout.strip() if proc.stdout else ""
    result: dict[str, object] = json.loads(stdout) if stdout else {}
    if not result.get("ok"):
        raise RuntimeError(f"ANTsPy failed: {result.get('error', 'unknown')}")
    return result


def _str_list(val: object) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(v) for v in val]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]


def _bool_list(val: object) -> list[bool]:
    if not isinstance(val, list):
        return []
    return [bool(v) for v in val]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]


# ── register_to_template ───────────────────────────────────────────────────────


def register_to_template(
    input_path: Path,
    transform_type: _TRANSFORM_TYPE | None = None,
    output_dir: Path | None = None,
    template_path: Path | None = None,
) -> RegisterToTemplateChoiceResult | RegisterToTemplateResult:
    """Register a structural NIfTI image to a standard-space template (default: MNI152).

    Load the ``registration`` skill before calling this tool. Do not pick a transform
    type or call this tool without first presenting options to the user and getting
    their confirmation — the skill defines the exact sequence to follow.

    Normalises a 3-D structural volume to a standard-space template using ANTsPy
    (the Python interface to ANTs — installed automatically as a package dependency,
    no separate binary install required). On first use the MNI152NLin2009cAsym 1 mm
    T1w template is downloaded from the templateflow S3 bucket and cached in
    ``~/.medmcp_neuro/templates/``; subsequent calls use the cached copy. A custom
    template can be supplied via ``template_path``.

    Output is written to ``output_dir`` (defaults to the same directory as the input).

    Transform types:
        - ``rigid``: 6 DOF rigid-body alignment (~1-2 min).
        - ``similarity``: 7 DOF rigid + uniform scaling (~1-2 min). Use when images
          are from different scanners with slightly different voxel scales.
        - ``affine``: 12 DOF affine alignment (~2-5 min).
        - ``synquick``: affine + fast SyN deformable warp (~5-10 min).
        - ``syn``: affine + full SyN deformable warp, best quality (~20-60 min on CPU).

    Output files written to ``output_dir``:
        - ``{stem}_space-{space}.nii.gz`` — registered image (BIDS-named).
        - ``{stem}_to_{space}_0GenericAffine.mat`` — affine/rigid transform.
        - ``{stem}_to_{space}_1Warp.nii.gz`` — SyN warp field (syn/synquick only).
        - ``{stem}_to_{space}_1InverseWarp.nii.gz`` — inverse warp (syn/synquick only).

    Args:
        input_path: Absolute path to the structural NIfTI (.nii or .nii.gz).
        transform_type: Registration transform. Always omit on the first call — the
            tool returns available options for the user to choose from. Re-call with
            the confirmed value to run registration.
        output_dir: Directory where outputs are written. Defaults to ``input_path.parent``.
        template_path: Custom reference template. Defaults to MNI152NLin2009cAsym 1 mm.

    Returns:
        ``RegisterToTemplateChoiceResult`` when ``transform_type`` is omitted, or
        ``RegisterToTemplateResult`` on success. Pass ``forward_transforms`` directly
        to ``apply_transform`` to warp additional images into template space. For
        native space, pass ``inverse_transforms`` with ``invert_flags=inverse_invert_flags``.

    Raises:
        FileNotFoundError: If ``input_path`` or the resolved template does not exist.
        RuntimeError: If antspyx is not installed or registration fails.
    """
    _check_antspy()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if transform_type is None:
        choice: RegisterToTemplateChoiceResult = {
            "available_transforms": ["rigid", "similarity", "affine", "synquick", "syn"],
            "recommended_transform": "rigid",
            "_render": (
                'Available transform types (recommended: "rigid"):\n'
                '  "rigid"      — 6 DOF rigid-body alignment (~1-2 min). '
                "Recommended for most use cases.\n"
                '  "similarity" — 7 DOF rigid + uniform scaling (~1-2 min). '
                "Use when images are from different scanners.\n"
                '  "affine"     — 12 DOF affine alignment (~2-5 min). Adds scaling and shear.\n'
                '  "synquick"   — affine + fast SyN deformable warp (~5-10 min).\n'
                '  "syn"        — affine + full SyN, best quality (~20-60 min on CPU).\n'
                "NEXT ACTION: Tell the user the available transform types with their "
                'descriptions. Ask them to confirm "rigid" or choose a different type, '
                "then call register_to_template again with the chosen transform_type."
            ),
        }
        return choice

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpl = template_path if template_path is not None else get_mni152_1mm()
    if not tmpl.exists():
        raise FileNotFoundError(f"Template not found: {tmpl}")

    stem = _nii_stem(input_path)
    space = _MNI_SPACE if template_path is None else _space_label(tmpl)
    out_registered = out_dir / f"{stem}_space-{space}.nii.gz"
    prefix = str(out_dir / f"{stem}_to_{space}_")
    ants_type = _TRANSFORM_MAP[transform_type]
    timeout = _TIMEOUT_SYN if transform_type == "syn" else _TIMEOUT_FAST

    print(
        f"[medmcp-neuro] register_to_template: ANTsPy {ants_type} …",
        file=sys.stderr,
        flush=True,
    )
    run_result = _run_antspy(
        {
            "operation": "register",
            "fixed_path": str(tmpl),
            "moving_path": str(input_path),
            "out_registered": str(out_registered),
            "outprefix": prefix,
            "type_of_transform": ants_type,
        },
        timeout,
    )

    if not out_registered.exists():
        raise RuntimeError(f"Registration completed but output not found: {out_registered}")

    fwd = _str_list(run_result.get("fwdtransforms"))
    inv = _str_list(run_result.get("invtransforms"))
    inv_flags = _bool_list(run_result.get("inverse_invert_flags"))

    result: RegisterToTemplateResult = {
        "registered_path": str(out_registered),
        "transform_prefix": prefix,
        "forward_transforms": fwd,
        "inverse_transforms": inv,
        "inverse_invert_flags": inv_flags,
        "template_path": str(tmpl),
        "transform_type": transform_type,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report as a compact key-value list:\n"
            "  Input:      <input_path> (from call arguments)\n"
            "  Registered: <registered_path>\n"
            "  Template:   <template_path>\n"
            "  Type:       <transform_type>\n"
            "Substitute actual values from the result dict.\n"
            "NEXT ACTION: Tell the user the registered image path. Ask if they want\n"
            "to apply the same warp to additional images (masks, parcellations, lesion\n"
            "maps) via apply_transform:\n"
            "  - native→template: transforms=forward_transforms (no invert_flags needed)\n"
            "  - template→native: transforms=inverse_transforms, "
            "invert_flags=inverse_invert_flags\n"
            'When the reference is the MNI template, set output_space="MNI152NLin2009cAsym".\n'
            "Otherwise ask what processing step to run next."
        ),
    }
    return result


# ── coregister ─────────────────────────────────────────────────────────────────


def coregister(
    fixed_path: Path,
    moving_paths: list[Path],
    transform_type: _TRANSFORM_TYPE | None = None,
    output_dir: Path | None = None,
) -> CoregisterChoiceResult | CoregisterResult:
    """Align multiple images of the same subject to a common reference image.

    Load the ``registration`` skill before calling this tool. Do not pick a transform
    type or call this tool without first presenting options to the user and getting
    their confirmation — the skill defines the exact sequence to follow.

    Registers each moving image to the fixed reference using ANTsPy. Intended for
    within-subject multi-contrast alignment (e.g. FLAIR or T2w to T1w, DWI b0 to T1w).
    Use ``syn`` or ``synquick`` when EPI distortion or other non-linear differences
    are present (e.g. DWI/fMRI b0 to T1w).

    Output is written to ``output_dir`` (defaults to the same directory as ``fixed_path``).

    Output files written to ``output_dir`` for each moving image:
        - ``{moving_stem}_space-{fixed_label}.nii.gz`` — aligned image (BIDS-named).
        - ``{moving_stem}_to_{fixed_label}_0GenericAffine.mat`` — transform (linear types).
        - ``{moving_stem}_to_{fixed_label}_1Warp.nii.gz`` — warp field (syn/synquick only).

    To apply the same transform to additional images, call ``apply_transform`` with
    the ``forward_transforms_list`` or ``inverse_transforms_list`` entries returned here.

    Args:
        fixed_path: Reference image (e.g. T1w skull-stripped volume).
        moving_paths: Images to align to the fixed reference (e.g. FLAIR, T2w, b0).
        transform_type: Registration transform. Always omit on the first call — the
            tool returns available options for the user to choose from. Re-call with
            the confirmed value to run registration.
        output_dir: Directory where outputs are written. Defaults to ``fixed_path.parent``.

    Returns:
        ``CoregisterChoiceResult`` when ``transform_type`` is omitted, or
        ``CoregisterResult`` on success.

    Raises:
        FileNotFoundError: If ``fixed_path`` or any entry in ``moving_paths`` is missing.
        ValueError: If ``moving_paths`` is empty.
        RuntimeError: If antspyx is not installed or registration fails.
    """
    _check_antspy()
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_path}")
    if not moving_paths:
        raise ValueError("moving_paths must not be empty")
    missing = [str(p) for p in moving_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Moving image(s) not found: {', '.join(missing)}")

    if transform_type is None:
        coreg_choice: CoregisterChoiceResult = {
            "available_transforms": ["rigid", "similarity", "affine", "synquick", "syn"],
            "recommended_transform": "rigid",
            "_render": (
                'Available transform types (recommended: "rigid"):\n'
                '  "rigid"      — 6 DOF rigid-body alignment (~1-2 min). '
                "Recommended for most within-subject coregistration.\n"
                '  "similarity" — 7 DOF rigid + uniform scaling (~1-2 min). '
                "Use when images are from different scanners.\n"
                '  "affine"     — 12 DOF affine alignment (~2-5 min). '
                "Use when voxel sizes or field-of-view differ.\n"
                '  "synquick"   — affine + fast SyN deformable warp (~5-10 min). '
                "Use for EPI distortion (DWI/fMRI b0 to T1w).\n"
                '  "syn"        — affine + full SyN, best quality (~20-60 min on CPU).\n'
                "NEXT ACTION: Tell the user the available transform types. "
                'Ask them to confirm "rigid" or choose a different type, '
                "then call coregister again with the chosen transform_type."
            ),
        }
        return coreg_choice

    out_dir = output_dir if output_dir is not None else fixed_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ants_type = _TRANSFORM_MAP[transform_type]
    space = _space_label(fixed_path)
    timeout = _TIMEOUT_SYN if transform_type == "syn" else _TIMEOUT_FAST

    registered_paths: list[str] = []
    transform_prefixes: list[str] = []
    forward_transforms_list: list[list[str]] = []
    inverse_transforms_list: list[list[str]] = []
    inverse_invert_flags_list: list[list[bool]] = []

    for moving in moving_paths:
        stem = _nii_stem(moving)
        out_registered = out_dir / f"{stem}_space-{space}.nii.gz"
        prefix = str(out_dir / f"{stem}_to_{space}_")

        print(
            f"[medmcp-neuro] coregister: {moving.name} → {fixed_path.name} …",
            file=sys.stderr,
            flush=True,
        )
        run_result = _run_antspy(
            {
                "operation": "register",
                "fixed_path": str(fixed_path),
                "moving_path": str(moving),
                "out_registered": str(out_registered),
                "outprefix": prefix,
                "type_of_transform": ants_type,
            },
            timeout,
        )

        if not out_registered.exists():
            raise RuntimeError(f"Coregistration completed but output not found: {out_registered}")

        registered_paths.append(str(out_registered))
        transform_prefixes.append(prefix)
        forward_transforms_list.append(_str_list(run_result.get("fwdtransforms")))
        inverse_transforms_list.append(_str_list(run_result.get("invtransforms")))
        inverse_invert_flags_list.append(_bool_list(run_result.get("inverse_invert_flags")))

    reg_list = "\n".join(f"  {p}" for p in registered_paths)
    fwd_lines = "\n".join(
        f"  [{moving_path}]: {', '.join(fwd)}"
        for moving_path, fwd in zip(registered_paths, forward_transforms_list, strict=True)
    )

    result: CoregisterResult = {
        "registered_paths": registered_paths,
        "transform_prefixes": transform_prefixes,
        "forward_transforms_list": forward_transforms_list,
        "inverse_transforms_list": inverse_transforms_list,
        "inverse_invert_flags_list": inverse_invert_flags_list,
        "fixed_path": str(fixed_path),
        "transform_type": transform_type,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report as:\n"
            "  Fixed:  <fixed_path>\n"
            "  Type:   <transform_type>\n"
            "  Registered images:\n"
            f"{reg_list}\n"
            "Substitute actual values from the result dict.\n"
            "Forward transforms per image (pass to apply_transform to warp additional "
            "images into fixed space):\n"
            f"{fwd_lines}\n"
            "For inverse (warp back to moving space): use inverse_transforms_list with "
            "the corresponding inverse_invert_flags_list entry.\n"
            "NEXT ACTION: Confirm all images aligned correctly. Ask what to do next\n"
            "(e.g. registration to MNI template, tissue segmentation, lesion analysis)."
        ),
    }
    return result


# ── apply_transform ────────────────────────────────────────────────────────────


def apply_transform(
    input_path: Path,
    reference_path: Path,
    transforms: list[str],
    output_dir: Path | None = None,
    interpolation: Literal["Linear", "NearestNeighbor", "BSpline"] = "Linear",
    output_space: str | None = None,
    invert_flags: list[bool] | None = None,
) -> ApplyTransformResult:
    """Apply a pre-computed ANTs transform to an image.

    Warps ``input_path`` to the space defined by ``reference_path`` using a
    pre-computed transform chain. Typically used after ``register_to_template``
    or ``coregister`` to bring additional images (brain masks, lesion maps,
    parcellations) into the same space without re-running registration.

    Compose transform chains by passing the list of transform files in application
    order. Pass ``forward_transforms`` from a ``register_to_template`` result to
    warp from native into template space; pass ``inverse_transforms`` together with
    ``invert_flags=inverse_invert_flags`` to go back into native space.

    Output is written to ``output_dir`` (defaults to the same directory as the input).

    Args:
        input_path: Image to transform (.nii or .nii.gz).
        reference_path: Defines the output grid (space, resolution, field of view).
        transforms: Ordered list of transform file paths from a prior registration.
            Pass ``forward_transforms`` or ``inverse_transforms`` from
            ``register_to_template`` or ``coregister`` directly.
        output_dir: Directory where the transformed image is written. Defaults to
            ``input_path.parent``.
        interpolation: Resampling method. Use ``"NearestNeighbor"`` for integer
            labels (brain masks, atlas parcellations). Defaults to ``"Linear"``.
        output_space: BIDS space label for the output filename. Inferred from the
            last BIDS entity of ``reference_path`` if omitted. Pass explicitly when
            the reference is a template (e.g. ``output_space="MNI152NLin2009cAsym"``).
        invert_flags: Per-transform inversion flags (one ``bool`` per entry in
            ``transforms``). Required when applying ``inverse_transforms`` from a
            rigid or affine registration — pass ``inverse_invert_flags`` from the
            registration result directly.

    Returns:
        ``ApplyTransformResult`` with the path to the transformed image.

    Raises:
        FileNotFoundError: If ``input_path`` or ``reference_path`` does not exist.
        RuntimeError: If antspyx is not installed or the transform fails.
    """
    _check_antspy()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference not found: {reference_path}")

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    space = output_space if output_space is not None else _space_label(reference_path)
    stem = _nii_stem(input_path)
    out_path = out_dir / f"{stem}_space-{space}.nii.gz"

    payload: dict[str, object] = {
        "operation": "apply_transforms",
        "input_path": str(input_path),
        "reference_path": str(reference_path),
        "transforms": transforms,
        "out_path": str(out_path),
        "interpolator": interpolation,
    }
    if invert_flags is not None:
        payload["invert_flags"] = invert_flags

    print(
        f"[medmcp-neuro] apply_transform: {input_path.name} → {space} space …",
        file=sys.stderr,
        flush=True,
    )
    _run_antspy(payload, _TIMEOUT_FAST)

    if not out_path.exists():
        raise RuntimeError(f"apply_transform completed but output not found: {out_path}")

    result: ApplyTransformResult = {
        "output_path": str(out_path),
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report as a compact key-value list:\n"
            "  Input:  <input_path> (from call arguments)\n"
            "  Output: <output_path>\n"
            "Substitute actual values from the result dict.\n"
            "NEXT ACTION: Confirm the transformed image exists and ask the user what to do next."
        ),
    }
    return result
