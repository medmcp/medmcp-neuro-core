"""Average repeated scans of one session (e.g. OASIS mpr-1..mpr-4, BIDS run-01/run-02).

Repeats of the same sequence in one session are averaged for signal-to-noise and to
dilute a scan spoiled by motion. Two methods:

- ``register="none"`` (default): the voxel-wise mean, no interpolation. Only for scans
  on an identical grid; refused otherwise.
- ``register="rigid"``: FreeSurfer's ``mri_robust_template`` (``--satit``, unbiased
  mid-space, one LTA per scan); needs the user's FreeSurfer licence. Without
  FreeSurfer or a licence, ANTsPy rigid registration of every scan to the first and
  the mean on the first scan's grid.

Every call checks that the scans are the same sequence (BIDS suffix, sidecar and
header parameters) and how well they agree (correlation to the other scans, a quick
rigid motion estimate), and flags a scan that clearly differs. Heavy work (ANTsPy,
FreeSurfer) runs in subprocesses so their output never reaches the MCP stdio pipe.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

from medmcp_neuro_core.tools._neuro import nii_stem, require_image
from medmcp_neuro_core.tools.freesurfer._env import freesurfer_env, freesurfer_home
from medmcp_neuro_core.tools.freesurfer._license import find_license
from medmcp_neuro_core.tools.freesurfer._workspace import resolve_inside, workspace_root

JsonDict = dict[str, Any]
Register = Literal["none", "rigid"]

_TIMEOUT = 3600
# Grid tolerance for "none": voxel sizes and the affine's rotation/zoom part, and its
# translation in mm.
_ZOOM_TOL = 1e-3
_ROT_TOL = 1e-3
_SHIFT_TOL_MM = 0.05
# A scan "clearly differs" above these (relative to the most central scan).
_FLAG_SHIFT_MM = 1.0
_FLAG_ROT_DEG = 1.0
_FLAG_CORR = 0.90
_FLAG_CORR_GAP = 0.05
# Repeat markers removed from file names (run-01, mpr-1, rep2, scan3, ...).
_REPEAT_TOKEN = re.compile(r"^(run|mpr|rep|repeat|scan|echo)-?\d+$", re.IGNORECASE)
# BIDS suffixes that name a contrast; two of these must match.
_CONTRASTS = frozenset(
    {
        "T1w",
        "T2w",
        "PDw",
        "T2starw",
        "FLAIR",
        "inplaneT1",
        "inplaneT2",
        "PDT2",
        "angio",
        "T1map",
        "T2map",
        "UNIT1",
        "MP2RAGE",
        "veno",
        "defacemask",
        "mask",
        "dseg",
    }
)
# Sidecar parameters that must agree (relative tolerance 1%).
_SIDECAR_NUMERIC = (
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "MagneticFieldStrength",
)
_SIDECAR_TEXT = ("SeriesDescription", "ProtocolName", "ScanningSequence", "PulseSequenceType")
_AVERAGE_MARKERS = ("desc-average", "_average")
_MAX_LINES = 5
_MAX_CHARS = 120


# ── small helpers ──────────────────────────────────────────────────────────────


def _load(path: Path) -> Any:
    import nibabel as nib

    return cast("Any", nib.load(str(path)))  # pyright: ignore[reportUnknownMemberType]


def _data(img: Any) -> Any:
    import numpy as np

    arr = np.asarray(img.dataobj, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _line(text: str) -> str:
    return text if len(text) <= _MAX_CHARS else text[: _MAX_CHARS - 1] + "…"


def _short_name(path: Path) -> str:
    """The repeat marker of a file name (``run-03``, ``mpr-3``), else the name."""
    marks = [t for t in nii_stem(path).split("_") if _REPEAT_TOKEN.match(t)]
    return marks[-1] if marks else path.name


def _stem_tokens(path: Path) -> list[str]:
    return [t for t in nii_stem(path).split("_") if t]


def default_output_name(first: Path) -> str:
    """``<first stem without the repeat marker>`` + ``desc-average`` / ``_average``.

    BIDS names keep their suffix last (``sub-01_ses-1_desc-average_T1w.nii.gz``);
    other names get ``_average`` appended (``OAS2_0001_MR1_anon_average.nii.gz``).
    """
    tokens = [t for t in _stem_tokens(first) if not _REPEAT_TOKEN.match(t)]
    bids = len(tokens) > 1 and "-" not in tokens[-1] and any("-" in t for t in tokens[:-1])
    tokens = [*tokens[:-1], "desc-average", tokens[-1]] if bids else [*tokens, "average"]
    return "_".join(tokens) + ".nii.gz"


def _suffix(path: Path) -> str | None:
    tokens = _stem_tokens(path)
    last = tokens[-1] if tokens else ""
    return last if last in _CONTRASTS else None


def _sidecar(path: Path) -> JsonDict | None:
    side = path.with_name(nii_stem(path) + ".json")
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text())
    except (OSError, ValueError):
        return None
    return cast("JsonDict", data) if isinstance(data, dict) else None


def _descrip_te(img: Any) -> float | None:
    """TE from a dcm2niix-style NIfTI ``descrip`` (``TE=2.9;Time=...``), if present."""
    try:
        raw = img.header["descrip"]
        text = bytes(raw.tobytes()).decode("latin-1", "replace")
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    m = re.search(r"TE=([0-9.]+)", text)
    return float(m.group(1)) if m else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.01 * max(abs(a), abs(b), 1e-9)


def sequence_problems(paths: list[Path], images: list[Any]) -> list[str]:
    """Reasons the scans look like different sequences (empty when they agree)."""
    problems: list[str] = []
    suffixes = {p.name: _suffix(p) for p in paths}
    named = {s for s in suffixes.values() if s is not None}
    if len(named) > 1:
        problems.append(
            "the file names name different contrasts ("
            + ", ".join(f"{n}: {s}" for n, s in suffixes.items() if s)
            + ")"
        )
    cars = [(p, _sidecar(p)) for p in paths]
    with_car = [(p, c) for p, c in cars if c is not None]
    if len(with_car) > 1:
        ref_path, ref = with_car[0]
        for key in _SIDECAR_NUMERIC:
            for p, car in with_car[1:]:
                a, b = ref.get(key), car.get(key)
                if isinstance(a, int | float) and isinstance(b, int | float) and not _close(a, b):
                    problems.append(f"{key} differs: {ref_path.name} {a}, {p.name} {b}")
    tes = [(p, _descrip_te(img)) for p, img in zip(paths, images, strict=True)]
    known = [(p, te) for p, te in tes if te is not None]
    if len(known) > 1 and any(not _close(known[0][1], te) for _, te in known[1:]):
        problems.append(
            "the headers give different echo times ("
            + ", ".join(f"{p.name}: TE={te}" for p, te in known)
            + ")"
        )
    return problems


def sequence_notes(paths: list[Path]) -> list[str]:
    """Sidecar text fields that differ: worth a look, not a refusal."""
    cars = [(p, _sidecar(p)) for p in paths]
    with_car = [(p, c) for p, c in cars if c is not None]
    notes: list[str] = []
    if len(with_car) > 1:
        for key in _SIDECAR_TEXT:
            values = {
                re.sub(r"\d+", "#", str(c.get(key))) for _, c in with_car if c.get(key) is not None
            }
            if len(values) > 1:
                notes.append(
                    f"Sidecar {key} differs between the scans; check they are one sequence."
                )
    return notes


def grid_problems(images: list[Any], names: list[str]) -> list[str]:
    """Why the scans are not on one voxel grid (empty when they are)."""
    import numpy as np

    problems: list[str] = []
    ref = images[0]
    ref_shape = tuple(_data_shape(ref))
    ref_zooms = np.asarray(ref.header.get_zooms()[:3], dtype=float)
    ref_aff = np.asarray(ref.affine, dtype=float)
    for img, name in zip(images[1:], names[1:], strict=True):
        shape = tuple(_data_shape(img))
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        aff = np.asarray(img.affine, dtype=float)
        if shape != ref_shape:
            problems.append(f"{name} has {shape} voxels, {names[0]} has {ref_shape}")
            continue
        if not np.allclose(zooms, ref_zooms, atol=_ZOOM_TOL):
            problems.append(
                f"{name} voxel size {tuple(round(float(z), 3) for z in zooms)} mm, "
                f"{names[0]} {tuple(round(float(z), 3) for z in ref_zooms)} mm"
            )
            continue
        if not np.allclose(aff[:3, :3], ref_aff[:3, :3], atol=_ROT_TOL):
            problems.append(f"{name} has a different orientation than {names[0]}")
            continue
        shift = float(np.linalg.norm(aff[:3, 3] - ref_aff[:3, 3]))
        if shift > _SHIFT_TOL_MM:
            problems.append(f"{name} field of view is shifted {shift:.2f} mm from {names[0]}")
    return problems


def _data_shape(img: Any) -> list[int]:
    shape = [int(s) for s in img.shape]
    while len(shape) > 3 and shape[-1] == 1:
        shape.pop()
    return shape


def _foreground(arr: Any) -> Any:
    import numpy as np

    positive = arr[arr > 0]
    if positive.size == 0:
        return np.ones(arr.shape, dtype=bool)
    return arr > 0.1 * float(np.percentile(positive, 99))  # pyright: ignore[reportUnknownArgumentType]


def _normalise(arrays: list[Any]) -> list[Any]:
    """Scale each scan to the first scan's mean foreground intensity."""
    import numpy as np

    target = float(arrays[0][_foreground(arrays[0])].mean())
    out: list[Any] = []
    for arr in arrays:
        mean = float(arr[_foreground(arr)].mean())
        out.append(arr * np.float32(target / mean) if mean > 0 else arr)
    return out


def correlations(arrays: list[Any]) -> tuple[int, list[float]]:
    """Pearson correlation of each scan with the most typical scan, in the head.

    The most typical scan is the one with the highest summed correlation to the
    others, so one bad scan does not lower every other scan's value; it gets its best
    correlation with another scan. Returns (index of that scan, correlations).
    """
    import numpy as np

    mask = _foreground(np.mean(arrays, axis=0))
    r = np.corrcoef(np.stack([a[mask] for a in arrays]))
    n = len(arrays)
    ref = int(np.argmax(r.sum(axis=1)))
    out = [float(r[i, ref]) for i in range(n)]
    out[ref] = float(max(r[ref, j] for j in range(n) if j != ref))
    return ref, out


def centre_of_mass_mm(img: Any, arr: Any) -> list[float]:
    """Intensity-weighted centre of the head in scanner coordinates (mm)."""
    import numpy as np

    mask = _foreground(arr)
    weights = np.where(mask, arr, 0).astype(np.float64)
    idx = np.indices(arr.shape).reshape(3, -1).astype(np.float64)
    w = weights.reshape(-1)
    centre_vox = (idx * w).sum(axis=1) / max(float(w.sum()), 1e-9)
    world = np.asarray(img.affine, dtype=float) @ np.append(centre_vox, 1.0)
    return [float(v) for v in world[:3]]


def motion_from_matrices(matrices: list[Any], centre: list[float]) -> tuple[int, list[JsonDict]]:
    """Each scan's shift (mm, at the head's centre) and rotation (deg) vs the central scan.

    ``matrices[i]`` maps points of scan 1 to scan i. The reference is the scan with
    the smallest total shift to all others, so one moved scan does not make every
    other scan look moved; with two scans it is scan 1.
    """
    import numpy as np

    mats = [np.asarray(m, dtype=float) for m in matrices]
    p = np.append(np.asarray(centre, dtype=float), 1.0)

    def rel(i: int, j: int) -> tuple[float, float]:
        m = np.linalg.inv(mats[i]) @ mats[j]
        shift = float(np.linalg.norm((m @ p)[:3] - p[:3]))
        cos = (float(np.trace(m[:3, :3])) - 1.0) / 2.0
        rot = float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))
        return shift, rot

    n = len(mats)
    if n <= 2:
        ref = 0
    else:
        totals = [sum(rel(i, j)[0] for j in range(n)) for i in range(n)]
        ref = int(np.argmin(totals))
    rows: list[JsonDict] = []
    for j in range(n):
        shift, rot = rel(ref, j)
        rows.append({"shift_mm": round(shift, 2), "rotation_deg": round(rot, 2)})
    return ref, rows


def _ants_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ants") is not None


def _run_worker(payload: JsonDict, timeout: int = _TIMEOUT) -> JsonDict:
    """Run ``_run_average`` in a subprocess and return its result dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        result_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "medmcp_neuro_core.tools._run_average"],
            input=json.dumps({**payload, "result_path": result_path}),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        try:
            with open(result_path) as f:
                result: JsonDict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            result = {}
        if not result.get("ok"):
            detail = result.get("error") or proc.stderr.strip()[-2000:] or "unknown"
            raise RuntimeError(f"ANTsPy failed (exit {proc.returncode}): {detail}")
        return result
    finally:
        Path(result_path).unlink(missing_ok=True)


def _snapshot(image_path: Path, destination: Path, title: str) -> None:
    """Three orthogonal slices through the centre of the averaged image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    img = cast("Any", nib.as_closest_canonical(_load(image_path)))  # pyright: ignore[reportUnknownMemberType]
    arr = _data(img)
    c = [s // 2 for s in arr.shape]
    positive = arr[arr > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0  # pyright: ignore[reportUnknownArgumentType]
    views = [
        ("Axial", arr[:, :, c[2]].T),
        ("Coronal", arr[:, c[1], :].T),
        ("Sagittal", arr[c[0], :, :].T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor="black")  # pyright: ignore[reportUnknownMemberType]
    for ax, (name, sl) in zip(axes, views, strict=True):
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(name, color="white", fontsize=11)
        ax.axis("off")
    fig.suptitle(title, color="white", fontsize=11)  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(destination, dpi=100, facecolor="black")  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def _save_float32(arr: Any, template: Any, path: Path) -> None:
    import nibabel as nib
    import numpy as np

    header = template.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(arr.astype(np.float32), template.affine, header)
    out.header.set_slope_inter(1.0, 0.0)  # pyright: ignore[reportUnknownMemberType]
    nib.save(out, str(path))  # pyright: ignore[reportUnknownMemberType]


def _package_version() -> str:
    try:
        return version("medmcp-neuro-core")
    except PackageNotFoundError:
        return "unknown"


# ── resolving the inputs ───────────────────────────────────────────────────────


def _resolve_groups(
    input_paths: list[str] | None,
    folder: str | None,
    pattern: str,
    groups: list[list[str]] | None,
    exclude: list[str] | None,
    ws: Path | None,
) -> tuple[list[list[Path]], list[list[Path]], list[str]]:
    given = sum(x is not None and len(x) > 0 for x in (input_paths, folder, groups))
    if given != 1:
        raise ValueError("Give exactly one of input_paths, folder (with pattern) or groups.")
    notes: list[str] = []
    raw: list[list[Path]]
    if groups:
        raw = [[resolve_inside(p, "groups", ws) for p in g] for g in groups]
    elif input_paths:
        raw = [[resolve_inside(p, "input_paths", ws) for p in input_paths]]
    else:
        root = resolve_inside(cast("str", folder), "folder", ws)
        if not root.is_dir():
            raise FileNotFoundError(f"folder not found: {root}")
        found = sorted(
            p for p in root.glob(pattern) if p.is_file() and p.name.endswith((".nii", ".nii.gz"))
        )
        skipped = [p for p in found if any(m in p.name for m in _AVERAGE_MARKERS)]
        if skipped:
            notes.append("Left out earlier averages: " + ", ".join(p.name for p in skipped))
        raw = [[p for p in found if p not in skipped]]
        if not raw[0]:
            raise FileNotFoundError(f"No NIfTI files match {pattern!r} in {root}")
    drop = [str(e).strip() for e in exclude or [] if str(e).strip()]
    kept: list[list[Path]] = []
    excluded: list[list[Path]] = []
    for g in raw:
        out = [p for p in g if p.name not in drop and str(p) not in drop]
        excluded.append([p for p in g if p not in out])
        kept.append(out)
    for g in kept:
        if len(g) < 2:
            raise ValueError(
                "Each group needs at least two scans to average, got "
                + (", ".join(p.name for p in g) or "none")
            )
        if len(set(g)) != len(g):
            raise ValueError("The same file is listed twice: " + ", ".join(p.name for p in g))
        for p in g:
            require_image(p)
    return kept, excluded, notes


def _targets(
    group: list[Path],
    output_path: str | None,
    output_dir: str | None,
    ws: Path | None,
    register: Register,
) -> JsonDict:
    if output_path:
        out = resolve_inside(output_path, "output_path", ws)
        name = out.name
        if name.endswith(".nii"):
            name += ".gz"
        elif not name.endswith(".nii.gz"):
            name += ".nii.gz"
        out = out.with_name(name)
    else:
        folder = resolve_inside(output_dir, "output_dir", ws) if output_dir else group[0].parent
        out = folder / default_output_name(group[0])
    stem = nii_stem(out)
    targets: JsonDict = {
        "image": out,
        "png": out.with_name(stem + ".png"),
        "json": out.with_name(stem + "_log.json"),
        "csv": out.with_name(stem + "_qc.csv"),
        "transforms": {},
    }
    if register == "rigid":
        targets["transforms"] = {str(p): out.with_name(f"{stem}_from-{nii_stem(p)}") for p in group}
    return targets


# ── the tool ───────────────────────────────────────────────────────────────────


def average_repeats(
    input_paths: list[str] | None = None,
    folder: str | None = None,
    pattern: str = "*.nii*",
    groups: list[list[str]] | None = None,
    register: Register = "none",
    output_path: str | None = None,
    output_dir: str | None = None,
    exclude: list[str] | None = None,
    normalise_intensity: bool = False,
    check_sequence: bool = True,
    overwrite: bool = False,
    license_path: str | None = None,
) -> JsonDict:
    """Average repeated scans of one session into one image (e.g. mpr-1..mpr-4, run-01/02).

    Load the ``registration`` skill first. Use only for repeats of the SAME sequence in
    one session (several MPRAGEs, T1w run-01/run-02); never mix sequences or sessions.
    Repeats exist for signal-to-noise and robustness to motion.

    ``register``: ``"none"`` (default) is the plain voxel-wise mean without
    interpolation; it needs an identical grid (shape, voxel size, orientation, field
    of view) and is refused otherwise. ``"rigid"`` aligns the scans first: FreeSurfer
    ``mri_robust_template --satit`` (unbiased mid-space, one LTA per scan; needs the
    user's FreeSurfer licence), or ANTsPy rigid to the first scan + mean when
    FreeSurfer cannot run (no licence); the result says which ran.

    Always checks the scans are one sequence (file names, sidecars, headers) and how
    well they agree (correlation with the other scans, a quick rigid motion estimate),
    and flags a scan that clearly differs; a plain mean still runs but warns.
    Optional intensity normalisation (off by default) scales every scan to the first
    scan's mean brightness first. Never overwrites inputs or existing outputs unless
    ``overwrite=True``. Takes seconds for "none", minutes for "rigid".

    Writes per group: the average (``.nii.gz``, float32; header of the first scan for
    "none"), a PNG of three slices, ``<name>_log.json`` (inputs, method, QC, excluded
    scans) and ``<name>_qc.csv``; for "rigid" also one transform per scan.

    Args:
        input_paths: The repeats of one session (absolute NIfTI paths).
        folder: Instead of input_paths: a folder whose files matching ``pattern`` are
            the repeats (earlier averages are left out).
        pattern: Glob for ``folder``, e.g. ``"*mpr-*.nii.gz"`` or ``"*_T1w.nii.gz"``.
        groups: Instead: several sessions at once, one list of paths per session.
        register: ``"none"`` (plain mean, same grid only) or ``"rigid"``.
        output_path: The averaged image (one group only). Default: next to the
            first scan, named after it without the repeat marker plus
            ``desc-average`` (BIDS) or ``_average``.
        output_dir: Folder for the outputs instead of the first scan's folder.
        exclude: File names or paths to leave out (e.g. a scan flagged as moved).
        normalise_intensity: Scale every scan to the first scan's mean intensity
            before averaging (FreeSurfer: ``--iscale``).
        check_sequence: Refuse scans that look like different sequences.
        overwrite: Replace existing outputs.
        license_path: The FreeSurfer licence file, if not at the default location.

    Returns:
        ``tell_the_user`` (at most five short lines, most important first),
        ``written`` (only files this call wrote) and per-group details.
    """
    started = time.monotonic()
    ws = workspace_root()
    kept, excluded, notes = _resolve_groups(input_paths, folder, pattern, groups, exclude, ws)
    if output_path and len(kept) > 1:
        raise ValueError("output_path is for one group; use output_dir for several groups.")

    plans: list[JsonDict] = []
    all_targets: list[Path] = []
    for g in kept:
        t = _targets(g, output_path, output_dir, ws, register)
        files = [t["image"], t["png"], t["json"], t["csv"]]
        trans = cast("dict[str, Path]", t["transforms"])
        files += [Path(f"{b}{ext}") for b in trans.values() for ext in (".lta", ".mat")]
        clash = [str(f) for f in files if f in g]
        if clash:
            raise ValueError(f"The output would overwrite an input: {', '.join(clash)}")
        all_targets += [f for f in files if isinstance(f, Path)]
        plans.append(t)
    if len(set(all_targets)) != len(all_targets):
        raise ValueError(
            "Two groups would write the same output names; give output_dir per call or "
            "rename the inputs."
        )
    if not overwrite:
        existing = [f for f in all_targets if f.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist (pass overwrite=True to replace them, or choose "
                "output_path/output_dir): " + ", ".join(str(f) for f in existing)
            )

    results: list[JsonDict] = []
    written: list[str] = []
    for g, ex, t in zip(kept, excluded, plans, strict=True):
        res = _average_group(g, ex, t, register, normalise_intensity, check_sequence, license_path)
        written += res.pop("_written")
        results.append(res)

    lines: list[str] = []
    for r in results:
        lines += r["tell"]
    lines = [_line(x) for x in lines][:_MAX_LINES]
    out: JsonDict = {
        "tell_the_user": lines,
        "written": written,
        "groups": [{k: v for k, v in r.items() if k != "tell"} for r in results],
        "notes": notes,
        "runtime_s": round(time.monotonic() - started, 1),
        "_render": (
            "Show tell_the_user first, line by line, as given. Then the averaged image "
            "path(s) and the PNG (show it if you can). If a scan is flagged, offer "
            "register='rigid' or exclude=[that scan]. Do not write ANTs/FSL commands "
            "yourself for this task."
        ),
    }
    return out


def _average_group(
    group: list[Path],
    excluded: list[Path],
    targets: JsonDict,
    register: Register,
    normalise: bool,
    check_sequence: bool,
    license_path: str | None,
) -> JsonDict:
    import numpy as np

    started = time.monotonic()
    names = [p.name for p in group]
    images = [_load(p) for p in group]
    label = default_output_name(group[0]).removesuffix(".nii.gz")
    base: JsonDict = {
        "status": "refused",
        "inputs": [str(p) for p in group],
        "excluded": [str(p) for p in excluded],
        "register": register,
        "_written": [],
    }
    for img, name in zip(images, names, strict=True):
        if len(_data_shape(img)) != 3:
            reason = f"{name} is not a 3-D volume (shape {tuple(img.shape)})"
            return {**base, "reason": reason, "tell": [f"Not averaged ({label}): {reason}."]}

    if check_sequence:
        problems = sequence_problems(group, images)
        if problems:
            reason = "; ".join(problems)
            return {
                **base,
                "reason": reason,
                "tell": [
                    f"Not averaged ({label}): the scans look like different sequences.",
                    reason,
                    "Average only repeats of one sequence.",
                ],
            }
    warnings = sequence_notes(group)

    grid = grid_problems(images, names)
    same_grid = not grid
    if register == "none" and not same_grid:
        reason = "; ".join(grid)
        return {
            **base,
            "reason": reason,
            "tell": [
                f"Not averaged ({label}): the scans are not on one voxel grid, so a plain "
                "mean is wrong.",
                reason,
                "Use register='rigid' to align them first.",
            ],
        }

    # ── QC (always, cheap) ────────────────────────────────────────────────────
    arrays = [_data(img) for img in images]
    coms = [centre_of_mass_mm(img, a) for img, a in zip(images, arrays, strict=True)]
    com_shift = [round(float(np.linalg.norm(np.subtract(c, coms[0]))), 2) for c in coms]
    corr: list[float | None] = [None] * len(group)
    corr_ref: int | None = None
    if same_grid:
        corr_ref, values = correlations(arrays)
        corr = [round(c, 4) for c in values]
    motion: list[JsonDict] | None = None
    ref = 0
    if _ants_available():
        try:
            quick = _run_worker({"operation": "quick_rigid", "paths": [str(p) for p in group]})
            ref, motion = motion_from_matrices(
                cast("list[Any]", quick["matrices"]), cast("list[float]", quick["centre"])
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"Quick motion estimate failed ({exc}); centre-of-mass shift only.")
    else:
        warnings.append("ANTsPy not available: motion estimated from the centre of mass only.")

    qc_rows: list[JsonDict] = []
    flagged: list[tuple[int, str]] = []
    for i, p in enumerate(group):
        row: JsonDict = {
            "scan": i + 1,
            "file": p.name,
            "corr_to_reference": corr[i],
            "corr_reference_scan": None if corr_ref is None else corr_ref + 1,
            "com_shift_mm": com_shift[i],
        }
        reasons: list[str] = []
        if motion is not None:
            row.update(motion[i])
            row["reference_scan"] = ref + 1
            moved: list[str] = []
            if motion[i]["shift_mm"] >= _FLAG_SHIFT_MM:
                moved.append(f"shifted ~{motion[i]['shift_mm']:.1f} mm")
            if motion[i]["rotation_deg"] >= _FLAG_ROT_DEG:
                moved.append(f"rotated ~{motion[i]['rotation_deg']:.1f}°")
            if moved:
                reasons.append(" and ".join(moved) + f" vs scan {ref + 1}")
        elif com_shift[i] >= _FLAG_SHIFT_MM:
            reasons.append(f"centre shifted ~{com_shift[i]:.1f} mm vs scan 1")
        c = corr[i]
        known = [v for v in corr if v is not None]
        if c is not None and (
            c < _FLAG_CORR or (len(known) > 2 and c < max(known) - _FLAG_CORR_GAP)
        ):
            reasons.append(f"agrees less (r={c:.2f})")
        row["flag"] = ", ".join(reasons)
        if reasons:
            flagged.append((i, ", ".join(reasons)))
        qc_rows.append(row)

    # ── average ───────────────────────────────────────────────────────────────
    out_image = cast("Path", targets["image"])
    out_image.parent.mkdir(parents=True, exist_ok=True)
    trans_targets = cast("dict[str, Path]", targets["transforms"])
    written: list[str] = []
    engine = "plain voxel-wise mean (no interpolation)"
    command: list[str] | None = None
    fs_failed: str | None = None
    why_not_fs = ""
    if register == "none":
        data = _normalise(arrays) if normalise else arrays
        _save_float32(np.mean(data, axis=0), images[0], out_image)
        written.append(str(out_image))
    else:
        fs_bin = freesurfer_home() / "bin" / "mri_robust_template"
        licence: Path | None = None
        if not fs_bin.is_file():
            why_not_fs = "FreeSurfer is not installed here"
        else:
            try:
                licence = find_license(license_path)
            except (FileNotFoundError, ValueError) as exc:
                why_not_fs = "no usable FreeSurfer licence"
                warnings.append(str(exc))
        if licence is not None:
            ltas = [Path(str(trans_targets[str(p)]) + ".lta") for p in group]
            command = [
                str(fs_bin),
                "--mov",
                *map(str, group),
                "--template",
                str(out_image),
                "--satit",
                "--average",
                "0",
                "--lta",
                *map(str, ltas),
            ]
            if normalise:
                command.append("--iscale")
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    env = freesurfer_env(
                        os.environ,
                        license_path=str(licence),
                        subjects_dir=tmp,
                        threads=min(8, os.cpu_count() or 1),
                    )
                    proc = subprocess.run(
                        command, env=env, cwd=tmp, capture_output=True, text=True, timeout=_TIMEOUT
                    )
                if proc.returncode != 0 or not out_image.is_file():
                    tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
                    fs_failed = f"exit {proc.returncode}: " + " | ".join(tail)
            except (OSError, subprocess.TimeoutExpired) as exc:
                fs_failed = str(exc)
            if fs_failed is None:
                tmpl = _load(out_image)
                _save_float32(_data(tmpl), tmpl, out_image)
                written.append(str(out_image))
                written += [str(lta) for lta in ltas if lta.is_file()]
                engine = "FreeSurfer mri_robust_template --satit --average 0 (mid-space)"
            else:
                for f in [out_image, *ltas]:
                    f.unlink(missing_ok=True)
                why_not_fs = "FreeSurfer failed"
                warnings.append(f"mri_robust_template failed ({fs_failed}).")
        if not written:
            if not _ants_available():
                raise RuntimeError(f"register='rigid' needs FreeSurfer or ANTsPy ({why_not_fs}).")
            mats = [Path(str(trans_targets[str(p)]) + ".mat") for p in group]
            with tempfile.TemporaryDirectory() as tmp:
                warped = [str(Path(tmp) / f"scan{i}.nii.gz") for i in range(len(group))]
                _run_worker(
                    {
                        "operation": "rigid_to_first",
                        "paths": [str(p) for p in group],
                        "warped_paths": warped,
                        "transform_paths": [str(m) for m in mats],
                    }
                )
                aligned = [_data(_load(Path(w))) for w in warped]
            data = _normalise(aligned) if normalise else aligned
            _save_float32(np.mean(data, axis=0), images[0], out_image)
            written.append(str(out_image))
            written += [str(m) for m in mats[1:] if m.is_file()]
            engine = f"ANTsPy rigid to scan 1 + mean on its grid (fallback: {why_not_fs})"
            for row, c in zip(qc_rows, correlations(aligned)[1], strict=True):
                row["corr_after_rigid"] = round(c, 4)

    png = cast("Path", targets["png"])
    kind = "plain" if register == "none" else "rigid"
    title = f"{out_image.name}: mean of {len(group)} scans ({kind})"
    try:
        _snapshot(out_image, png, title)
        written.append(str(png))
    except Exception as exc:  # the average stands without the picture
        warnings.append(f"PNG snapshot not written: {exc}")

    # ── messages ──────────────────────────────────────────────────────────────
    tell: list[str] = []
    method = (
        "plain mean"
        if register == "none"
        else ("rigid, FreeSurfer" if engine.startswith("FreeSurfer") else "rigid, ANTsPy fallback")
    )
    tell.append(f"Averaged {len(group)} scans ({method}) -> {out_image.name}")
    for i, why in flagged:
        scan = f"Scan {i + 1} ({_short_name(group[i])})"
        if register == "none":
            tell.append(f"{scan} {why}: plain mean will blur; use register='rigid' or exclude it.")
        else:
            tell.append(f"{scan} {why}: aligned, but check it.")
    corr_known = [c for c in corr if c is not None]
    summary: list[str] = []
    if corr_known:
        summary.append(f"r {min(corr_known):.3f}-{max(corr_known):.3f}")
    if motion is not None:
        summary.append(
            f"motion ≤{max(m['shift_mm'] for m in motion):.1f} mm/"
            f"{max(m['rotation_deg'] for m in motion):.1f}°"
        )
    if summary and not flagged:
        tell.append("Scans agree: " + ", ".join(summary) + ".")
    if register == "rigid" and not engine.startswith("FreeSurfer"):
        tell.append(f"FreeSurfer not used ({why_not_fs}); ANTsPy rigid to scan 1 instead.")
    if normalise:
        tell.append("Intensities scaled to the first scan before averaging.")

    elapsed = round(time.monotonic() - started, 1)
    log: JsonDict = {
        "tool": "average_repeats",
        "package": f"medmcp-neuro-core {_package_version()}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": [
            {
                "path": str(p),
                "shape": _data_shape(img),
                "voxel_mm": [round(float(z), 4) for z in img.header.get_zooms()[:3]],
            }
            for p, img in zip(group, images, strict=True)
        ],
        "excluded": [str(p) for p in excluded],
        "register": register,
        "engine": engine,
        "command": command,
        "normalise_intensity": normalise,
        "same_grid": same_grid,
        "grid_differences": grid,
        "qc": qc_rows,
        "qc_thresholds": {
            "shift_mm": _FLAG_SHIFT_MM,
            "corr_gap": _FLAG_CORR_GAP,
            "rotation_deg": _FLAG_ROT_DEG,
            "corr": _FLAG_CORR,
        },
        "warnings": warnings,
        "output": str(out_image),
        "runtime_s": elapsed,
    }
    json_path = cast("Path", targets["json"])
    csv_path = cast("Path", targets["csv"])
    json_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    written.append(str(json_path))
    columns = [
        "scan",
        "file",
        "corr_to_reference",
        "corr_reference_scan",
        "corr_after_rigid",
        "com_shift_mm",
        "shift_mm",
        "rotation_deg",
        "reference_scan",
        "flag",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(qc_rows)
    written.append(str(csv_path))

    return {
        "status": "averaged",
        "output": str(out_image),
        "png": str(png) if str(png) in written else None,
        "log_json": str(json_path),
        "qc_csv": str(csv_path),
        "inputs": [str(p) for p in group],
        "excluded": [str(p) for p in excluded],
        "register": register,
        "engine": engine,
        "same_grid": same_grid,
        "qc": qc_rows,
        "flagged_scans": [i + 1 for i, _ in flagged],
        "warnings": warnings,
        "runtime_s": elapsed,
        "tell": tell,
        "_written": written,
    }
