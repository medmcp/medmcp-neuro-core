"""White-matter lesion segmentation tool using LST-AI."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro.tools import _run_lstai
from medmcp_neuro.tools._run_lstai import Backend

_TIMEOUT = 7200  # LST-AI on CPU can run for well over an hour


class SegmentLesionsResult(TypedDict):
    """Successful lesion-segmentation result."""

    lesion_mask_path: str
    annotated_path: str | None
    output_files: list[str]
    t1_path: str
    flair_path: str
    backend: str
    device: str
    skull_stripped: bool
    _render: str


def _nii_set(directory: Path) -> set[Path]:
    """Return the set of NIfTI files currently in ``directory`` (non-recursive)."""
    if not directory.exists():
        return set()
    return {p for p in directory.iterdir() if p.name.endswith((".nii", ".nii.gz"))}


def _pick_segmentation(files: list[Path]) -> Path | None:
    """Heuristically pick the primary binary lesion mask from produced files."""
    segs = [p for p in files if "seg" in p.name.lower() and "label" not in p.name.lower()]
    if segs:
        return segs[0]
    return files[0] if files else None


def _pick_annotated(files: list[Path]) -> Path | None:
    """Pick the region-annotated (labelled) segmentation, if LST-AI produced one."""
    labelled = [p for p in files if "label" in p.name.lower() or "annot" in p.name.lower()]
    return labelled[0] if labelled else None


def segment_lesions(
    t1_path: Path,
    flair_path: Path,
    output_dir: Path | None = None,
    device: str = "cpu",
    gpu_id: int = 0,
    skull_stripped: bool = False,
    backend: Backend | None = None,
) -> SegmentLesionsResult:
    """Segment white-matter (MS) lesions from paired T1w + FLAIR images using LST-AI.

    LST-AI registers the FLAIR to the T1w and into MNI space, runs an ensemble
    segmentation network, and (unless restricted) annotates lesions by region.
    It is run as an external program through one of two interchangeable backends:

    - ``"native"`` — the ``lst`` console script from a dedicated LST-AI virtualenv
      (located via ``$MEDMCP_LST_AI_BIN`` or PATH). Preferred: faster, no daemon,
      and it reuses any GPU directly. The ``greedy`` binary is downloaded once on
      first use if not already on PATH.
    - ``"docker"`` — the ``jqmcginnis/lst-ai`` image (greedy + HD-BET baked in).
      Used as a fallback when the native install is absent. Requires a running
      Docker daemon (and the NVIDIA container runtime for GPU).

    When ``backend`` is None the native backend is used if available, otherwise
    docker.

    Skull stripping: LST-AI normally strips skulls itself with its own pinned
    HD-BET. If you have already run this package's ``skull_strip`` tool on **both**
    the T1w and FLAIR, pass ``skull_stripped=True`` so LST-AI skips that step —
    this avoids the bundled HD-BET and is the recommended path in this ecosystem.

    Before calling, ask the user which compute device to use:
    - ``"cpu"`` — always available, slow (often >1 h).
    - ``"cuda"`` — NVIDIA GPU (uses ``gpu_id``), much faster.
    LST-AI has no Apple-Silicon (MPS) path. Default to ``"cpu"`` if unspecified.

    Args:
        t1_path: Absolute path to the T1-weighted NIfTI (.nii or .nii.gz), 3-D.
        flair_path: Absolute path to the FLAIR NIfTI (.nii or .nii.gz), 3-D.
        output_dir: Directory for LST-AI outputs. Defaults to the T1w's directory.
        device: ``"cpu"`` (default) or ``"cuda"``.
        gpu_id: GPU index used when ``device="cuda"``. Defaults to ``0``.
        skull_stripped: Set True only when both inputs are already brain-extracted;
            passes ``--skull-stripped`` to LST-AI.
        backend: ``"native"``, ``"docker"``, or None to auto-select.

    Returns:
        ``SegmentLesionsResult`` with the lesion mask path, the region-annotated
        map (if produced), and the full list of files LST-AI wrote.

    Raises:
        FileNotFoundError: If ``t1_path`` or ``flair_path`` does not exist.
        RuntimeError: If no backend is available or LST-AI fails.
        ValueError: If ``device`` is not ``"cpu"`` or ``"cuda"``.
    """
    if not t1_path.exists():
        raise FileNotFoundError(f"T1w input not found: {t1_path}")
    if not flair_path.exists():
        raise FileNotFoundError(f"FLAIR input not found: {flair_path}")

    out_dir = output_dir if output_dir is not None else t1_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _run_lstai.resolve_backend(backend)
    dev_flag = _run_lstai.device_flag(device, gpu_id)

    before = _nii_set(out_dir)
    env = os.environ.copy()

    if selected == "native":
        lst_bin = _run_lstai.native_lst_bin()
        assert lst_bin is not None  # guaranteed by resolve_backend
        greedy = _run_lstai.ensure_greedy()
        # LST-AI shells out to `greedy`; make sure it is discoverable on PATH.
        env["PATH"] = f"{Path(greedy).parent}{os.pathsep}{env.get('PATH', '')}"
        with tempfile.TemporaryDirectory(prefix="lst_temp_") as temp_dir:
            cmd = _run_lstai.build_native_command(
                lst_bin=lst_bin,
                t1_path=t1_path,
                flair_path=flair_path,
                output_dir=out_dir,
                temp_dir=Path(temp_dir),
                device=dev_flag,
                skull_stripped=skull_stripped,
                extra_args=[],
            )
            _invoke(cmd, env=env, label="LST-AI (native)")
    else:
        cmd = _run_lstai.build_docker_command(
            image=_run_lstai.docker_image(),
            t1_path=t1_path,
            flair_path=flair_path,
            output_dir=out_dir,
            device=dev_flag,
            skull_stripped=skull_stripped,
            extra_args=[],
        )
        _invoke(cmd, env=env, label="LST-AI (docker)")

    produced = sorted(_nii_set(out_dir) - before)
    if not produced:
        raise RuntimeError(f"LST-AI completed but no new NIfTI outputs were found in {out_dir}.")

    seg = _pick_segmentation(produced)
    annotated = _pick_annotated(produced)
    if seg is None:
        raise RuntimeError(f"Could not identify a lesion segmentation among: {produced}")

    return {
        "lesion_mask_path": str(seg),
        "annotated_path": str(annotated) if annotated else None,
        "output_files": [str(p) for p in produced],
        "t1_path": str(t1_path),
        "flair_path": str(flair_path),
        "backend": selected,
        "device": device,
        "skull_stripped": skull_stripped,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the lesion segmentation result as a compact key-value list:\n"
            "  Lesion mask: <lesion_mask_path>\n"
            "  Annotated:   <annotated_path>   (omit this line if null)\n"
            "  Backend:     <backend>\n"
            "  Device:      <device>\n"
            "Substitute values from the result dict. Omit internal keys.\n"
            "NEXT ACTION: Tell the user the lesion mask path. If a T1w→template "
            "registration exists, offer to warp the lesion mask into MNI space with "
            "apply_transform using interpolation='NearestNeighbor'. The tool already "
            "verified the outputs exist — do not attempt to recheck them."
        ),
    }


def _invoke(cmd: list[str], *, env: dict[str, str], label: str) -> None:
    """Run an LST-AI backend command, streaming stderr and raising on failure."""
    print(f"[medmcp-neuro] segment_lesions: starting {label} …", file=sys.stderr, flush=True)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=_TIMEOUT,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )


__all__ = ["SegmentLesionsResult", "segment_lesions"]
