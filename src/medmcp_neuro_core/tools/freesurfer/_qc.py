"""A quick visual check of a recon-all result: aparc+aseg over the T1.

Three orthogonal slices through the middle of the segmentation, with FreeSurfer's
own label colours (FreeSurferColorLUT.txt). Enough to spot a failed skull strip, a
misregistered atlas or a segmentation that leaks into the skull; not a substitute
for checking the surfaces in freeview.
"""

import os
from pathlib import Path
from typing import Any, cast


def _lut(path: Path) -> dict[int, tuple[float, float, float]]:
    """Label id -> RGB (0-1) from a FreeSurfer colour table."""
    colours: dict[int, tuple[float, float, float]] = {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return colours
    for line in lines:
        parts = line.split()
        if len(parts) < 5 or line.lstrip().startswith("#"):
            continue
        try:
            label, r, g, b = int(parts[0]), int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        colours[label] = (r / 255, g / 255, b / 255)
    return colours


def qc_snapshot(subject_dir: Path, destination: Path) -> Path:
    """Write a PNG of aparc+aseg over T1 (axial, coronal, sagittal) and return its path.

    Raises:
        FileNotFoundError: ``mri/T1.mgz`` or ``mri/aparc+aseg.mgz`` is missing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    t1_path = subject_dir / "mri" / "T1.mgz"
    seg_path = subject_dir / "mri" / "aparc+aseg.mgz"
    for p in (t1_path, seg_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} is missing; the recon-all run is not complete")

    def load(p: Path) -> Any:
        # nibabel is untyped; everything it returns is treated as Any.
        img = cast("Any", nib.load(str(p)))  # pyright: ignore[reportUnknownMemberType]
        canonical = cast("Any", nib.as_closest_canonical(img))  # pyright: ignore[reportUnknownMemberType]
        return cast("Any", np.asarray(canonical.dataobj))

    t1 = load(t1_path).astype(np.float32)
    seg = load(seg_path).astype(np.int32)
    inside = np.argwhere(seg > 0)
    centre = inside.mean(axis=0).round().astype(int) if inside.size else np.array(seg.shape) // 2

    home = Path(os.environ.get("FREESURFER_HOME", "/opt/freesurfer"))
    colours = _lut(home / "FreeSurferColorLUT.txt")
    labels = np.unique(seg)
    rgba = np.zeros((*seg.shape, 4), dtype=np.float32)
    for label in labels:
        if label == 0:
            continue
        r, g, b = colours.get(int(label), ((label * 37 % 255) / 255, 0.5, 0.5))
        rgba[seg == label] = (r, g, b, 0.45)

    vmax = float(np.percentile(t1[t1 > 0], 99.5)) if np.any(t1 > 0) else 1.0  # pyright: ignore[reportUnknownArgumentType]
    # RAS canonical: axis 0 = x (left->right), 1 = y (posterior->anterior), 2 = z.
    views = [
        ("Axial", t1[:, :, centre[2]].T, rgba[:, :, centre[2]].transpose(1, 0, 2)),
        ("Coronal", t1[:, centre[1], :].T, rgba[:, centre[1], :].transpose(1, 0, 2)),
        ("Sagittal", t1[centre[0], :, :].T, rgba[centre[0], :, :].transpose(1, 0, 2)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor="black")  # pyright: ignore[reportUnknownMemberType]
    for ax, (title, base, overlay) in zip(axes, views, strict=True):
        ax.imshow(base, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax.imshow(overlay, origin="lower")
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
    fig.suptitle(  # pyright: ignore[reportUnknownMemberType]
        f"{subject_dir.name}: aparc+aseg on T1 (FreeSurfer; research use only)",
        color="white",
        fontsize=11,
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=110, facecolor="black")  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)
    return destination
