"""Bundled brain atlases: registry, pinned sources, and on-demand preparation.

Only atlases whose licence allows redistribution are listed here (see NOTICE). Each
one is downloaded from a pinned source, checked against a SHA-256, and prepared into
``~/.medmcp_neuro_core/atlases/<name>/`` — the container image runs this at build
time so the tools work with networking denied; running from source it happens on
first use.

Prepared layout (identical for every atlas, so the tools need no per-atlas code)::

    <name>/
        labels.tsv        index<TAB>name<TAB>file   (file only for probabilistic atlases)
        atlas_dseg.nii.gz (label atlases)  |  maps/<file>.nii.gz (probabilistic atlases)
        ATTRIBUTION.txt   source, licence, citation, changes made
        <upstream licence file, where the source ships one>
        .complete         written last; its absence means "prepare again"

Deliberately absent (licence forbids redistribution or is unclear): JHU ICBM-DTI-81 /
JHU white-matter tractography, FSL XTRACT, Juelich, Brainnetome, Julich-Brain.
"""

import csv
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ATLAS_ROOT = Path.home() / ".medmcp_neuro_core" / "atlases"

AtlasName = Literal[
    "pandora_tractseg",
    "hcp1065",
    "schaefer2018_100",
    "schaefer2018_200",
    "schaefer2018_400",
]
AtlasKind = Literal["labels", "probabilistic"]

_PANDORA_COMMIT = "2caf44296666c13aa5c4d7d1c88d14fe553d9553"
_PANDORA_RAW = (
    f"https://raw.githubusercontent.com/MASILab/Pandora-WhiteMatterAtlas/{_PANDORA_COMMIT}"
)
_HCP1065_RELEASE = "https://github.com/frankyeh/data-atlas/releases/download/hcp1065"
_TEMPLATEFLOW = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym"
_CBIG_COMMIT = "634f676630929a71297852d01dd92a287103e861"


@dataclass(frozen=True)
class Source:
    """One pinned upstream file."""

    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class AtlasSpec:
    """A redistributable atlas and how to report it."""

    name: str
    kind: AtlasKind
    title: str
    space: str
    licence: str
    attribution: str
    sources: tuple[Source, ...]


_SCHAEFER_SHA: dict[int, tuple[str, str]] = {
    # n parcels: (dseg.nii.gz sha256, dseg.tsv sha256)
    100: (
        "6451253870267d695bce91b292feb60d4805eb3817fb1bd4f8dc626f901f3def",
        "a81a63f67fdaffd7b2b4b6c7df05f75b16899cba70100f464bbccba0a257cc6c",
    ),
    200: (
        "7395671a6045355bb773395ad6b8c35705cd8dcffd7f4b32cce656944f8b5ede",
        "32089161a6f8dc3c6dc5b545ba26ad65febbbf314088e194f28e1133649e6f3d",
    ),
    400: (
        "ae402a191ed5e8a37546f6fc25a30affd6813dbd01f2797c8db095f5d45d6795",
        "91fe503df22267bb6c9ab3b9e9e82eec0e08f496d4f9dcb9661e67d1abb14f07",
    ),
}


def _schaefer(n: int) -> AtlasSpec:
    dseg_sha, tsv_sha = _SCHAEFER_SHA[n]
    desc = f"{n}Parcels7Networks"
    return AtlasSpec(
        name=f"schaefer2018_{n}",
        kind="labels",
        title=f"Schaefer 2018 cortical parcellation, {n} parcels (7 networks)",
        space="MNI152NLin2009cAsym",
        licence="MIT (CBIG repository licence)",
        attribution=(
            f"Schaefer 2018 cortical parcellation ({n} parcels, 7 networks), "
            "MNI152NLin2009cAsym version from TemplateFlow. MIT licence, "
            "Copyright (c) 2016 Computational Brain Imaging Group (CBIG). "
            "Cite: Schaefer et al., Cerebral Cortex 28(9):3095-3114 (2018), "
            "doi:10.1093/cercor/bhx179."
        ),
        sources=(
            Source(
                f"{_TEMPLATEFLOW}/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_"
                f"desc-{desc}_dseg.nii.gz",
                dseg_sha,
                "upstream_dseg.nii.gz",
            ),
            Source(
                f"{_TEMPLATEFLOW}/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-{desc}_dseg.tsv",
                tsv_sha,
                "upstream_dseg.tsv",
            ),
            Source(
                f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/{_CBIG_COMMIT}/LICENSE.md",
                "0e38f0e8b304f13de2ce2911320faaa08ced178e031d3bbebc3e4769762e5f50",
                "LICENSE-CBIG.md",
            ),
        ),
    )


ATLASES: dict[str, AtlasSpec] = {
    "pandora_tractseg": AtlasSpec(
        name="pandora_tractseg",
        kind="probabilistic",
        title="Pandora white-matter atlas, TractSeg bundle definitions (72 bundles)",
        space="MNI152NLin2009cAsym",
        licence="CC BY 4.0",
        attribution=(
            "Pandora white matter atlas (TractSeg set, 72 probabilistic bundles incl. "
            'corticospinal tract L/R), Hansen et al., "Pandora: 4-D white matter bundle '
            'population-based atlases derived from diffusion MRI fiber tractography", '
            "Neuroinformatics 19:447-460 (2021), doi:10.1007/s12021-020-09497-1. "
            "Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); "
            "split from the 4-D volume into one 3-D file per bundle, values unchanged. "
            "Built from HCP, BLSA and VU data (HCP: WU-Minn Consortium, "
            "1U54MH091657)."
        ),
        sources=(
            Source(
                f"{_PANDORA_RAW}/TractSeg/TractSeg.nii.gz",
                "373f99472e6cca444b3f8763bc8ac0eced0fbb9214aa8df4b1cdfb944f80cbbc",
                "TractSeg.nii.gz",
            ),
            Source(
                f"{_PANDORA_RAW}/TractSeg/TractSeg_info.csv",
                "9cf03befb659a492d8fd67440f950edba1819f55f40594e4d306887e52849033",
                "TractSeg_info.csv",
            ),
            Source(
                f"{_PANDORA_RAW}/license.txt",
                "cbd5af318286b74656f145dff5091907cc23d6d8c9ad09e0291ec2497792ba41",
                "LICENSE-Pandora.txt",
            ),
        ),
    ),
    "hcp1065": AtlasSpec(
        name="hcp1065",
        kind="probabilistic",
        title="HCP1065 population-probability tract atlas (Yeh 2022, 64 tracts)",
        # The maps are in ICBM 2009a nonlinear asymmetric space; 2009a and 2009c share
        # MNI coordinates, so they are used in world coordinates as 2009c (approximate).
        space="MNI152NLin2009aAsym",
        licence="CC BY-SA 4.0",
        attribution=(
            'HCP1065 population-probability tract atlas, Yeh FC, "Population-based '
            "tract-to-region connectome of the human brain and its hierarchical "
            'topology", Nat Commun 13:4933 (2022), doi:10.1038/s41467-022-32595-4. '
            "Licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); "
            "files redistributed unmodified. Data were provided in part by the Human "
            "Connectome Project, WU-Minn Consortium (Principal Investigators: David Van "
            "Essen and Kamil Ugurbil; 1U54MH091657). Atlas space is ICBM 2009a; used "
            "here in MNI world coordinates as an approximation of 2009c."
        ),
        sources=(
            Source(
                f"{_HCP1065_RELEASE}/hcp1065_prob_coverage_nifti.zip",
                "4577abce53e05c732a0d52c88f4ee2d63488bb685f664faad2ea44aa90be8e8b",
                "hcp1065_prob_coverage_nifti.zip",
            ),
        ),
    ),
    "schaefer2018_100": _schaefer(100),
    "schaefer2018_200": _schaefer(200),
    "schaefer2018_400": _schaefer(400),
}


@dataclass(frozen=True)
class AtlasEntry:
    """One region (label atlas) or one tract map (probabilistic atlas)."""

    index: int
    name: str
    path: Path | None  # probabilistic: the map file; labels: None


@dataclass(frozen=True)
class PreparedAtlas:
    """A prepared atlas on disk."""

    spec: AtlasSpec
    root: Path
    dseg_path: Path | None
    entries: tuple[AtlasEntry, ...]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(src: Source, dest_dir: Path) -> Path:
    dest = dest_dir / src.filename
    urllib.request.urlretrieve(src.url, dest)
    digest = _sha256(dest)
    if digest != src.sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {src.url}: expected {src.sha256}, got {digest}. "
            "The upstream file changed; re-check its licence before updating the pin."
        )
    return dest


def _write_labels_tsv(path: Path, rows: list[tuple[int, str, str]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["index", "name", "file"])
        writer.writerows(rows)


def _prepare_pandora(dl: Path, out: Path) -> None:
    import nibabel as nib
    import numpy as np

    with open(dl / "TractSeg_info.csv", newline="") as fh:
        info = list(csv.DictReader(fh))
    img = nib.load(str(dl / "TractSeg.nii.gz"))  # pyright: ignore[reportUnknownMemberType]
    data = np.asarray(img.dataobj, dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
    if data.ndim != 4 or data.shape[3] != len(info):
        raise RuntimeError(f"Unexpected Pandora TractSeg shape {data.shape} for {len(info)} rows")
    affine = img.affine  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
    maps = out / "maps"
    maps.mkdir()
    rows: list[tuple[int, str, str]] = []
    for row in info:
        idx = int(row["ID"])
        fname = f"{row['File Name']}.nii.gz"
        vol = nib.Nifti1Image(data[..., idx], affine)  # pyright: ignore[reportUnknownArgumentType]
        vol.header.set_xyzt_units("mm")  # pyright: ignore[reportUnknownMemberType]
        nib.save(vol, str(maps / fname))  # pyright: ignore[reportUnknownMemberType]
        rows.append((idx + 1, row["Bundle Name"], f"maps/{fname}"))
    _write_labels_tsv(out / "labels.tsv", rows)
    shutil.copy2(dl / "LICENSE-Pandora.txt", out / "LICENSE-Pandora.txt")


def _hcp_name(stem: str) -> str:
    """``Corticospinal_Tract_L`` -> ``Corticospinal Tract left``."""
    parts = stem.split("_")
    side = {"L": "left", "R": "right"}.get(parts[-1])
    words = parts[:-1] if side else parts
    return " ".join([*words, side] if side else words)


def _prepare_hcp1065(dl: Path, out: Path) -> None:
    maps = out / "maps"
    maps.mkdir()
    with zipfile.ZipFile(dl / "hcp1065_prob_coverage_nifti.zip") as zf:
        members = sorted(
            m for m in zf.namelist() if m.startswith("prob/") and m.endswith(".nii.gz")
        )
        for m in members:
            with zf.open(m) as src, open(maps / Path(m).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    rows = [
        (i + 1, _hcp_name(Path(m).name[: -len(".nii.gz")]), f"maps/{Path(m).name}")
        for i, m in enumerate(members)
    ]
    _write_labels_tsv(out / "labels.tsv", rows)


def _prepare_schaefer(dl: Path, out: Path) -> None:
    shutil.copy2(dl / "upstream_dseg.nii.gz", out / "atlas_dseg.nii.gz")
    rows: list[tuple[int, str, str]] = []
    with open(dl / "upstream_dseg.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append((int(row["index"]), row["name"], ""))
    _write_labels_tsv(out / "labels.tsv", rows)
    shutil.copy2(dl / "LICENSE-CBIG.md", out / "LICENSE-CBIG.md")


def _attribution_text(spec: AtlasSpec) -> str:
    lines = [
        spec.title,
        f"Licence: {spec.licence}",
        f"Space: {spec.space}",
        "",
        spec.attribution,
        "",
        "Sources (SHA-256 verified at download):",
        *(f"  {s.url}  sha256={s.sha256}" for s in spec.sources),
    ]
    return "\n".join(lines) + "\n"


def prepare_atlas(name: str, root: Path | None = None) -> Path:
    """Download (pinned + checksummed) and prepare one atlas; return its directory.

    Idempotent: an already-prepared atlas (``.complete`` present) is returned as is.
    """
    spec = ATLASES.get(name)
    if spec is None:
        raise ValueError(f"Unknown atlas {name!r}. Available: {', '.join(ATLASES)}")
    base = root if root is not None else ATLAS_ROOT
    out = base / name
    if (out / ".complete").exists():
        return out

    print(f"[medmcp-neuro-core] Preparing atlas {name} in {out} …", file=sys.stderr, flush=True)
    base.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        dl = Path(tmp) / "download"
        stage = Path(tmp) / name
        dl.mkdir()
        stage.mkdir()
        for src in spec.sources:
            _download(src, dl)
        if name == "pandora_tractseg":
            _prepare_pandora(dl, stage)
        elif name == "hcp1065":
            _prepare_hcp1065(dl, stage)
        else:
            _prepare_schaefer(dl, stage)
        (stage / "ATTRIBUTION.txt").write_text(_attribution_text(spec))
        (stage / ".complete").touch()
        stage.rename(out)
    return out


def prepare_all(root: Path | None = None) -> None:
    """Prepare every bundled atlas (the container image calls this at build time)."""
    for name in ATLASES:
        prepare_atlas(name, root)


def load_atlas(name: str, root: Path | None = None) -> PreparedAtlas:
    """Return a prepared atlas, preparing it first if needed."""
    spec = ATLASES.get(name)
    if spec is None:
        raise ValueError(f"Unknown atlas {name!r}. Available: {', '.join(ATLASES)}")
    out = prepare_atlas(name, root)
    entries: list[AtlasEntry] = []
    with open(out / "labels.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            file = row.get("file") or ""
            entries.append(AtlasEntry(int(row["index"]), row["name"], out / file if file else None))
    dseg = out / "atlas_dseg.nii.gz" if spec.kind == "labels" else None
    return PreparedAtlas(spec=spec, root=out, dseg_path=dseg, entries=tuple(entries))
