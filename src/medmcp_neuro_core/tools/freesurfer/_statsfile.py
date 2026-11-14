"""Reading FreeSurfer's ``.stats`` files, exactly as FreeSurfer wrote them.

Every number this stack reports comes from these files (written by FreeSurfer's
mri_segstats and mris_anatomical_stats); nothing is recomputed. The format:

- ``# Measure <key>, <ShortName>, <description>, <value>, <unit>`` lines with the
  global measures (eTIV, BrainSegVol, MeanThickness, WhiteSurfArea, ...);
- a ``# ColHeaders <name> <name> ...`` line naming the table's columns;
- whitespace-separated table rows (structure names contain no spaces).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Measure:
    """One ``# Measure`` line."""

    key: str
    name: str
    description: str
    value: float
    unit: str


@dataclass
class StatsFile:
    """A parsed ``.stats`` file."""

    path: Path
    measures: list[Measure] = field(default_factory=list[Measure])
    columns: list[str] = field(default_factory=list[str])
    rows: list[dict[str, str]] = field(default_factory=list[dict[str, str]])

    def measure(self, name: str) -> Measure | None:
        """The measure with this short name (``"eTIV"``, ``"MeanThickness"``), if any."""
        return next((m for m in self.measures if m.name == name), None)


def read_stats(path: Path) -> StatsFile:
    """Parse a FreeSurfer ``.stats`` file.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file has no ``# ColHeaders`` line.
    """
    stats = StatsFile(path=path)
    with path.open(errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("# Measure "):
                fields = [f.strip() for f in line[len("# Measure ") :].split(",")]
                if len(fields) >= 5:
                    try:
                        value = float(fields[3])
                    except ValueError:
                        continue
                    stats.measures.append(
                        Measure(fields[0], fields[1], fields[2], value, fields[4])
                    )
            elif line.startswith("# ColHeaders"):
                stats.columns = line.split()[2:]
            elif line.startswith("#") or not line.strip():
                continue
            elif stats.columns:
                values = line.split()
                if len(values) == len(stats.columns):
                    stats.rows.append(dict(zip(stats.columns, values, strict=True)))
    if not stats.columns:
        raise ValueError(f"{path} has no '# ColHeaders' line; not a FreeSurfer stats table")
    return stats
