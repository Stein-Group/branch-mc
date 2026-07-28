"""Rung S3 comparison harness (Q12): engine vs RASPA2, side by side.

RASPA runs on the cluster (Dipesh); this reads the engine's run_mof CSV and a
small table of RASPA reference values, equilibrates the engine series (Geweke),
and prints per-species loading, selectivity, and the z-score of the difference so
sign-off is mechanical. No simulation here -- pure comparison, so it is unit
tested on canned numbers.

RASPA reference file: whitespace/CSV columns `species value stderr`, one guest
per line (values in the same unit as the engine column being compared, e.g.
mol/kg or <N>).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from scripts.mof.analysis import summarize_series


def zscore(a: float, sa: float, b: float, sb: float) -> float:
    """Standard score of (a - b) given independent standard errors."""
    denom = math.sqrt(sa * sa + sb * sb)
    return float("nan") if denom == 0.0 else (a - b) / denom


def compare(
    engine: dict[str, dict[str, float]],
    raspa: dict[str, tuple[float, float]],
) -> list[dict[str, float | str]]:
    """One row per species: engine mean +/- stderr, RASPA value +/- stderr, z.
    `engine[species] = {mean, stderr}`; `raspa[species] = (value, stderr)`."""
    rows: list[dict[str, float | str]] = []
    for s in engine:
        em, es = engine[s]["mean"], engine[s]["stderr"]
        rv, rs = raspa.get(s, (float("nan"), 0.0))
        rows.append(
            {
                "species": s,
                "engine": em,
                "engine_err": es,
                "raspa": rv,
                "raspa_err": rs,
                "z": zscore(em, es, rv, rs),
            }
        )
    return rows


def selectivity(engine: dict[str, dict[str, float]], a: str, b: str) -> float:
    """Loading ratio N_a / N_b from the equilibrated engine means."""
    nb = engine[b]["mean"]
    return float("nan") if nb == 0 else engine[a]["mean"] / nb


def engine_from_csv(path: Path, species: list[str]) -> dict[str, dict[str, float]]:
    """Equilibrated per-species mean + blocked stderr from a run_mof CSV."""
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, float]] = {}
    for s in species:
        series = np.asarray([float(r[f"N_{s}"]) for r in rows])
        out[s] = summarize_series(series)
    return out


def _read_raspa(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        out[parts[0]] = (float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0)
    return out


def format_table(rows: list[dict[str, float | str]]) -> str:
    head = f"{'species':>8} {'engine':>12} {'RASPA':>12} {'z':>8}"
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['species']:>8} {r['engine']:>7.4f}+/-{r['engine_err']:<5.4f}"
            f" {r['raspa']:>7.4f}+/-{r['raspa_err']:<5.4f} {r['z']:>8.2f}"
        )
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine-csv", type=Path, required=True)
    p.add_argument(
        "--raspa", type=Path, required=True, help="species value stderr file"
    )
    p.add_argument("--species", default="Xe,Kr")
    a = p.parse_args(argv)
    species = [s.strip() for s in a.species.split(",")]
    eng = engine_from_csv(a.engine_csv, species)
    ras = _read_raspa(a.raspa)
    rows = compare(eng, ras)
    print(format_table(rows))
    if len(species) >= 2:
        print(
            f"\nselectivity S({species[0]}/{species[1]}) engine = "
            f"{selectivity(eng, species[0], species[1]):.3f}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
