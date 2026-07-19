"""V0: run the stored configurations through RASPA2 and compare against our oracle.

This is rung V0. ref_energy.py (P0.5) is the executable definition of the
Hamiltonian; RASPA is the independent implementation. If they disagree, our
Hamiltonian is not the one the anchor's numbers came from, and no engine work
should start. Gate S1 hangs on this.

D4 pins RASPA3 as primary and RASPA2 (2.0.4x) as the tie-breaker. RASPA3 needs a
C++23 toolchain that is not available here; RASPA2 builds, and it is also the code
the anchor itself ran (their ref 75). So this checks against the version the
published numbers came from, and the RASPA3 leg remains for Dipesh.

Exact placement goes through RASPA2's own restart file rather than a
reimplementation of its format: run once so it writes one, patch the
`Adsorbate-atom-position` lines to our coordinates, rerun with zero cycles and
RestartFile yes. RASPA then reports the energy of exactly the configuration we
asked about.

What is compared: RASPA's Host-Adsorbate VDW energy against our guest_framework
term, and its Adsorbate-Adsorbate VDW against our guest_guest. Host-host is
excluded on both sides (rigid framework, so it is a constant RASPA reports
separately and never varies).

    python -m scripts.mof.run_v0 --forcefield ff/forcefield_A.yaml \\
        --cif structures/MFM-300-In.cif --v0 raspa/v0 --work /tmp/v0run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.mof.raspa2_input import write_case
from scripts.mof.units import ForceField, load_forcefield

# RASPA2 prints the energy-status block in KELVIN already: constants.c defines
# ENERGY_TO_KELVIN = ENERGY_CONVERSION_FACTOR * N_A / R = 1.202724138894716 and
# applies it on output. Converting again here put every term out by exactly that
# factor -- uniformly, across all six configurations and both energy terms, which
# is the signature of a unit error rather than of physics. No conversion.
_ENERGY_RE = {
    "host_adsorbate_vdw": re.compile(r"Host/Adsorbate VDW energy:\s+([-\d.eE+]+)"),
    "adsorbate_adsorbate_vdw": re.compile(
        r"Adsorbate/Adsorbate VDW energy:\s+([-\d.eE+]+)"
    ),
}


def _run(case: Path, raspa_dir: str) -> str:
    env = dict(os.environ, RASPA_DIR=raspa_dir)
    proc = subprocess.run(
        [f"{raspa_dir}/bin/simulate", "simulation.input"],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    out = sorted((case / "Output" / "System_0").glob("*.data"))
    if proc.returncode != 0 or not out:
        raise RuntimeError(
            f"RASPA2 failed in {case}: {proc.stdout[-800:]}{proc.stderr[-800:]}"
        )
    return out[0].read_text()


def _patch_restart(restart: Path, positions: np.ndarray) -> None:
    """Replace the adsorbate coordinates, leaving every other byte alone."""
    lines = restart.read_text().splitlines()
    k = 0
    for i, line in enumerate(lines):
        if line.startswith("Adsorbate-atom-position:"):
            head = line.split(":")[0]
            idx = line.split(":")[1].split()[:2]
            p = positions[k]
            lines[i] = (
                f"{head}: {idx[0]:>5s} {idx[1]:>5s}"
                f"{p[0]:>22.12f}{p[1]:>22.12f}{p[2]:>22.12f}"
            )
            k += 1
    if k != len(positions):
        raise RuntimeError(
            f"{restart}: patched {k} positions, expected {len(positions)}"
        )
    restart.write_text("\n".join(lines) + "\n")


def single_point(
    name: str,
    positions: np.ndarray,
    types: list[str],
    ff: ForceField,
    cif: Path,
    work: Path,
    T: float,
    unit_cells: tuple[int, int, int],
    present: tuple[str, ...],
    raspa_dir: str,
) -> dict[str, float]:
    """One configuration through RASPA2, guests exactly where we put them."""
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    guests = {g: 1.0 / len(counts) for g in counts}

    case = work / name
    if case.exists():
        shutil.rmtree(case)
    # pass 1: let RASPA create the molecules and write a restart file
    write_case(
        case, ff, cif, T, 1.0e5, guests, unit_cells, 0, 0, present, counts=counts
    )
    _run(case, raspa_dir)

    restart = next((case / "Restart" / "System_0").glob("restart_*"))
    # RASPA reads its initial restart from RestartInitial/, so move it there
    init = case / "RestartInitial" / "System_0"
    init.mkdir(parents=True, exist_ok=True)
    target = init / restart.name
    shutil.copy(restart, target)
    _patch_restart(target, positions)

    # pass 2: zero cycles, reading the patched positions back, and creating
    # NOTHING. RestartFile and CreateNumberOfMolecules are additive in RASPA2: it
    # reads the restart AND then creates the requested molecules on top. Leaving
    # the count at 1 gave two Xe -- ours at the site plus one dropped at random --
    # which read as a host-guest energy 2.29x ours and a guest-guest term where
    # the configuration has no pair at all.
    write_case(
        case,
        ff,
        cif,
        T,
        1.0e5,
        guests,
        unit_cells,
        0,
        0,
        present,
        counts=dict.fromkeys(counts, 0),
        restart=True,
    )
    shutil.rmtree(case / "Output", ignore_errors=True)
    text = _run(case, raspa_dir)

    got = {}
    for key, rx in _ENERGY_RE.items():
        m = rx.search(text)
        if m is None:
            raise RuntimeError(f"{name}: could not find {key} in RASPA output")
        got[key] = float(m.group(1))
    return got


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--v0", type=Path, required=True, help="dir with v0_reference.json")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument("--unit-cells", default="2,2,2")
    p.add_argument("--present", default="Xe,Kr,In,C,O,H")
    p.add_argument("--raspa-dir", default="/opt/raspa2")
    p.add_argument("--only", default=None, help="run one configuration by name")
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield)
    ref = json.loads((a.v0 / "v0_reference.json").read_text())
    uc = tuple(int(x) for x in a.unit_cells.split(","))
    a.work.mkdir(parents=True, exist_ok=True)

    print(f"RASPA2 at {a.raspa_dir}; force field {ff.name} ({ff.sha256[:12]})")
    print(
        f"{'configuration':<28} {'term':<18} {'ref_energy [K]':>16} {'RASPA2 [K]':>16} {'rel diff':>12}"
    )
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for cfg in ref["configurations"]:
        if a.only and cfg["name"] != a.only:
            continue
        got = single_point(
            cfg["name"],
            np.array(cfg["positions"]),
            cfg["types"],
            ff,
            a.cif,
            a.work,
            a.T,
            uc,  # type: ignore[arg-type]
            tuple(a.present.split(",")),
            a.raspa_dir,
        )
        pairs = [
            ("guest_framework", got["host_adsorbate_vdw"]),
            ("guest_guest", got["adsorbate_adsorbate_vdw"]),
        ]
        for term, raspa in pairs:
            mine = cfg["ref_energy_kelvin"][term]
            denom = max(abs(mine), 1.0)
            rel = abs(mine - raspa) / denom
            worst = max(worst, rel)
            print(
                f"{cfg['name']:<28} {term:<18} {mine:16.6f} {raspa:16.6f} {rel:12.2e}"
            )
            rows.append(
                {
                    "name": cfg["name"],
                    "term": term,
                    "ref": mine,
                    "raspa2": raspa,
                    "rel": rel,
                }
            )
    print(f"\nworst relative difference: {worst:.3e}")
    (a.work / "v0_comparison.json").write_text(
        json.dumps(
            {"rung": "V0", "code": "RASPA2", "worst_rel": worst, "rows": rows}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
