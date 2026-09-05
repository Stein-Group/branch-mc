"""Rung V1: Henry coefficients in MFM-300(In), engine vs RASPA2.

Henry's constant is the infinite-dilution limit, so it needs no GCMC loop, only
Widom insertion into the framework (A3.4) plus the K* -> mol/kg/Pa conversion
(v1v2_units). The 1:1 comparison is against RASPA2's own Widom, since that is the
code the anchor ran; the anchor's Table S4 is a secondary reference (their
DFT-optimized structure differs from the CIF here, so exact agreement is not
expected and a ~1.5x offset is ordinary).

    python -m scripts.mof.v1_henry --forcefield ff/forcefield_A.yaml \\
        --cif structures/MFM-300-In.cif --T 298 --samples 500000

Reports K* and K_H for Xe and Kr, the Henry selectivity K_H(Xe)/K_H(Kr), and --
if a RASPA2 build is found -- runs RASPA2's Widom on the same case and prints both
side by side.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.mof.build_framework import build, write_extxyz
from scripts.mof.units import load_forcefield, reduce_tables
from scripts.mof.v1v2_units import henry_mol_per_kg_per_pa


def engine_henry(
    cif: Path,
    ff_path: Path,
    T: float,
    samples: int,
    guests: tuple[str, ...] = ("Xe", "Kr"),
    reference: str = "Xe",
    seed: int = 1,
) -> dict:
    from gcmc.analysis.widom import widom_henry
    from gcmc.forcefield.lj_table import FrameworkField
    from gcmc.forcefield.pair_table import PairTable
    from gcmc.run.io_framework import load_framework_typed

    ff = load_forcefield(ff_path)
    red = reduce_tables(ff, T, reference=reference)
    assert ff.core_floor_c is not None, "core floor not measured; run P0.6"
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )

    atoms, meta = build(cif, ff)
    mass_amu = float(atoms.get_masses().sum())
    cell_vol = float(np.prod(meta["supercell_angstrom"]))

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ex = Path(td) / "framework.extxyz"
        write_extxyz(atoms, ex)
        pos, tids = load_framework_typed(ex, table)
        box = np.asarray(meta["supercell_angstrom"]) / red.sigma_ref
        field = FrameworkField(
            pos / red.sigma_ref, tids, table, box, ff.rc / red.sigma_ref
        )
        beta = 1.0 / red.T_star
        out = {}
        for g in guests:
            w = widom_henry(
                field,
                table.index(g),
                beta,
                np.random.default_rng(seed),
                n_samples=samples,
            )
            kstar = w["henry_star"]
            out[g] = {
                "k_star": kstar,
                "k_star_stderr": w["stderr"],
                "k_henry": henry_mol_per_kg_per_pa(kstar, mass_amu, cell_vol, T),
            }
        return out


def raspa2_widom(
    cif: Path, ff_path: Path, T: float, guest: str, cycles: int, raspa_dir: str
) -> float | None:
    """RASPA2's average Widom Rosenbluth weight for one guest, or None if RASPA2
    is not built. Cross-check for the engine's K*."""
    if not Path(raspa_dir, "bin", "simulate").exists():
        return None
    import os
    import re
    import tempfile

    from scripts.mof.raspa2_input import write_case

    ff = load_forcefield(ff_path)
    with tempfile.TemporaryDirectory() as td:
        case = Path(td)
        write_case(
            case,
            ff,
            cif,
            T,
            1.0e5,
            {guest: 1.0},
            (2, 2, 2),
            cycles,
            0,
            ("Xe", "Kr", "In", "C", "O", "H"),
        )
        # switch the swap move for a Widom move
        sim = (case / "simulation.input").read_text()
        sim = sim.replace(
            "SwapProbability           1.0",
            "SwapProbability           0.0\n            WidomProbability          1.0",
        )
        (case / "simulation.input").write_text(sim)
        env = dict(os.environ, RASPA_DIR=raspa_dir)
        subprocess.run(
            [f"{raspa_dir}/bin/simulate", "simulation.input"],
            cwd=case,
            env=env,
            capture_output=True,
            timeout=1800,
        )
        data = next((case / "Output" / "System_0").glob("*.data")).read_text()
        m = re.search(r"Average Widom Rosenbluth-weight:\s+([-\d.eE+]+)", data)
        return float(m.group(1)) if m else None


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument("--samples", type=int, default=500_000)
    p.add_argument("--raspa-dir", default="/opt/raspa2")
    p.add_argument("--raspa-cycles", type=int, default=5000)
    p.add_argument("--no-raspa", action="store_true")
    p.add_argument("--guests", default="Xe,Kr", help="species names, comma-separated")
    p.add_argument(
        "--reference",
        default="Xe",
        help="species whose self-pair sets the reduction scale (P0.4)",
    )
    a = p.parse_args(argv)

    guests = tuple(g.strip() for g in a.guests.split(",") if g.strip())
    eng = engine_henry(
        a.cif, a.forcefield, a.T, a.samples, guests=guests, reference=a.reference
    )
    print(f"Rung V1: Henry coefficients in {a.cif.stem} at {a.T:g} K\n")
    print(f"{'guest':<6} {'K* (engine)':>16} {'K_H [mol/kg/Pa]':>18}")
    for g in guests:
        e = eng[g]
        print(
            f"{g:<6} {e['k_star']:>10.2f} +/-{e['k_star_stderr']:<4.1f} {e['k_henry']:>18.4e}"
        )
    if len(guests) >= 2:
        g0, g1 = guests[0], guests[1]
        sel = eng[g0]["k_henry"] / eng[g1]["k_henry"]
        anchor = "   (anchor Table S4: 9.94)" if (g0, g1) == ("Xe", "Kr") else ""
        print(f"\nHenry selectivity K_H({g0})/K_H({g1}) = {sel:.2f}{anchor}")

    if not a.no_raspa:
        print("\nRASPA2 cross-check (same yaml, same structure):")
        for g in guests:
            w = raspa2_widom(a.cif, a.forcefield, a.T, g, a.raspa_cycles, a.raspa_dir)
            if w is None:
                print(f"  {g}: RASPA2 not available at {a.raspa_dir}")
            else:
                rel = abs(w - eng[g]["k_star"]) / w
                print(
                    f"  {g}: RASPA2 <W> = {w:.3f}   engine K* = {eng[g]['k_star']:.3f}   rel diff {rel:.2e}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
