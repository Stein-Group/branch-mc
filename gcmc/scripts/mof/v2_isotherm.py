"""Rung V2: the single-component Xe isotherm in MFM-300(In), engine vs RASPA2.

Sweeps pressure, turns each pressure into a reduced activity through Peng-Robinson
(so both codes sit on the same state point, G1), runs the MOF metropolis loop
(Phase 3) to equilibrium, and reports uptake in mol/kg. The 1:1 comparison is
RASPA2 GCMC at the same pressures; the anchor's Figure 6c is the secondary
reference.

Equilibration is the cost here, not correctness -- the loop's energy is already
gated to ref_energy.py and RASPA2 (V0), so V2 is about running long enough that
the mean loading has converged, not about whether each step is right. The driver
therefore reports a block standard error over the second half of the run and
leaves the cycle budget to the caller.

    python -m scripts.mof.v2_isotherm --forcefield ff/forcefield_A.yaml \\
        --cif structures/MFM-300-In.cif --T 298 --pressures 1e4,1e5,1e6 \\
        --windows 400 --window-cycles 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gcmc.mof_field import FrameworkField
from gcmc.potentials.pair_table import PairTable
from scripts.mof.build_framework import build, write_extxyz
from scripts.mof.units import load_forcefield, peng_robinson, reduce_tables
from scripts.mof.v1v2_units import loading_mol_per_kg, z_star_per_pa


def run_point(
    field: FrameworkField,
    table: PairTable,
    guest_type: int,
    beta: float,
    z_star: float,
    box: np.ndarray,
    volume: float,
    n_cells: float,
    rc: float,
    windows: int,
    window_cycles: int,
    seed: int,
) -> dict:
    """One pressure point: equilibrate, then average loading over the back half."""
    from gcmc.mof_metropolis import advance_metropolis_mof

    rng = np.random.default_rng(seed)
    pos = np.empty((0, box.shape[0]))
    ns = []
    for w in range(windows):
        res = advance_metropolis_mof(
            pos,
            rng,
            field=field,
            table=table,
            guest_type=guest_type,
            beta=beta,
            z_star=z_star,
            box=box,
            volume=volume,
            n_cells=n_cells,
            upper_cutoff=rc,
            move_factor=1.0,
            n_neigh=40,
            window=window_cycles,
        )
        pos = res.positions
        ns.append(res.n)
    back = np.array(ns[windows // 2 :], float)
    return {
        "n_mean": float(back.mean()),
        "n_stderr": float(back.std(ddof=1) / np.sqrt(len(back))),
    }


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument("--pressures", default="1e4,1e5,1e6", help="Pa, comma-separated")
    p.add_argument("--windows", type=int, default=400)
    p.add_argument("--window-cycles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    from gcmc.io_framework import load_framework_typed

    ff = load_forcefield(a.forcefield)
    red = reduce_tables(ff, a.T)
    assert ff.core_floor_c is not None, "core floor not measured; run P0.6"
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    atoms, meta = build(a.cif, ff)
    mass_amu = float(atoms.get_masses().sum())

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
    volume = float(np.prod(box))
    n_cells = float(np.prod(meta["replication"]))
    xe = table.index("Xe")
    sr = red.sigma_ref

    print(f"Rung V2: Xe isotherm in {a.cif.stem} at {a.T:g} K")
    print(f"{'P [bar]':>9} {'phi':>8} {'z*':>12} {'<N>':>10} {'loading [mol/kg]':>18}")
    for P in [float(x) for x in a.pressures.split(",")]:
        phi = peng_robinson(ff, a.T, P, {"Xe": 1.0}).phi["Xe"]
        z_star = phi * P * z_star_per_pa(sr, a.T)
        r = run_point(
            field,
            table,
            xe,
            beta,
            z_star,
            box,
            volume,
            n_cells,
            ff.rc / sr,
            a.windows,
            a.window_cycles,
            a.seed,
        )
        loading = loading_mol_per_kg(r["n_mean"], mass_amu)
        print(
            f"{P / 1e5:>9.3g} {phi:>8.4f} {z_star:>12.4e} "
            f"{r['n_mean']:>7.2f}+/-{r['n_stderr']:<4.2f} {loading:>18.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
