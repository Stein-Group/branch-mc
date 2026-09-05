"""Rung V3: mixture isotherms, engine vs RASPA2. Any guests, framework or bulk.

The single-component rungs (V1 Henry, V2 isotherm) fix the framework and the
guest-guest Hamiltonian. V3 is the first test that the multi-species machinery
(Phase 4) samples the right JOINT distribution: at a fixed gas composition it
must reproduce RASPA2's per-species uptake and, more sharply, the selectivity,
which is a ratio and so cancels most of what V2 already checked while exposing
any bias in the species-selection or identity-swap acceptance.

The guests, their gas-phase mole fractions, the forcefield, the structure, and
the reduction reference are all arguments: any species defined in the forcefield
yaml (LJ block + eos criticals) can be mixed, in any count. Each state point
turns total pressure and composition into per-species partial fugacities through
the mixture Peng-Robinson (G1, same state point as RASPA2), converts each to a
reduced activity, and runs the multi-species loop to equilibrium. With
--bulk-box the framework is empty and the same loop samples the bulk mixture
(useful as a self-check: the dilute limit must approach the ideal <N_s> = z*_s V).

    python -m scripts.mof.v3_mixture --forcefield ff/forcefield_A.yaml \\
        --cif structures/MFM-300-In.cif --T 298 --pressures 1e4,1e5,1e6 \\
        --guests Xe,Kr --y 0.2,0.8 --windows 500 --window-cycles 200

    python -m scripts.mof.v3_mixture --forcefield ff/forcefield_A.yaml \\
        --bulk-box 30 --T 298 --pressures 1e5 --guests Xe,Kr --y 0.5,0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gcmc.engine.framework import advance_metropolis_mof_ms
from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable
from scripts.mof.system_setup import setup_system
from scripts.mof.units import peng_robinson
from scripts.mof.v1v2_units import loading_mol_per_kg, z_star_per_pa


def run_point_ms(
    field: FrameworkField,
    table: PairTable,
    z_star: dict[int, float],
    beta: float,
    box: np.ndarray,
    volume: float,
    n_cells: float,
    rc: float,
    windows: int,
    window_cycles: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    """One state point: equilibrate the mixture, then average each species'
    count over the back half. Returns per-type mean and block standard error."""
    rng = np.random.default_rng(seed)
    pos = np.empty((0, box.shape[0]))
    spc = np.empty(0, dtype=np.int64)
    guest_types = sorted(z_star)
    history: dict[int, list[int]] = {t: [] for t in guest_types}
    for _ in range(windows):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star=z_star,
            beta=beta,
            box=box,
            volume=volume,
            n_cells=n_cells,
            upper_cutoff=rc,
            move_factor=1.0,
            n_neigh=40,
            window=window_cycles,
        )
        pos, spc = res.positions, res.species
        assert res.n_by_type is not None
        for t in guest_types:
            history[t].append(res.n_by_type[t])
    out: dict[int, dict[str, float]] = {}
    for t in guest_types:
        back = np.array(history[t][windows // 2 :], float)
        out[t] = {
            "n_mean": float(back.mean()),
            "n_stderr": float(back.std(ddof=1) / np.sqrt(len(back)))
            if len(back) > 1
            else 0.0,
        }
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, default=None, help="framework structure")
    p.add_argument(
        "--bulk-box",
        type=float,
        default=None,
        help="cubic box side in angstrom for a bulk (no framework) run",
    )
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument("--pressures", default="1e4,1e5,1e6", help="Pa, comma-separated")
    p.add_argument("--guests", default="Xe,Kr", help="species names, comma-separated")
    p.add_argument(
        "--y", default=None, help="gas mole fractions, comma-separated; default uniform"
    )
    p.add_argument(
        "--reference",
        default="Xe",
        help="species whose self-pair sets the reduction scale (P0.4)",
    )
    p.add_argument("--windows", type=int, default=500)
    p.add_argument("--window-cycles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    if (a.cif is None) == (a.bulk_box is None):
        p.error("exactly one of --cif or --bulk-box is required")

    guests = [g.strip() for g in a.guests.split(",") if g.strip()]
    if a.y is None:
        y = dict.fromkeys(guests, 1.0 / len(guests))
    else:
        fracs = [float(x) for x in a.y.split(",")]
        if len(fracs) != len(guests):
            p.error("--y must list one mole fraction per guest")
        if abs(sum(fracs) - 1.0) > 1e-9:
            p.error("--y must sum to 1")
        y = dict(zip(guests, fracs, strict=True))

    sysm = setup_system(
        a.forcefield,
        a.T,
        cif=a.cif,
        bulk_box=a.bulk_box,
        reference=a.reference,
    )
    ff, table, field = sysm.ff, sysm.table, sysm.field
    sr = sysm.sigma_ref
    gids = {g: table.index(g) for g in guests}
    box = sysm.box
    mass_amu = sysm.mass_amu
    n_cells = sysm.n_cells
    system = sysm.label

    beta = sysm.beta
    volume = sysm.volume
    per_pa = z_star_per_pa(sr, a.T)
    comp = ", ".join(f"y_{g}={y[g]:g}" for g in guests)

    print(f"Rung V3: {'/'.join(guests)} in {system} at {a.T:g} K, {comp}")
    unit = "mol/kg" if a.cif is not None else "<N>"
    cols = " ".join(f"{g + ' [' + unit + ']':>14}" for g in guests)
    sel_cols = " ".join(f"{'S(' + guests[0] + '/' + g + ')':>12}" for g in guests[1:])
    print(f"{'P [bar]':>9} {cols} {sel_cols}".rstrip())
    for P in [float(x) for x in a.pressures.split(",")]:
        pr = peng_robinson(ff, a.T, P, y)
        z_star = {gids[g]: pr.fugacity_pa[g] * per_pa for g in guests}
        r = run_point_ms(
            field,
            table,
            z_star,
            beta,
            box,
            volume,
            n_cells,
            ff.rc / sr,
            a.windows,
            a.window_cycles,
            a.seed,
        )
        n0 = r[gids[guests[0]]]["n_mean"]
        vals = []
        for g in guests:
            n = r[gids[g]]["n_mean"]
            vals.append(loading_mol_per_kg(n, mass_amu) if a.cif is not None else n)
        # selectivity S(g0/g) = (x_g0/x_g)/(y_g0/y_g), the convention the
        # anchor uses for S(Xe/Kr); n means suffice since the mass factors
        # cancel in the ratio
        sels = []
        for g in guests[1:]:
            n = r[gids[g]]["n_mean"]
            if n > 0 and y[guests[0]] > 0:
                sels.append((n0 / n) / (y[guests[0]] / y[g]))
            else:
                sels.append(float("nan"))
        row = " ".join(f"{v:>14.4f}" for v in vals)
        sel_row = " ".join(f"{s:>12.3f}" for s in sels)
        print(f"{P / 1e5:>9.3g} {row} {sel_row}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
