"""Command-line front end for run_mof: one record-run leg, one process (R2.5).

`run_mof` is a library taking a MofRunSpec, which is the right shape for tests but
cannot be driven from a job script. This is the missing CLI. It does no physics of
its own -- it builds a MofRunSpec and calls run_mof -- so the chain, the identity
hash and the checkpoint format are exactly those of the library path.

TWO WAYS TO NAME A STATE POINT, and the distinction is the plan's, not a
convenience:

  --fugacity "Xe=19888.52,Kr=79799.73"   RECORD runs. Fugacity in Pa is the state
                                         variable of record (R2.1/R2.2): both codes
                                         are handed the SAME f_s, no EOS in the
                                         loop, nothing to drift.
  --pressure 1e5 --y "Xe=0.2,Kr=0.8"     EXPLORATORY runs. Peng-Robinson converts
                                         P,y -> f_s here; phi is printed and
                                         archived so the number can be reproduced.

Legs: `--windows` is a TOTAL target, not an increment. Resubmitting the same
command with a larger --windows continues the same directory bit-identically
(RNG state round-trips through the checkpoint); killing it by walltime anywhere is
safe because every window is flushed to windows.csv as it completes.

The auxiliary: `--sweep-rc-aux` is a PROPOSAL radius in reduced units (sigma_ref),
never a physics cutoff. Omit it for the correctness rungs -- without it the repack
move is a heat bath -- and pass `--sweep-rc-aux $(rc*)` as the exact-no-op control
arm when benchmarking. It is part of the run identity, so an aux-on leg can never
resume an aux-off directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.mof.run_mof import MofRunSpec, mof_run_id, run_mof
from scripts.mof.system_setup import setup_system
from scripts.mof.units import load_forcefield, peng_robinson
from scripts.mof.v1v2_units import z_star_per_pa


def _kv(text: str, what: str) -> dict[str, float]:
    """Parse "Xe=1.0,Kr=2.0" into {"Xe": 1.0, "Kr": 2.0}."""
    out: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--{what} wants NAME=VALUE pairs, got {item!r}")
        name, _, value = item.partition("=")
        out[name.strip()] = float(value)
    if not out:
        raise SystemExit(f"--{what} parsed to nothing")
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, default=None, help="framework structure")
    p.add_argument(
        "--bulk-box", type=float, default=None, help="cubic box side in A, no framework"
    )
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument(
        "--reference", default="Xe", help="species setting the reduction scale"
    )
    # --- state point: exactly one of these two routes ---
    p.add_argument("--fugacity", default=None, help='Pa per species, "Xe=1e4,Kr=8e4"')
    p.add_argument("--pressure", type=float, default=None, help="Pa, with --y")
    p.add_argument("--y", default=None, help='gas mole fractions, "Xe=0.2,Kr=0.8"')
    # --- chain ---
    p.add_argument(
        "--windows", type=int, required=True, help="TOTAL target, not a delta"
    )
    p.add_argument("--window-cycles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--move-factor", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=50)
    # --- sweep move (A5.6) ---
    p.add_argument(
        "--sweep-moves-per-window",
        type=int,
        default=0,
        help="0 = conventional sampling only (the V2/S3 arm)",
    )
    p.add_argument("--sweep-L-long", type=float, default=None, help="reduced units")
    p.add_argument("--sweep-w", type=float, default=0.9, help="reduced units")
    p.add_argument("--sweep-n-trials", type=int, default=12)
    p.add_argument(
        "--sweep-rc-aux",
        type=float,
        default=None,
        help="short-range PROPOSAL radius in reduced units; omit for no auxiliary",
    )
    p.add_argument("--root", type=Path, default=Path("runs/mof"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the state point, print the plan and the run id, run nothing",
    )
    a = p.parse_args(argv)

    if (a.cif is None) == (a.bulk_box is None):
        p.error("exactly one of --cif or --bulk-box is required")
    if (a.fugacity is None) == (a.pressure is None):
        p.error("exactly one of --fugacity (record) or --pressure/--y (exploratory)")
    if a.pressure is not None and a.y is None:
        p.error("--pressure needs --y")

    sysm = setup_system(
        a.forcefield, a.T, cif=a.cif, bulk_box=a.bulk_box, reference=a.reference
    )
    per_pa = z_star_per_pa(sysm.sigma_ref, a.T)

    if a.fugacity is not None:
        f_pa = _kv(a.fugacity, "fugacity")
        phi: dict[str, float] | None = None
    else:
        y = _kv(a.y, "y")
        if abs(sum(y.values()) - 1.0) > 1e-9:
            p.error("--y must sum to 1")
        pr = peng_robinson(load_forcefield(a.forcefield), a.T, a.pressure, y)
        f_pa = dict(pr.fugacity_pa)
        phi = dict(pr.phi)

    z_star = {g: f * per_pa for g, f in f_pa.items()}

    # The guard is a property of the cell, not of the run, so resolve the rod length
    # against it here rather than discovering at runtime that every sweep silently
    # declined. sqrt(L^2 + 2 w^2) <= min(box) - rc, with a little headroom.
    budget = float(min(sysm.box)) - sysm.field.rc
    L_max = (
        (budget**2 - 2.0 * a.sweep_w**2) ** 0.5 if budget**2 > 2 * a.sweep_w**2 else 0.0
    )
    L_long = a.sweep_L_long if a.sweep_L_long is not None else 0.95 * L_max
    if a.sweep_moves_per_window:
        if L_max <= 0.0:
            p.error(
                f"no admissible rod in this cell: min(box) - rc = {budget:.3f} "
                f"reduced, which cannot hold a rod of width {a.sweep_w}. Widen the "
                f"supercell along the channel axis or narrow --sweep-w."
            )
        if L_long > L_max:
            p.error(
                f"--sweep-L-long {L_long:g} fails the periodic guard in this cell "
                f"(max {L_max:.3f} reduced at w={a.sweep_w:g}, i.e. "
                f"{L_max * sysm.sigma_ref:.2f} A). The move would decline EVERY "
                f"attempt and the sweep arm would silently equal the conventional "
                f"one. Lower it, narrow --sweep-w, or lengthen the supercell."
            )
        if a.sweep_rc_aux is not None and a.sweep_rc_aux < 1.4:
            print(
                f"WARNING: --sweep-rc-aux {a.sweep_rc_aux:g} is below 1.4 sigma. The "
                f"LJ first shell runs to ~1.5 sigma, so a shorter proposal cannot see "
                f"the structure it proposes into: sd(Delta_corr) explodes and "
                f"acceptance collapses. The failure is SILENT (the run completes at a "
                f"plausible density) -- watch aux_corr_sd in windows.csv.",
                file=sys.stderr,
            )

    spec = MofRunSpec(
        forcefield=a.forcefield,
        T=a.T,
        z_star_by_name=z_star,
        seed=a.seed,
        windows=a.windows,
        window_cycles=a.window_cycles,
        cif=a.cif,
        bulk_box=a.bulk_box,
        reference=a.reference,
        move_factor=a.move_factor,
        checkpoint_every=a.checkpoint_every,
        sweep_moves_per_window=a.sweep_moves_per_window,
        sweep_L_long=L_long,
        sweep_w=a.sweep_w,
        sweep_n_trials=a.sweep_n_trials,
        sweep_rc_aux=a.sweep_rc_aux,
        root=a.root,
    )

    plan = {
        "run_id": mof_run_id(spec),
        "system": sysm.label,
        "box_reduced": [float(x) for x in sysm.box],
        "sigma_ref_angstrom": sysm.sigma_ref,
        "rc_star": float(sysm.field.rc),
        "T": a.T,
        "fugacity_pa": f_pa,
        "fugacity_coefficient": phi,
        "z_star": z_star,
        "windows_target": a.windows,
        "window_cycles": a.window_cycles,
        "seed": a.seed,
        "sweep": {
            "moves_per_window": a.sweep_moves_per_window,
            "L_long_reduced": L_long,
            "L_long_angstrom": L_long * sysm.sigma_ref,
            "L_max_reduced": L_max,
            "w_reduced": a.sweep_w,
            "n_trials": a.sweep_n_trials,
            "rc_aux_reduced": a.sweep_rc_aux,
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if a.dry_run:
        return 0

    result = run_mof(spec)
    counts = {result.system.table.types[t]: n for t, n in result.n_by_type.items()}
    print(
        f"leg done: {result.paths.directory} resumed_from={result.resumed_from} "
        f"windows_done={result.windows_done} N={counts}"
    )
    # provenance beside the outputs, per R1.6/R2.5: the plan that produced this leg
    (result.paths.directory / "state_point.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
