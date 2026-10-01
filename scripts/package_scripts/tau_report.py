"""Read finished run directories and report tau_N per wall-second per engine.

    python -m scripts.tau_report runs/tau_crossover --out tau.json

This is the reading half of `run-files/tau_crossover.json`, and it reads only
what a run wrote: the `N` and `wall_s` columns of `run.log`, the `glb_*` and
`glb_nhat_*` columns when the Glauber engine filled them, and `params.json` for the
state point. Nothing here re-runs physics, so a sweep that finished months ago
on a machine that no longer exists still analyses.

WHY IT EXISTS AS A SCRIPT rather than as a test or a library function.
`gcmc/analysis/autocorr.py` is the estimator and sits at L3 with the other
estimators; this is the command line that points it at a run directory, which
is the same split as `gcmc/analysis/demixing.py` and `scripts/demixing.py`. And
it is not a test because the answer is a measurement with an error bar and a
reliability flag rather than a pass or a fail -- `tau` at one state point is a
number to report, not a threshold to clear.

THE ONE RULE IT ENFORCES, because writing it down was not enough. A `tau`
estimated from a series shorter than about fifty `tau` is biased LOW, and the
bias flatters whichever engine is being measured -- measured on this very
comparison, where 40 windows made the Glauber engine look 1.5x faster than both
incumbents and 400 windows showed it at parity. `Tau.reliable` is that test, and
the ratio between two arms is REFUSED here when either arm fails it. An
unreliable row still prints its numbers, marked, because knowing a run was too
short is the useful output; what is withheld is the comparison that would be
read as a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcmc.analysis.autocorr import tau_int

#: Columns read by name, so a log whose optional groups differ still parses.
#: `N` and `wall_s` are in the `core` group and cannot be dropped, which is what
#: makes this script safe to point at any run directory.
_NEED = ("N", "wall_s")


def read_log(path: Path) -> dict[str, np.ndarray]:
    """The run log as named float columns, header taken from the `#` line."""
    with path.open() as fh:
        head = fh.readline()
    if not head.startswith("#"):
        raise ValueError(f"{path}: no '#' header line; not a run log")
    names = head.lstrip("#").split()
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        raise ValueError(f"{path}: no data rows")
    if rows.shape[1] != len(names):
        raise ValueError(
            f"{path}: header names {len(names)} columns but rows have "
            f"{rows.shape[1]}; the log was written by a different version"
        )
    return {n: rows[:, i] for i, n in enumerate(names)}


def analyse_run(d: Path, *, burn: float = 0.5) -> dict[str, Any]:
    """One run directory -> its state point, its tau and its Glauber diagnostics.

    `burn` is the leading FRACTION of windows discarded. Half is deliberate and
    is not a tuning knob: these runs start from a lattice at a guessed N and the
    approach to the stationary density is itself a long correlation, so a burn-in
    short enough to leave any of it in the series inflates `tau` without bound.
    """
    cols = read_log(d / "run.log")
    for k in _NEED:
        if k not in cols:
            raise ValueError(f"{d}: run.log has no '{k}' column")
    params = json.loads((d / "params.json").read_text())
    smp = params.get("sampler", {})
    ens = params.get("ensemble", {})

    n_all = cols["N"]
    keep = int(len(n_all) * burn)
    N = n_all[keep:]
    wall_total = float(cols["wall_s"].sum())
    wall_per_window = wall_total / len(n_all)
    t = tau_int(N)

    out: dict[str, Any] = {
        "dir": d.name,
        "engine": smp.get("engine", "metropolis_dynamics"),
        "sampler": smp.get("type"),
        "mu": ens.get("mu"),
        "box": ens.get("box"),
        "windows": int(len(n_all)),
        "windows_measured": int(len(N)),
        "N_mean": float(N.mean()),
        "N_var": float(N.var(ddof=1)) if len(N) > 1 else float("nan"),
        "tau_windows": t.tau,
        "reliable": bool(t.reliable),
        "wall_s_per_window": wall_per_window,
        # THE METRIC. Wall-seconds per independent value of N, which is what a
        # core-hour buys. Wall rather than CPU because that is what the log
        # carries; run the sweep at --threads 1 with one run per core and the
        # two agree to a few percent, and they do not agree at all if the node
        # is oversubscribed -- which the launcher's core budget is there to stop.
        "wall_s_per_independent_N": t.tau * wall_per_window,
        "n_eff": t.n_eff,
        "sem_N": t.sem,
    }

    # The Glauber engine's own diagnostics, when the group was written. A nonzero
    # truncation counter is a SCOPE statement and not an exactness failure -- the
    # kernel is still reversible on the region it moves -- but it means M was too
    # small for the occupancy at this state point, so it is reported rather than
    # summarised away.
    if "glb_step" in cols:
        steps = float(cols["glb_step"].sum())
        out["glauber"] = {
            "steps": steps,
            "identity_fraction": float(cols["glb_same"].sum()) / steps,
            "dN_per_step": float(cols["glb_dN"].sum()) / steps,
            "max_k": float(cols["glb_kmax"].max()),
            "cut_k": float(cols["glb_cut_k"].sum()),
            "cut_M": float(cols["glb_cut_M"].sum()),
            "infeasible": float(cols["glb_infeas"].sum()),
        }
        if "glb_nhat_sum" in cols and float(cols["glb_nhat_n"].sum()) > 0:
            # The Rao-Blackwellised estimator, chain-averaged over the kept
            # windows: the ladder's own mean of N rather than the sampled one.
            n = float(cols["glb_nhat_n"][keep:].sum())
            out["glauber"]["N_hat"] = float(cols["glb_nhat_sum"][keep:].sum()) / n
    return out


def collect(roots: list[Path], *, burn: float = 0.5) -> list[dict[str, Any]]:
    """Every run directory under the given paths, analysed. A path that IS a run
    directory is taken as one; otherwise its immediate children that hold a
    `run.log` are.

    A directory that cannot be read is REPORTED AND SKIPPED rather than taking
    the sweep with it: one killed run with a truncated last line is recoverable
    by resubmitting that run, and losing the analysis of the other thirteen is
    not.
    """
    dirs: list[Path] = []
    for r in roots:
        if (r / "run.log").is_file():
            dirs.append(r)
            continue
        dirs += sorted(c for c in r.iterdir() if (c / "run.log").is_file())
    rows = []
    for d in dirs:
        try:
            rows.append(analyse_run(d, burn=burn))
        except (OSError, ValueError) as exc:  # a killed run, a truncated log
            print(f"  SKIPPED {d.name}: {exc}")
    return rows


def _mu_key(row: dict[str, Any]) -> float:
    mu = row.get("mu")
    if isinstance(mu, dict):  # a typed state point; sort by the first species
        return float(next(iter(mu.values()), 0.0))
    return float(mu if mu is not None else 0.0)


def ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per state point, the engine ratio -- or a refusal naming what was short.

    The refusal is the point of this function. Two arms of one comparison are
    only comparable when both are resolved, and the number this would otherwise
    print is exactly the number that was wrong at 40 windows.
    """
    out: dict[str, Any] = {}
    keys = sorted({(_mu_key(r), r["sampler"]) for r in rows})
    for mu, sampler in keys:
        arms = {
            r["engine"]: r for r in rows if _mu_key(r) == mu and r["sampler"] == sampler
        }
        if len(arms) < 2:
            continue
        label = f"mu{mu:g}/{sampler}"
        short = [e for e, r in arms.items() if not r["reliable"]]
        if short:
            out[label] = {
                "refused": (
                    "tau is not resolved for "
                    + ", ".join(sorted(short))
                    + "; a tau from a series shorter than ~50 tau is biased low, "
                    "so this ratio would flatter one arm. Raise "
                    "run.target_cycles and re-run."
                ),
                "tau_windows": {e: r["tau_windows"] for e, r in arms.items()},
                "windows_measured": {e: r["windows_measured"] for e, r in arms.items()},
            }
            continue
        cost = {e: r["wall_s_per_independent_N"] for e, r in arms.items()}
        base = cost.get("metropolis_dynamics")
        out[label] = {
            "wall_s_per_independent_N": cost,
            "speedup_over_metropolis_dynamics": (
                {e: base / c for e, c in cost.items()} if base else None
            ),
            "tau_windows": {e: r["tau_windows"] for e, r in arms.items()},
            "wall_s_per_window": {e: r["wall_s_per_window"] for e, r in arms.items()},
        }
    return out


def report(rows: list[dict[str, Any]]) -> None:
    """One line per run, then the per-state-point comparison.

    THE SAMPLER IS IN THE TABLE, not only in the grouping. `engine` is
    `metropolis_dynamics` for the channel arm AND for the single-particle reference
    arm, so without its own column two rows of one mu read as a duplicate and
    the reference arm looks like a second measurement of the same thing. It is
    a third arm of the comparison and `ratios` keys on the sampler for the same
    reason.
    """
    rows = sorted(rows, key=lambda r: (_mu_key(r), str(r["sampler"]), r["engine"]))
    print(
        f"\n  {'sampler':16s} {'engine':20s} {'mu':>7s} {'<N>':>8s} {'Var N':>8s} "
        f"{'tau':>7s} {'rel':>5s} {'s/win':>7s} {'s/indep':>8s}  glauber"
    )
    for r in rows:
        w = r.get("glauber")
        note = ""
        if w:
            note = f"id {w['identity_fraction']:.2f}  |dN| {w['dN_per_step']:.2f}"
            if w["cut_M"] or w["cut_k"]:
                note += f"  TRUNCATED {w['cut_M']:.0f}/{w['cut_k']:.0f}"
        print(
            f"  {r['sampler']!s:16s} {r['engine']:20s} {_mu_key(r):7.2f} "
            f"{r['N_mean']:8.2f} {r['N_var']:8.2f} {r['tau_windows']:7.2f} "
            f"{'yes' if r['reliable'] else 'NO':>5s} "
            f"{r['wall_s_per_window']:7.3f} {r['wall_s_per_independent_N']:8.3f}  {note}"
        )
    rat = ratios(rows)
    if not rat:
        return
    print("\n  per state point:")
    for label, v in rat.items():
        if "refused" in v:
            print(f"    {label}: REFUSED -- {v['refused']}")
            continue
        sp = v["speedup_over_metropolis_dynamics"] or {}
        got = "  ".join(
            f"{e} {c:.3f} s" + (f" ({sp[e]:.2f}x)" if e in sp else "")
            for e, c in v["wall_s_per_independent_N"].items()
        )
        print(f"    {label}: {got}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="run dirs or their parent")
    ap.add_argument("--out", type=Path, help="write the full table as JSON")
    ap.add_argument(
        "--burn",
        type=float,
        default=0.5,
        help="leading fraction of windows discarded (default 0.5)",
    )
    a = ap.parse_args(argv)
    rows = collect(a.roots, burn=a.burn)
    if not rows:
        print("no run directories with a run.log found")
        return 1
    report(rows)
    if a.out:
        payload = {"runs": rows, "per_state_point": ratios(rows)}
        a.out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
        print(f"\n  wrote {a.out}")
    n_short = sum(1 for r in rows if not r["reliable"])
    if n_short:
        print(
            f"\n  {n_short} of {len(rows)} runs have an UNRESOLVED tau. Raise "
            "run.target_cycles; a tau needs a series of about 50 tau and these "
            f"have {min(r['windows_measured'] for r in rows)} windows measured."
        )
    trunc = [r for r in rows if (r.get("glauber") or {}).get("cut_M")]
    if trunc:
        print(
            f"  {len(trunc)} runs hit the pool ceiling. M was below the window "
            "occupancy there: raise sampler.trial_density, and note that the "
            "lambda <= 4 rule caps it on the LJ path."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
