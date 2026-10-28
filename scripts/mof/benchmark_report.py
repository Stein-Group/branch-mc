"""C8: integrated autocorrelation time per unit of language-neutral cost.

BRANCH_MC is Python and RASPA2 is C, so CPU seconds are never compared across
the two codes. The cost unit is the PARTICLE ENERGY EVALUATION (PEE): the energy
of one particle (or one trial position) against everything within rc, framework
and guests. Each move kind costs a fixed number of PEE, set by the algorithm:

  translation                 2   old and new position
  conventional insert/delete  1   the new (insert) or the old (delete) particle
  CBMC insert/delete, k       k   k first-bead trials; deletion retraces k - 1
                                  extra trials plus the old position
  identity swap               2   the particle as its old and its new species
  CFCMC lambda move           2   the fractional molecule at old and new lambda
                                  (+1 on an insertion/removal event: the new
                                  fractional molecule's placement)
  ladder window move          M one-body rows (branch or channel body)
                              + intra-pool pairs / P,
                              P = pair evaluations per PEE (mean neighbours
                              within rc), so a pair costs what it costs

Where the counts come from:
  BRANCH_MC  run.log streams per-window attempts (*_tried) and the ladder's
             summed pool size (ladder_sum_pool); the intra-pool pairs are
             ladder_sum_pool * (M - 1) / 2 with M = state_point.json's pool_M,
             so every row's cost is exact.
  RASPA2     completed legs print exact totals per move (parse_raspa2 --moves);
             for the stream, attempts per printed sample are the EXPECTED count
             from simulation.input's move weights and RASPA2's own cycle length
             max(20, N) * n_components, N counting fractional molecules. The
             CFCMC insertion/removal events (+1 PEE each) are not in that
             expectation; they are a few % of lambda moves and are read off the
             completed-leg statistics when the exact figure is needed.

Same estimator for both codes: `gcmc.analysis.autocorr.tau_int` (Sokal window)
on the same observable, with its `reliable` flag (n >= 50 tau). A ratio between
two arms is printed only when both are reliable; an unreliable tau is biased low
and would flatter whichever arm is short.

THE CONTROL. Conventional BRANCH_MC and conventional RASPA2 (k = 1) are the same
algorithm. Their tau per PEE must agree within error at a shared state point, or
the cost model is wrong and no Fig 2 ratio means anything. `--control` names the
pair and the report prints that comparison first.

    python -m scripts.mof.benchmark_report \\
        --branch conv=runs/mof/mof_..._abc --branch branch=runs/mof/mof_..._def \\
        --raspa cfcmc=RUN/run.log:RUN/raspa2/simulation.input:RUN/moves.json \\
        --observable x_Xe --pairs-per-pee 340 --control conv,raspa_conv --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gcmc.analysis.autocorr import tau_int
from gcmc.run.io import LOG_CYCLE, read_log

PEE_TRANSLATION = 2.0
PEE_SWAP_IDENTITY = 2.0
PEE_CF_LAMBDA = 2.0
PEE_CF_EVENT = 1.0


@dataclass
class ArmResult:
    label: str
    code: str
    observable: str
    n: int
    mean: float
    sem: float
    tau_samples: float
    reliable: bool
    pee_per_sample: float
    tau_pee: float  # PEE per statistically independent sample (tau * cost/sample)
    note: str = ""


Columns = dict[str, np.ndarray]


def _n_total(cols: Columns) -> np.ndarray:
    return np.sum([v for k, v in cols.items() if k.startswith("N_")], axis=0)


def observable_series(cols: Columns, name: str) -> np.ndarray:
    """N_total (the sum of the N_<species> columns), N_<species>, or
    x_<species> = N_s / N_total (rows with N = 0 dropped, since a composition is
    undefined there)."""
    total = _n_total(cols)
    if name == "N_total":
        return np.asarray(total, float)
    if name.startswith("x_"):
        keep = total > 0
        return np.asarray(cols["N_" + name[2:]][keep] / total[keep], float)
    return np.asarray(cols[name], float)


def _drop_burn_in(cols: Columns, burn: float) -> Columns:
    cut = int(burn * len(cols[LOG_CYCLE]))
    return {k: v[cut:] for k, v in cols.items()}


def branch_pee_per_row(
    cols: Columns, pool_m: int | None, pairs_per_pee: float | None
) -> np.ndarray:
    """Exact cost of each run.log row."""
    zero = np.zeros(len(cols[LOG_CYCLE]))
    c = PEE_TRANSLATION * cols.get("translate_tried", zero)
    c = c + cols.get("insert_tried", zero) + cols.get("delete_tried", zero)
    c = c + PEE_SWAP_IDENTITY * cols.get("swap_tried", zero)
    if "ladder_sum_pool" in cols:
        pool = cols["ladder_sum_pool"]
        c = c + pool
        if pool_m is None:
            raise ValueError("a window-exchange run needs its pool size M")
        if pool_m > 1:
            if pairs_per_pee is None:
                raise ValueError(
                    "a window-exchange run needs --pairs-per-pee to convert its "
                    "intra-pool pair evaluations into PEE"
                )
            c = c + pool * (pool_m - 1) / 2.0 / pairs_per_pee
    return np.asarray(c, float)


def pool_size(run_dir: Path) -> int | None:
    """The ladder's pool size M from the run's state_point.json (None: no ladder)."""
    sp = Path(run_dir) / "state_point.json"
    if not sp.exists():
        return None
    ladder = json.loads(sp.read_text()).get("ladder") or {}
    return ladder.get("pool_M")


def analyse_branch(
    label: str, run_dir: Path, observable: str, burn: float, pairs_per_pee: float | None
) -> ArmResult:
    run_dir = Path(run_dir)
    cols = _drop_burn_in(read_log(run_dir / "run.log"), burn)
    x = observable_series(cols, observable)
    pee = branch_pee_per_row(cols, pool_size(run_dir), pairs_per_pee)
    t = tau_int(x)
    per = float(pee.sum() / max(len(x), 1))
    return ArmResult(
        label,
        "branch_mc",
        observable,
        t.n,
        t.mean,
        t.sem,
        t.tau,
        t.reliable,
        per,
        t.tau * per,
    )


def parse_raspa2_moves(sim_input: Path) -> dict[str, Any]:
    """Per-component move weights and the swap trial count from simulation.input."""
    comps: list[dict[str, float]] = []
    k_swap = 10.0  # RASPA2's default (input.c)
    for raw in Path(sim_input).read_text().splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "Component" and len(tok) > 2 and tok[2] == "MoleculeName":
            comps.append({})
        elif tok[0] == "NumberOfTrialPositionsForTheFirstBeadSwap":
            k_swap = float(tok[1])
        elif comps and tok[0] in (
            "TranslationProbability",
            "SwapProbability",
            "CFSwapLambdaProbability",
            "IdentityChangeProbability",
        ):
            comps[-1][tok[0]] = float(tok[1])
    if not comps:
        raise ValueError(f"{sim_input}: no components")
    return {"components": comps, "k_swap": k_swap}


def raspa2_expected_pee_per_step(
    moves: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Expected PEE of one RASPA2 MC step (component uniform, then a move by that
    component's normalised weights), and the expected attempts per step by kind."""
    k = moves["k_swap"]
    comps = moves["components"]
    pee = 0.0
    per_kind: dict[str, float] = {}
    for c in comps:
        tot = sum(c.values())
        if tot <= 0:
            continue
        for key, w in c.items():
            f = (w / tot) / len(comps)
            per_kind[key] = per_kind.get(key, 0.0) + f
            if key == "TranslationProbability":
                pee += f * PEE_TRANSLATION
            elif key == "SwapProbability":
                pee += f * k  # insert k trials, delete k (old + k - 1)
            elif key == "CFSwapLambdaProbability":
                pee += f * PEE_CF_LAMBDA
            elif key == "IdentityChangeProbability":
                pee += f * PEE_SWAP_IDENTITY
    return pee, per_kind


def analyse_raspa2(
    label: str,
    parsed_log: Path,
    sim_input: Path,
    observable: str,
    burn: float,
    moves_json: Path | None = None,
) -> ArmResult:
    cols = _drop_burn_in(read_log(parsed_log), burn)
    x = observable_series(cols, observable)
    moves = parse_raspa2_moves(sim_input)
    pee_step, _ = raspa2_expected_pee_per_step(moves)
    ncomp = len(moves["components"])
    cyc = np.round(cols[LOG_CYCLE] * 1e6)
    if len(cyc) < 2:
        raise ValueError(f"{parsed_log}: fewer than two samples")
    every = float(np.median(np.diff(cyc)))
    # steps per cycle: max(20, N_adsorbates) * n_components, N including the
    # fractional molecules (monte_carlo.c: NumberOfAdsorbateMolecules)
    nfrac = sum(
        (v for k, v in cols.items() if k.startswith("frac_")), np.zeros(len(cyc))
    )
    nads = _n_total(cols) + nfrac
    steps = np.maximum(20.0, nads) * ncomp * every
    pee = steps * pee_step
    note = ""
    if moves_json is not None and Path(moves_json).exists():
        m = json.loads(Path(moves_json).read_text())
        note = f"completed legs: {m.get('completed_legs', 0)}"
    t = tau_int(x)
    per = float(pee.sum() / max(len(x), 1))
    return ArmResult(
        label,
        "raspa2",
        observable,
        t.n,
        t.mean,
        t.sem,
        t.tau,
        t.reliable,
        per,
        t.tau * per,
        note,
    )


def compare(a: ArmResult, b: ArmResult) -> dict[str, Any]:
    """Cost ratio b/a of one independent sample; refused unless both reliable."""
    out: dict[str, Any] = {"a": a.label, "b": b.label}
    if not (a.reliable and b.reliable):
        out["ratio"] = None
        out["why"] = "at least one tau is unreliable (series < 50 tau): biased low"
        return out
    out["ratio"] = b.tau_pee / a.tau_pee
    # equilibrium check alongside: the two arms must sample the same mean
    z = (a.mean - b.mean) / math.hypot(a.sem, b.sem) if (a.sem or b.sem) else 0.0
    out["mean_z"] = z
    return out


def mean_neighbours(
    cif: Path,
    forcefield: Path,
    replicate: tuple[int, int, int] | None = None,
    guests_per_angstrom3: float = 0.0,
) -> float:
    """Pair evaluations per PEE: framework atoms (plus guests) inside a sphere of
    radius rc around a uniformly placed point, on average. Its only use is to
    price the window exchange's intra-pool pairs in PEE units."""
    from gcmc.forcefield.species_file import load_forcefield
    from gcmc.framework.structure import build

    ff = load_forcefield(forcefield)
    atoms, meta = build(Path(cif), ff, replicate=replicate)
    vol = float(np.prod(meta["supercell_angstrom"]))
    sphere = 4.0 / 3.0 * math.pi * ff.rc**3
    return (len(atoms) / vol + guests_per_angstrom3) * sphere


def _pairs(text: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for item in text or []:
        lab, _, rest = item.partition("=")
        if not rest:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        out.append((lab, rest))
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--branch", action="append", help="LABEL=run_dir (run.log)")
    p.add_argument(
        "--raspa",
        action="append",
        help="LABEL=run.log:raspa2/simulation.input[:moves.json]",
    )
    p.add_argument("--observable", default="N_total", help="N_total, N_Xe, x_Xe, ...")
    p.add_argument("--burn", type=float, default=0.2, help="leading fraction dropped")
    p.add_argument("--pairs-per-pee", type=float, default=None)
    p.add_argument("--control", default=None, help="LABEL_A,LABEL_B: same algorithm")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    arms: dict[str, ArmResult] = {}
    for lab, path in _pairs(a.branch):
        arms[lab] = analyse_branch(
            lab, Path(path), a.observable, a.burn, a.pairs_per_pee
        )
    for lab, spec in _pairs(a.raspa):
        parts = spec.split(":")
        mj = Path(parts[2]) if len(parts) > 2 else None
        arms[lab] = analyse_raspa2(
            lab, Path(parts[0]), Path(parts[1]), a.observable, a.burn, mj
        )

    print(f"observable {a.observable}; cost unit = particle energy evaluations (PEE)")
    print(
        f"{'arm':<14}{'code':<11}{'mean':>10}{'sem':>9}{'tau[smp]':>10}{'PEE/smp':>11}{'tau[PEE]':>12}  ok"
    )
    for r in arms.values():
        print(
            f"{r.label:<14}{r.code:<11}{r.mean:>10.4g}{r.sem:>9.2g}{r.tau_samples:>10.3g}"
            f"{r.pee_per_sample:>11.4g}{r.tau_pee:>12.4g}  {'yes' if r.reliable else 'NO'}"
        )
    report: dict[str, Any] = {"arms": {k: asdict(v) for k, v in arms.items()}}
    if a.control:
        ca, cb = a.control.split(",")
        c = compare(arms[ca], arms[cb])
        report["control"] = c
        print(f"\nCONTROL {ca} vs {cb} (same algorithm, must be ~1): {c}")
    labels = list(arms)
    report["ratios"] = [compare(arms[labels[0]], arms[b]) for b in labels[1:]]
    for c in report["ratios"]:
        print(f"cost of one independent sample, {c['b']} / {c['a']}: {c.get('ratio')}")
    if a.json is not None:
        a.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
