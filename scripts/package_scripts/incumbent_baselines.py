"""G0 of the design plan: the baseline and the two incumbents.

WHY THIS EXISTS AS A SCRIPT AND NOT AS A TEST. Every number here is a
MEASUREMENT, not an assertion: tau_N per CPU-second, what GC-TMMC costs to get
an isotherm, what staged insertion buys at high density, what the transfer
matrix costs once its selection radius is forced to the full cutoff. A test
would have to pick a threshold for each, and there is no honest threshold before
the measurement exists -- that is the whole point of a baseline. The gates that
DO assert live in tests/; this produces the numbers they were chosen from, and
writes them to a JSON so a later run can be compared rather than re-argued.

    python -m scripts.incumbent_baselines --out g0.json            everything
    python -m scripts.incumbent_baselines --only tau dp            two sections

SIZED FOR ONE MACHINE, DELIBERATELY. Errington's T = 0.7 GC-TMMC run took 3.7
CPU hours on a 1 GHz Pentium III, so the incumbent is cheap; Siderius & Shen's
3e10 moves were to generate low-uncertainty REFERENCE DATA, not to get an
isotherm, and scoping against that number is how a baseline becomes a cluster
job. Every section below finishes in minutes and says what it measured, so the
comparison is reproducible on a laptop and the cluster is for production runs.

WHAT G0.1 MEANS BY "SURFACE IT". The add rejection has two causes with different
fixes -- no clique was found (a geometric failure, before the acceptance test)
and the acceptance test rejected -- and the counters already encode both:
`add_step` counts proposals, `add_attempt` those that reached the test,
`add_accept` those accepted. So the split is `add_step - add_attempt` against
`add_attempt - add_accept`, and nothing needed a new counter (which would have
added two permanently-zero integers to every summary in the project and moved
52 matrix digests). `split_add_rejection` below is that arithmetic, named.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np

from gcmc import config_specs
from gcmc.analysis.autocorr import tau_int
from gcmc.engine import metropolis_dynamics
from gcmc.engine.expanded_ensemble import ExpandedGCMC, fresh_state
from gcmc.run import sim_setup
from gcmc.sampling.flat_histogram import (
    FlatHistogram,
    ln_pi_eigenvector,
    ln_pi_multiband,
    ln_pi_three_band,
)
from gcmc.system.state import ChainState


def split_add_rejection(counters: dict[str, int]) -> dict[str, float]:
    """G0.1: the add rejection, split by CAUSE rather than reported as one rate.

    Two causes, two different fixes, and an aggregate acceptance ratio hides
    which one is biting:

      no_clique  the proposal never reached the acceptance test -- every trial
                 overlapped, or the transfer matrix found no clique of the
                 required size. The fix is a larger or denser trial cloud.
      rejected   the acceptance test ran and said no. The fix is a different
                 state point, a shorter move, or a better bias.

    Returns fractions of PROPOSALS, so the two plus the accepted fraction sum
    to one and each can be read against the others. An engine with no add move
    gets `{"n_add_steps": 0}` and no split at all.

    MEASURED at T* = 1.5, V = 512, 400 windows, `branch` sampler: mu = -4 gives
    no_clique 0.00 / rejected 0.09 / accepted 0.91, and mu = -2.5 gives
    0.54 / 0.12 / 0.34. The channel sampler at mu = -2.5 gives 0.70 / 0.08 /
    0.22. So at rho* = 0.59 between a half and seven tenths of all add proposals
    never reach an acceptance test at all, and a denser cloud -- not a better
    bias -- is what that asks for. This is the quantity the Glauber engine removes.
    """
    step = float(counters.get("add_step", 0))
    att = float(counters.get("add_attempt", 0))
    acc = float(counters.get("add_accept", 0))
    if step <= 0.0:
        # THE HEAT BATH TAKES THIS BRANCH, and it is not a degenerate case to
        # paper over with zeros: it has no add move, so the split does not
        # EXIST rather than being zero, and `accepted: 0.0` would read as "never
        # accepted" for an engine that never rejects anything. A NaN was worse
        # still -- it is not valid JSON and every row of the tau table carried
        # three of them.
        return {"n_add_steps": 0.0}
    return {
        "no_clique": (step - att) / step,
        "rejected": (att - acc) / step,
        "accepted": acc / step,
    }


def _lattice(n: int, box: float, dim: int) -> np.ndarray:
    g = int(math.ceil(n ** (1.0 / dim)))
    xs = (np.arange(g) + 0.5) * (box / g)
    return np.array(np.meshgrid(*[xs] * dim, indexing="ij")).reshape(dim, -1).T[:n]


def _run_chain(cfg: Any, n_windows: int, n0: int, seed: int) -> dict[str, Any]:
    """One chain; returns the N series, the counter totals and the CPU time."""
    pot = cfg.build_potential()
    wgh = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(cfg.dimension, seed=seed)
    st.positions = _lattice(n0, cfg.box, cfg.dimension)
    if cfg.engine == "glauber_dynamics":
        from gcmc.engine import glauber_dynamics

        adv = glauber_dynamics.advance
    else:
        adv = metropolis_dynamics.advance
    Ns: list[float] = []
    tot: dict[str, float] = {}
    t0 = time.process_time()
    for _ in range(n_windows):
        out = adv(st, cfg, pot, wgh)
        Ns.append(float(out["N"]))
        for k, v in out["counters"].items():
            tot[k] = tot.get(k, 0.0) + float(v)
    cpu = time.process_time() - t0
    return {"N": np.asarray(Ns), "counters": tot, "cpu": cpu}


def _params(
    *,
    mu: float,
    sampler: str,
    engine: str = "metropolis_dynamics",
    box: float = 8.0,
    T: float = 1.5,
    window: int = 200,
    seed: int = 1,
) -> dict[str, Any]:
    """The G0 systems: an LJ cavity sampler, an LJ channel sampler, the Glauber engine.

    `lj_hd_T1.5` is the plan's name for the cavity (branch) route at T* = 1.5 --
    the high-density route, which is why insertion difficulty is the whole
    question for it -- and `lj_channel_T1.5` is the oriented-cuboid
    transfer-matrix route at the same temperature. Same Hamiltonian, three
    kernels, which is what makes tau_N per CPU-second comparable between them.

    THE CHEMICAL POTENTIALS ARE NOT THE PLAN'S LITERAL -2 / 0 / +1, and the
    difference is a convention rather than a change of state point. This package
    carries the TOTAL chemical potential with Lambda = 1, so beta mu = ln rho +
    beta mu_ex; at T* = 1.5 that puts the three densities the plan is asking
    about -- dilute, mid-liquid, dense -- at mu = -4.0, -3.0 and -2.5, which
    measure rho* = 0.11, 0.51 and 0.59 in a box of 8. Taking -2 / 0 / +1
    literally in this convention would put the top two points beyond freezing,
    where tau_N measures how long a jammed configuration takes to unjam and not
    how well a sampler exchanges.

    Box 8 is V = 512 -- the volume the README's EOS row and the NIST SRSW tables
    both use -- and at L = 5, w = 1.5 the body diagonal is 5.431 against the LJ
    guard's limit of box - uc = 5.5, so this is the largest body the periodic
    guard allows at that volume.
    """
    if sampler == "channel":
        smp: dict[str, Any] = {
            "type": "channel",
            "depth": 1,
            "trial_density": 1.5,
            "channel_length": 5.0,
            "channel_width": 1.5,
        }
        if engine != "metropolis_dynamics":
            smp["engine"] = engine
    else:
        smp = {"type": sampler, "depth": 1, "trials": 30, "trial_box_side": 3.0}
    return {
        "potential": {"main": "lennard_jones", "T": T},
        "sampler": smp,
        "ensemble": {"dimension": 3, "box": box, "mu": mu},
        "run": {"seed": seed, "sampling_window": window},
    }


# --------------------------------------------------------------------- sections
def section_tau(
    n_windows: int = 80,
    seeds: tuple[int, ...] = (1, 2),
    mus: tuple[float, ...] = (-4.0, -3.0, -2.5),
) -> dict[str, Any]:
    """G0.2 and G0.3: tau_N per CPU-second at three densities for three kernels.

    THE METRIC FOR EVERY LATER PERFORMANCE CLAIM, and the only thing the two
    engines can be compared on: the Glauber engine has no acceptance test, so an
    acceptance ratio says nothing about it. Reported in CPU-SECONDS rather than
    in samples, because a sample is not a unit of cost and the three routes log
    N at the same cadence only by coincidence.

    The Glauber engine is measured here alongside the two incumbents rather
    than in a separate section, so the comparison is at the same state points,
    the same box and the same number of windows -- which is the only way a ratio
    of tau_N per CPU-second means anything.

    `n_windows` and `mus` are exposed because whether a tau may be quoted at all
    depends on them: --tau-windows 320 --tau-mu -2.5 is how the dense point was
    brought inside `Tau.reliable`, and the defaults below are what the whole
    table was measured at.

    EIGHTY WINDOWS, FORTY OF THEM MEASURED, because `Tau.reliable` is what
    decides whether a number may be quoted. tau_N here is of order half a window
    to one window, and a tau estimated from a series shorter than about 50 tau is
    biased LOW -- which flatters whichever engine is being measured. At 24
    windows every point came back `reliable = False`; the `reliable` flag on
    every row below is part of the result and not a diagnostic to ignore.
    """
    routes = (
        ("lj_hd_T1.5", "branch", "metropolis_dynamics"),
        ("lj_channel_T1.5", "channel", "metropolis_dynamics"),
        ("lj_channel_T1.5/glauber", "channel", "glauber_dynamics"),
    )
    out: dict[str, Any] = {}
    for name, sampler, engine in routes:
        for mu in mus:
            rows = []
            for seed in seeds:
                cfg = sim_setup.from_params(
                    _params(mu=mu, sampler=sampler, engine=engine, seed=seed)
                )
                n0 = {(-4.0): 60, (-3.0): 260, (-2.5): 300}[mu]
                r = _run_chain(cfg, n_windows, n0, seed)
                half = len(r["N"]) // 2
                t = tau_int(r["N"][half:])
                rows.append(
                    {
                        "seed": seed,
                        "N_mean": float(r["N"][half:].mean()),
                        "tau_samples": t.tau,
                        "reliable": t.reliable,
                        "cpu_s": r["cpu"],
                        "tau_cpu_s": t.tau * r["cpu"] / len(r["N"]),
                        "add_split": split_add_rejection(
                            {k: int(v) for k, v in r["counters"].items()}
                        ),
                    }
                )
            out[f"{name}/mu{mu:+.1f}"] = rows
    return out


def section_tmmc(
    mu: float = 0.0,
    box: float = 8.0,
    n_max: int = 240,
    windows: int = 40,
    refresh_every: int = 8,
) -> dict[str, Any]:
    """G0.4: GC-TMMC on the EXISTING single-particle Metropolis engine.

    Errington's settings, as far as they transfer: N_min = 0 (required for the
    pressure reference and the grand-free-energy relation), the biasing function
    refreshed periodically, and the collection matrix incremented with the
    UNBIASED acceptance of every proposal including the geometric failures.

    The one setting that does NOT transfer is his 30% displacement / 70%
    creation-destruction split: `moves.draw_move` is a frozen 50/25/25 and
    changing it would move every chain in the project. The consequence is a
    smaller exchange flux per proposal, which costs wall clock and nothing else.

    Returns ln Pi over the visited range, the flatness achieved, and the CPU
    time -- the three things the STOP CONDITION needs.
    """
    cfg = sim_setup.from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {"type": "metropolis"},
            "ensemble": {"dimension": 3, "box": box, "mu": mu},
            "run": {"seed": 5, "sampling_window": 4000},
        }
    )
    fh = FlatHistogram(n_min=0, n_max=n_max)
    model = metropolis_dynamics.MetropolisModel(
        beta=cfg.beta,
        box=np.asarray(cfg.box_dims, float),
        volume=cfg.total_volume,
        upper_cutoff=cfg.upper_cutoff,
        lower_cutoff=cfg.lower_cutoff,
        n_neigh=cfg.no_neigh,
        dim=cfg.dimension,
        move_factor=cfg.move_factor,
        log_z={0: cfg.beta * cfg.chem_pot},
        species_probs=(1.0,),
        species_ids=(0,),
        hard_core=True,
        bias=fh,
    )
    rng = np.random.default_rng(5)
    pos = np.empty((0, 3))
    from gcmc.system.state import zero_counters

    counters = zero_counters()
    t0 = time.process_time()
    for w in range(windows):
        pos, _spc, _n = metropolis_dynamics.metropolis_window(
            pos, None, rng, model, 4000, counters
        )
        if (w + 1) % refresh_every == 0:
            fh.refresh(damping=1.0)
    cpu = time.process_time() - t0
    lp = fh.ln_pi()
    seen = fh.visits > 0
    return {
        "cpu_s": cpu,
        "proposals": int(sum(counters[k] for k in ("add_step", "remove_step"))),
        "visited_range": [int(np.argmax(seen)), int(n_max - np.argmax(seen[::-1]))],
        "flatness": fh.flatness(),
        "three_band": fh.is_three_band(),
        "ln_pi": [float(v) for v in lp],
        "visits": [float(v) for v in fh.visits],
        # <N> from ln Pi, which is the whole point: an isotherm from ONE run
        "N_from_ln_pi": _mean_from_ln_pi(lp, fh.visits),
        "add_split": split_add_rejection(counters),
    }


def _mean_from_ln_pi(ln_pi: np.ndarray, visits: np.ndarray) -> float:
    """<N> from the macrostate distribution, over the visited range only."""
    seen = np.asarray(visits) > 0
    lp = np.asarray(ln_pi)[seen]
    ns = np.arange(len(visits))[seen]
    p = np.exp(lp - lp.max())
    return float((ns * p).sum() / p.sum())


def section_expanded(
    mu: float = 3.0, box: float = 8.0, pilot: int = 60_000, prod: int = 400_000
) -> dict[str, Any]:
    """G0.5: expanded-ensemble staged insertion, the second incumbent.

    Siderius & Shen's remedy for the same pathology: grow the LJ sigma stepwise
    through stages so a particle never has to appear at full size in a gap that
    cannot hold it. Not rejection-free, but cheap, general, needs no transfer
    matrix, works in 3D, and is already validated -- so it is the baseline a new
    kernel has to beat rather than an alternative to mention.

    A pilot run tunes the stage weights (the same -log(visits) rule the
    macrostate bias uses); production then measures <N> from the m = 0 slice,
    which is exactly muVT (see the module docstring of `engine.expanded_ensemble`).
    """
    rng = np.random.default_rng(7)
    eng = ExpandedGCMC(
        box=box,
        dim=3,
        beta=1.0,
        mu=mu,
        hard_core=True,
        n_stages=8,
        move_factor=0.25,
    )
    st = fresh_state(box, 3, np.empty((0, 3)), rng)
    eng.run(st, rng, pilot)
    weights = eng.tune_stage_weights().tolist()
    t0 = time.process_time()
    series = eng.run(st, rng, prod)
    cpu = time.process_time() - t0
    half = len(series) // 2
    t = tau_int(series[half:]) if len(series) > 4 else None
    return {
        "cpu_s": cpu,
        "stage_weights": weights,
        "n_stage0_samples": int(len(series)),
        "N_mean": float(series[half:].mean()) if len(series) else float("nan"),
        "tau_samples": None if t is None else t.tau,
        "tau_cpu_s": None if t is None else t.tau * cpu / max(1, len(series)),
        "counters": dict(eng.counters),
    }


def section_dp(reps: int = 12) -> dict[str, Any]:
    """G0.6: re-measure the transfer matrix's cost with rc = uc.

    THE OLD NUMBERS ARE NOT A VALID PLANNING BASIS for this engine. The
    `channel.py` docstring quotes 13-51 ms at n = 40-320, measured with the
    SELECTION radius shortened by a dual-cutoff auxiliary; the Glauber engine has no
    acceptance step to divide a shortened radius back out of, so it scores every
    pair at the full cutoff and the DP's eviction window is correspondingly
    wider.

    Measured at the pool sizes section 2B.6 quotes, with both a capped kmax and
    kmax = M, so the cost of capping is visible rather than asserted.
    """
    from gcmc.sampling.channel import ChannelSweepHardCore, ChannelSweepLJ

    out: dict[str, Any] = {}
    for M in (64, 96, 160):
        for kmax in (6, 12, M):
            for lj, uc in (("hard_core", 1.0), ("lj_rc2.5", 2.5)):
                # the body length follows M at 4 trials per sigma, the section
                # 3.5 scoping limit, so lambda is held fixed across pool sizes
                L = M / 4.0
                w = 1.5
                ts = []
                for r in range(reps):
                    rng = np.random.default_rng(1000 + r)
                    U = np.column_stack(
                        [
                            rng.uniform(-L / 2, L / 2, M),
                            rng.uniform(-w / 2, w / 2, M),
                            rng.uniform(-w / 2, w / 2, M),
                        ]
                    )
                    mw = np.exp(rng.normal(0.0, 1.0, M))
                    t0 = time.process_time()
                    if lj == "hard_core":
                        sw: Any = ChannelSweepHardCore(U, 1.0, kmax)
                    else:
                        sw = ChannelSweepLJ(U, mw, 0.9, uc, 1.0 / 1.5, kmax)
                    sw.sample(min(2, kmax), rng)
                    ts.append(time.process_time() - t0)
                out[f"M{M}/kmax{kmax}/{lj}"] = {
                    "median_ms": float(1e3 * np.median(ts)),
                    "states": int(getattr(sw, "states", -1)),
                }
    return out


def section_multiband(n: int = 30, dmax: int = 4) -> dict[str, Any]:
    """G0.7: the multi-band ln Pi solve, as a feasibility spike.

    Errington's eq. 11 is a sequential recursion that requires a TRI-BANDED
    collection matrix -- "restricting transitions to neighbouring macrostates
    limits the macrostate transition probability matrix to three bands, which
    enables one to obtain a unique solution"; without it "this is an
    overspecified problem and a minimization technique needs to be
    implemented." The Glauber engine changes N by up to M in one step, so the
    recursion does not apply to it at all and G9 is unreachable without a
    multi-band solve.

    The spike: build a reversible kernel with a known ln Pi and bands out to
    dmax, then solve it two structurally different ways -- weighted least
    squares on the log-ratios, and the stationary left eigenvector of the
    row-normalised matrix. Agreement between those two is evidence about the
    SOLVE; agreement with the planted answer is evidence it is the right one.
    """
    true = -0.5 * ((np.arange(n) - 0.4 * n) / (0.15 * n)) ** 2
    P = np.exp(true - true.max())
    offs = [d for d in range(-dmax, dmax + 1) if d != 0]
    q = 1.0 / len(offs)
    K = np.zeros((n, n))
    for i in range(n):
        for d in offs:
            j = i + d
            if 0 <= j < n:
                K[i, j] = q * min(1.0, P[j] / P[i])
        K[i, i] = 1.0 - K[i].sum()
    C = P[:, None] * K * 1e7
    lsq, resid = ln_pi_multiband(C)
    eig = ln_pi_eigenvector(C)
    three = None
    try:
        ln_pi_three_band(C)
    except ValueError as e:
        three = str(e)[:120]
    ref = true - true[0]
    return {
        "bands": dmax,
        "lsq_max_err": float(np.abs(lsq - ref).max()),
        "lsq_weighted_rms_residual": resid,
        "eig_max_err": float(np.abs(eig - ref).max()),
        "lsq_vs_eig": float(np.abs(lsq - eig).max()),
        "three_band_refused": three,
        "verdict": (
            "feasible"
            if max(float(np.abs(lsq - ref).max()), float(np.abs(eig - ref).max()))
            < 1e-8
            else "INFEASIBLE"
        ),
    }


def section_srsw() -> dict[str, Any]:
    """G0.8: the NIST SRSW specifics, recorded rather than remembered.

    Consequences to write into the run files, and the one that constrains a
    gate: with uc = 3 sigma and L = 8 sigma the section 3.5 LJ guard caps the
    channel at 8 - 3 = 5 sigma of body DIAGONAL, i.e. about 4.7 sigma of length
    at w = 1.5. That is the same length G6 wants anyway (Siderius & Shen used
    axial lengths of 6 and 12 sigma, and their 3.935 nm MWCNT is 2.043 nm long,
    about 6 sigma_Ar), so it is the point rather than a limitation.
    """
    L = 8.0
    uc = 3.0
    w = 1.5
    max_diag = L - uc
    max_len = math.sqrt(max(0.0, max_diag**2 - 2.0 * w**2))
    return {
        "volume_sigma3": 512.0,
        "box_sigma": L,
        "truncation_sigma": uc,
        "long_range_correction": True,
        "T_star_grid": "0.70 to 1.50 in 0.05 steps",
        "raw_macrostate_distributions_published": True,
        "T_star_used_by_G4_5": [1.20, 1.50],
        "tail_term_mandatory": True,
        "max_body_diagonal_sigma": max_diag,
        "max_channel_length_at_w1.5_sigma": max_len,
    }


def section_guard(sink: str | None = None) -> dict[str, Any]:
    """G0.9: the guard scan verdict, at `rc_sel` and advisory at `uc`.

    TWO SOURCES, because a params-file scan alone would miss most of the channel
    configurations in the tree: there are three params files with a channel
    sampler and about three dozen channel Configs built programmatically inside
    the test suite, and none of the latter is named by any file. So the scan
    reads both -- the files statically, and the JSONL sink that
    `guard_audit.record` fills when `GCMC_GUARD_AUDIT` is set during a test run:

        GCMC_GUARD_AUDIT=/tmp/guard.jsonl pytest -q -m ""
        python -m scripts.incumbent_baselines --only guard   # reads it if present

    A maintained list of test modules would have been the alternative, and this
    project has already been bitten by one.
    """
    import os
    from pathlib import Path

    from gcmc.run import guard_audit

    root = Path(__file__).resolve().parents[1]
    recs = guard_audit.scan_params_files(
        [root / "run-files", root / "examples", root / "tests"]
    )
    path = sink or os.environ.get(guard_audit.SINK_ENV)
    from_sink = guard_audit.load_sink(Path(path)) if path else []
    summary = guard_audit.summarise(recs + from_sink)
    summary["from_params_files"] = len(recs)
    summary["from_test_sink"] = len(from_sink)
    summary["sink"] = path
    return summary


SECTIONS = {
    "tau": section_tau,
    "tmmc": section_tmmc,
    "expanded": section_expanded,
    "dp": section_dp,
    "multiband": section_multiband,
    "srsw": section_srsw,
    "guard": section_guard,
}


def main() -> None:
    """Run the requested sections and write one JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="g0_baseline.json")
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS))
    # `tau` alone gets knobs, because it is the section whose answer depends on
    # how long it ran: `Tau.reliable` needs a series of about 50 tau, and at the
    # dense point tau is three windows. The defaults are what the report quotes.
    ap.add_argument("--tau-windows", type=int, default=80)
    ap.add_argument("--tau-mu", type=float, nargs="*")
    a = ap.parse_args()
    want = a.only or list(SECTIONS)
    out: dict[str, Any] = {}
    for name in want:
        t0 = time.time()
        print(f"== {name} ...", flush=True)
        kw: dict[str, Any] = {}
        if name == "tau":
            kw["n_windows"] = a.tau_windows
            if a.tau_mu:
                kw["mus"] = tuple(a.tau_mu)
        out[name] = SECTIONS[name](**kw)
        print(f"== {name} done in {time.time() - t0:.1f}s", flush=True)
        print(json.dumps(out[name], indent=1, default=str)[:2500], flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
