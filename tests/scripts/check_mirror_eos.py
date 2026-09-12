"""Depth >= 1 proposal validation: the mirror geometry against an exact density.

Why this exists
---------------
The depth >= 1 proposal used to draw the cavity centre uniformly for an `add`
but pin it exactly ON an existing particle for a `remove`. Those two measures
are mutually singular, so no reverse path could ever reproduce a forward cavity
and detailed balance failed. Both directions now use the depth-0 mirror form
(anchor + uniform offset over the cavity body, anchor forced into the removal
set), for which the engine's existing acceptance algebra is exactly right:

    centre + removal set   (m+1) V / ((N+1) v)      [C(n_cav, m) cancels only
                                                     because n'_cav = n_cav + 1
                                                     and the anchor is removed]
    trial cloud            v / (n_trials - n_remove)
    selection              Z_{m+1} / Z_m            [requires a UNIFORM draw]
    ------------------------------------------------------------------
    product   e^{beta mu} (V/(N+1)) (Z_{m+1}/Z_m) (n_add/(n_trials - n_remove))

Stage-A statistical parity cannot test this: stage A IS the old engine, and the
fix deliberately changes n_cav (a particle-pinned centre guaranteed an occupant,
so occupancy was biased high). So this compares against physics instead.

Reference
---------
metropolis in the same box, which is exact and independent of the cavity
machinery. Cross-checked separately against a closed-form grand-canonical sum in
a 3 x 3 periodic box, which saturates at N = 6 so the partition sum terminates:
<N> = 3.5037, eta = 0.30583.

Measured with the fix in place (2D, box 10, mu = 1, 2.2e5 cycles):
    metropolis  0.30906 +- 0.00111   (reference)
    branch d1   0.30733 +- 0.00066   L = 2.0, 30 trials
    rosenbl d1  0.30709 +- 0.00092   L = 2.0, 30 trials
    branch d2   0.30928 +- 0.00066   L = 3.0, 40 trials   <- the k = 3 regime
Before the fix, branch d1 read 0.31595 +- 0.00075, i.e. +0.0069 (5 sigma).

Usage:
    python tests/scripts/check_mirror_eos.py           # quick, ~3 min
    python tests/scripts/check_mirror_eos.py full      # ~12 min, tighter bars
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gcmc import config_specs
from gcmc.engine import metropolis_dynamics as engine
from gcmc.run.sim_setup import from_params
from gcmc.system.state import ChainState

BOX = 10.0
MU = 1.0
SIGMA = 1.0
BURN_FRAC = 0.1
N_BLOCK = 16


def run(kind, *, depth=0, trials=30, side=2.0, nwin=600, win=200, seed=1):
    """Equilibrium packing fraction from an EMPTY start.

    Empty, not random: a random start overlaps, a hard-core chain can never
    separate an overlap, and the run then reports an inflated density forever.
    geometry.assert_no_overlap guards the production paths against exactly this;
    here we simply start from nothing.
    """
    sampler = {"type": kind}
    if kind != "metropolis":
        sampler.update({"trials": trials, "trial_box_side": side, "depth": depth})
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": MU, "dimension": 2, "box": BOX},
            "sampler": sampler,
            "run": {"sampling_window": win, "seed": seed},
        }
    )
    pot = cfg.build_potential()
    weigher = engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    state = ChainState.fresh(2, seed=seed)
    state.positions = np.empty((0, 2))
    series = [
        engine.advance(state, cfg, pot, weigher)["packing_fraction"]
        for _ in range(nwin)
    ]

    # the chain must never manufacture an overlap either
    pts = state.positions
    delta = pts[:, None, :] - pts[None, :, :]
    delta -= np.round(delta / BOX) * BOX
    dist = np.sqrt((delta**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    assert dist.min() > SIGMA - 1e-9, f"{kind} produced an overlap: {dist.min():.6f}"

    tail = np.asarray(series[int(BURN_FRAC * nwin) :])
    width = len(tail) // N_BLOCK
    blocks = tail[: N_BLOCK * width].reshape(N_BLOCK, width).mean(axis=1)
    return float(tail.mean()), float(blocks.std(ddof=1) / np.sqrt(N_BLOCK))


def main(argv):
    nwin = 2200 if (argv and argv[0] == "full") else 400
    cases = [
        ("metropolis", {"kind": "metropolis"}),
        (
            "rosenbluth d1",
            {"kind": "rosenbluth", "depth": 1, "trials": 30, "side": 2.0},
        ),
        ("branch d1", {"kind": "branch", "depth": 1, "trials": 30, "side": 2.0}),
        ("branch d2", {"kind": "branch", "depth": 2, "trials": 40, "side": 3.0}),
    ]
    results = {}
    for label, kwargs in cases:
        mean, err = run(nwin=nwin, **kwargs)
        results[label] = (mean, err)
        print(f"  {label:16s} eta = {mean:.5f} +- {err:.5f}", flush=True)

    ref, ref_err = results["metropolis"]
    fails = 0
    print()
    for label, (mean, err) in results.items():
        if label == "metropolis":
            continue
        combined = float(np.hypot(err, ref_err))
        z = abs(mean - ref) / combined
        verdict = "ok  " if z <= 3.0 else "FAIL"
        if z > 3.0:
            fails += 1
        print(
            f"  {verdict} {label:16s} d_eta = {mean - ref:+.5f} "
            f"+- {combined:.5f}  ({z:.1f} sigma)"
        )

    if fails:
        raise SystemExit(
            f"\n{fails} sampler(s) disagree with metropolis by more than 3 sigma. "
            f"The depth >= 1 proposal or the clique draw has regressed; check that "
            f"remove/repack still offsets the anchor (metropolis_dynamics, "
            f"parallel) and that "
            f"count_and_sample is still called with rng="
        )
    print("\nALL MIRROR-EOS TESTS PASSED")


if __name__ == "__main__":
    main(sys.argv[1:])
