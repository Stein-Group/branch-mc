"""Item 1: the muVT move selection must not force "add" at N=0.

For hard spheres the single-occupancy configurational integral is exactly Q_1 = V
(one sphere never overlaps itself in a box wider than sigma), and Q_0 = 1, so the
grand-canonical ratio

    P(N=0) / P(N=1) = 1 / (z V),   z = exp(mu),

is exact and free of every higher-N term, because both probabilities carry the
same 1/Xi. That makes it a clean, analytic target: no MC integration of Q_2 is
needed. The old code proposed "add" with probability 1 at N=0 while proposing
"remove" with probability 1/4 at N=1, an asymmetry its acceptance never corrected,
which skews this ratio by a factor of four. The fixed code draws one fixed move
distribution at every N, so the measured ratio matches 1/(zV).

The same ratio must hold for the branch sampler, which shares the exchange logic.
"""

import math

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics
from gcmc.run import sim_setup
from gcmc.system.state import ChainState

# Slow tier: every test here measures an occupancy ratio P(N=0)/P(N=1) by long
# muVT sampling (~240s total). The move-selection / RNG structure these guard is
# already frozen in the default tier by the stage-A trajectory identity oracle.
pytestmark = pytest.mark.slow

BOX = 2.0
DIM = 3
MU = -2.7726  # z = exp(mu) = 0.0625, so zV = 0.5 and P0/P1 = 2.0
N_STEPS = 200_000
BURN = 5_000


def _sample_pn(sampler_block, steps=N_STEPS, seed=1):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": DIM, "box": BOX},
        "sampler": sampler_block,
        "run": {"sampling_window": 1},
    }
    cfg = sim_setup.from_params(p)
    pot = cfg.build_potential()
    w = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(DIM, seed=seed)
    st.positions = np.empty((0, DIM))
    counts = {}
    for i in range(steps + BURN):
        metropolis_dynamics.advance(st, cfg, pot, w)
        if i >= BURN:
            n = len(st.positions)
            counts[n] = counts.get(n, 0) + 1
    return counts, cfg.box**DIM


def _check_ratio(name, sampler_block, tol, steps=N_STEPS):
    counts, V = _sample_pn(sampler_block, steps=steps)
    z = math.exp(MU)
    expected = 1.0 / (z * V)
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    measured = n0 / n1
    rel = abs(measured - expected) / expected
    total = sum(counts.values())
    hist = {k: round(counts.get(k, 0) / total, 4) for k in sorted(counts)}
    print(
        f"[{name}] P0/P1 measured={measured:.4f} expected={expected:.4f} "
        f"rel_err={rel:.3f} P(N)={hist}"
    )
    assert rel < tol, f"{name}: P0/P1 off by {rel:.1%} (>{tol:.0%})"
    return measured, expected


def test_metropolis_n0_ratio():
    _check_ratio("metropolis", {"type": "metropolis"}, tol=0.05)


def test_branch_n0_ratio():
    _check_ratio(
        "branch",
        {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
        tol=0.06,
    )


# NO CHANNEL ARM HERE ANY MORE. Box 2 admits no fixed-length rod -- the periodic
# guard needs sqrt(L^2 + 2 w^2) <= box - sigma = 1.0, which no admissible width
# allows -- and the two tests that used to cover it ran the CYCLIC rod, whose two
# ends met through the boundary. That rod is gone (no separable exterior, so no
# ladder engine could run it), and its replacement, channel_length = "full",
# cannot help here either: a frozen gap of one sigma leaves a body of length one
# in a box of two. The P0/P1 ratio is still the sharpest cheap test of detailed
# balance, so a channel arm belongs in this file again at a box that can hold a
# rod; it is not a free parametrisation of this one.
if __name__ == "__main__":
    test_metropolis_n0_ratio()
    test_branch_n0_ratio()
    print("PASS test_item01_detailed_balance")
