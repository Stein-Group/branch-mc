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
N_STEPS_CHANNEL = 100_000  # channel moves cost more per step
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


# The full-span channel is the only channel variant that can run in this box: the
# fixed-length rod must satisfy sqrt(L^2 + 2 w^2) <= box - sigma = 1.0, which no
# admissible width allows at box 2. The full-span rod lifts that guard and instead
# scores minimum-image distances, so it spans the box and its two ends meet
# through the periodic boundary. P0/P1 is a property of the equilibrium
# distribution alone, so it holds for any move obeying detailed balance: it is the
# sharpest cheap test that the seam is neither double counted nor dropped.
# R16: the rod kind comes from the SAMPLER, not from a length. This block said
# `type: "channel"` with `channel_length: "full"`, which `from_params` has
# refused since R16 landed -- so both tests below have been erroring, not
# failing on physics, ever since. Nothing noticed because the slow suite had
# never been run against this tree.
_FULL_SPAN = {
    "type": "channel_cyclic",
    "trial_density": 2.5,
    "channel_width": 1.0,
}


def test_channel_cyclic_n0_ratio():
    _check_ratio(
        "channel_cyclic", {**_FULL_SPAN, "depth": 1}, tol=0.05, steps=N_STEPS_CHANNEL
    )


def test_channel_cyclic_tetris_n0_ratio():
    _check_ratio(
        "channel_cyclic_tetris",
        {**_FULL_SPAN, "full_repacking": True},
        tol=0.05,
        steps=N_STEPS_CHANNEL,
    )


if __name__ == "__main__":
    test_metropolis_n0_ratio()
    test_branch_n0_ratio()
    test_channel_cyclic_n0_ratio()
    test_channel_cyclic_tetris_n0_ratio()
    print("PASS test_item01_detailed_balance")
