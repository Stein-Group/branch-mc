"""The conventional sampler against Carnahan-Starling, at three densities.

Why this exists. tests/engine/test_stageB_eos.py records a real, pre-existing
discrepancy against CS -- about -2% for hard spheres, -11% for LJ -- and is
careful to say it is a finding about the method rather than about any refactor.
That test drives the CHANNEL sampler. Nothing pinned the conventional
rosenbluth/cavity path against the same oracle, so the -2% had no established
baseline to be a deviation FROM.

It does now: the conventional path reproduces CS to a few parts in ten thousand
across mu = 0, 1.5 and 3.0. That narrows the older finding to the sweep, and it
gives Phase 1-4 a Hamiltonian-level oracle that does not depend on the frozen
original -- which matters, because the original itself does not reproduce CS at
these densities (mof_raspa_plan.md ledger, F2).

Carnahan-Starling for hard spheres:

    beta mu = ln(rho) + (8 eta - 9 eta^2 + 3 eta^3) / (1 - eta)^3,  eta = pi rho / 6

exact enough below freezing (eta < 0.494) that any real disagreement here is the
sampler's, not the equation's.

The error bar is taken over independent seeds, not over sampling windows, for the
reason spelled out at length in test_stageB_eos.py: N's autocorrelation time is
far longer than one window, so window-wise standard errors understate the true
error several-fold and manufacture disagreements between an engine and itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from gcmc import config_specs
from gcmc.engine import MC_engine as engine
from gcmc.run import sim_setup as config
from gcmc.system.state import ChainState

# mu, and the tolerance in eta. 0.004 is roughly 1.5% at these densities: loose
# enough for 4 seeds x 60 windows, tight enough that the original's +2.8% at
# mu = 1.5 and +22% at mu = 3.0 would both fail it outright.
POINTS = [(0.0, 0.004), (1.5, 0.004), (3.0, 0.004)]
SEEDS = (11, 22, 33, 44)
WINDOWS = 60


def cs_eta(mu: float) -> float:
    """The packing fraction Carnahan-Starling puts at this chemical potential."""

    def f(eta: float) -> float:
        rho = 6.0 * eta / np.pi
        return np.log(rho) + (8 * eta - 9 * eta**2 + 3 * eta**3) / (1 - eta) ** 3 - mu

    return float(brentq(f, 1e-9, 0.49))


def _run(mu: float, seed: int) -> float:
    cfg = config.from_params(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": mu, "dimension": 3},
            "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
            "run": {"sampling_window": 1000},
        }
    )
    pot = cfg.build_potential()
    w = engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(cfg.dimension, seed=seed)
    st.positions = np.random.default_rng(seed).uniform(0, cfg.box, size=(60, 3))
    eta = [engine.advance(st, cfg, pot, w)["packing_fraction"] for _ in range(WINDOWS)]
    return float(np.mean(eta[WINDOWS // 2 :]))  # drop the fill-up half


@pytest.mark.slow
@pytest.mark.parametrize("mu,tol", POINTS, ids=[f"mu{p[0]}" for p in POINTS])
def test_conventional_sampler_matches_carnahan_starling(mu, tol):
    """One number per seed; the spread across seeds is the error bar."""
    per_seed = np.array([_run(mu, s) for s in SEEDS])
    got = per_seed.mean()
    err = per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    want = cs_eta(mu)
    print(
        f"\n  mu = {mu:4.1f}   engine eta = {got:.4f} +/- {err:.4f} (n = {len(SEEDS)} seeds)"
        f"   CS = {want:.4f}   diff = {got - want:+.4f} ({100 * (got - want) / want:+.2f}%)"
    )
    assert abs(got - want) < tol, (
        f"mu = {mu}: engine eta {got:.4f} +/- {err:.4f} vs Carnahan-Starling {want:.4f}. "
        "The conventional path has drifted off the equation of state; this is not a "
        "threshold to loosen."
    )


@pytest.mark.slow
def test_the_deviation_does_not_grow_with_density():
    """The shape that caught the original (ledger F2): it tracks CS at eta = 0.13
    and is +22% by eta = 0.26, because its error rides on cavity occupancy. A
    per-point tolerance alone would not have caught that at low density, so assert
    the shape too: the deviation must not grow as the cavity fills."""
    devs = []
    for mu, _ in POINTS:
        got = float(np.mean([_run(mu, s) for s in SEEDS[:2]]))
        devs.append(abs(got - cs_eta(mu)) / cs_eta(mu))
    print(f"\n  relative deviation vs CS by mu {[f'{d:.4f}' for d in devs]}")
    assert max(devs) < 0.015
    assert devs[-1] < devs[0] + 0.01, f"deviation grows with density: {devs}"
