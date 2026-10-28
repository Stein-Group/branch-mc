"""The semi-grand composition is the one the activities predict, and it is not
the one `p_trial` predicts.

This is the M21 arbiter transplanted to a third sampler path, for the reason
S4 phase 7 gives when it transplanted it to the second: the species-draw
bookkeeping is separate code on each path, and a derivation verified on one says
nothing about the others. Here the path is NVT, where the engine forces the
repack move -- remove k, insert k -- so the composition relaxes while N does not
move at all.

TWO STATE POINTS, and they fail in different directions:

  * EQUAL activities. Degenerate species are one species wearing three labels, so
    the stationary composition must be uniform for any `p_trial` whatsoever --
    `p_trial` is a proposal knob and a proposal knob cannot move an equilibrium
    distribution (M6, M21). The model file's `p_trial` is (0.7, 0.2, 0.1), so an
    uncorrected species draw lands on THAT, and the two candidate answers are a
    third of the axis apart.
  * UNEQUAL activities. With identical species the semi-grand distribution over
    composition is multinomial with `p_s = z_s / sum_t z_t`, exactly, whatever
    the density, the temperature or the force field -- the labels do no physics.
    So `mu_C = -kT ln 2` must give x = (0.4, 0.4, 0.2) and nothing else. This is
    the arm that would catch an activity ratio dropped, inverted or applied once
    per repack instead of once per particle: the equal-activity arm above is
    blind to all three, since at equal activities every such error is the
    identity.

BOTH ARMS RUN FROM OPPOSITE STARTS AND MUST MEET. Not a formality: with the
identity swap unavailable at fixed N, labels relax only through the repack, so an
arm read too early reports its start. S4 phase 7 recorded that failure after it
nearly produced a false alarm, and the same rule applies here -- the assertion
below is on the two starts agreeing FIRST, and on the prediction second.

Tolerances are derived, not chosen: the per-window spread of `x_s` is the
multinomial `sqrt(p (1 - p) / N)`, and the window series is correlated, so the
band used is that sd rather than an error on the mean.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gcmc.forcefield.species_file import load_forcefield
from gcmc.run import io as gio
from gcmc.run import runner

pytestmark = pytest.mark.slow

MODEL = "ff/species_degenerate_reduced.yaml"
BOX = 8.0
N_SIDE = 7
T = 1.5
WINDOWS = 70
BURN = 35


class _FramePaths:
    """The one attribute `io.write_frame` reads off a RunPaths."""

    def __init__(self, trajectory) -> None:
        """Hold the destination."""
        self.trajectory = trajectory


def _run_arm(tmp_path, tag: str, comp, mu: dict[str, float]) -> np.ndarray:
    """One arm: build a start at composition `comp`, run, return x_s per window."""
    ff = load_forcefield(MODEL)
    g = (np.arange(N_SIDE) + 0.5) * (BOX / N_SIDE)
    pos = np.array([[x, y, z] for x in g for y in g for z in g])
    n = len(pos)
    counts = [int(round(c * n)) for c in comp]
    counts[-1] = n - sum(counts[:-1])
    labels = np.concatenate(
        [np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]
    )
    spc = np.random.default_rng(2).permutation(labels)

    start = tmp_path / f"{tag}.extxyz"
    gio.write_frame(
        _FramePaths(start),
        pos,
        two_dim=False,
        box=BOX,
        info={"step": 0},
        append=False,
        species=spc,
        symbols=ff.symbol,
    )
    params = {
        "potential": {"main": "lennard_jones", "T": T, "species_file": MODEL},
        "sampler": {
            "type": "channel",
            "trial_density": 1.0,
            "channel_width": 1.5,
            "channel_length": 3.0,
        },
        "engine": {"depth": 1},
        "ensemble": {
            "type": "semigrand",
            "dimension": 3,
            "box": BOX,
            "N": n,
            "mu": mu,
        },
        "run": {
            "kind": "single",
            "seed": 13,
            "traj_every": 0,
            "target_cycles": WINDOWS * 2000,
            "sampling_window": 2000,
            "checkpoint_every": 20,
            "initial_config": str(start),
        },
    }
    p = tmp_path / f"{tag}.json"
    p.write_text(json.dumps(params))
    _state, paths = runner.run(str(p), root=str(tmp_path / f"runs_{tag}"))

    log = gio.read_log(paths.log)
    counts_series = np.column_stack([log[f"N_{s}"] for s in ("A", "B", "C")])
    assert set(counts_series.sum(axis=1).tolist()) == {float(n)}, (
        "N moved during a fixed-N run, so this is not the ensemble being tested"
    )
    return counts_series / counts_series.sum(axis=1, keepdims=True)


def _band(expected: np.ndarray, n: int) -> np.ndarray:
    """Per-window multinomial spread of x_s at `n` particles."""
    return np.sqrt(expected * (1.0 - expected) / n)


def _check(x_lo: np.ndarray, x_hi: np.ndarray, expected: np.ndarray, n: int) -> None:
    lo, hi = x_lo[BURN:].mean(axis=0), x_hi[BURN:].mean(axis=0)
    band = _band(expected, n)
    assert np.abs(lo - hi).max() < 2.0 * band.max(), (
        f"the two starts have NOT met: {lo.round(4).tolist()} against "
        f"{hi.round(4).tolist()}, band {band.round(4).tolist()}. Read no further "
        "-- an arm that has not converged reports its start, not its ensemble"
    )
    for label, x in (("start A", lo), ("start B", hi)):
        dev = np.abs(x - expected) / band
        assert dev.max() < 3.0, (
            f"{label} settled at {x.round(4).tolist()}, expected "
            f"{expected.tolist()}, which is {dev.round(2).tolist()} multinomial "
            "sd away"
        )


def test_equal_activities_give_a_uniform_composition_whatever_p_trial_is(tmp_path):
    mu = {"A": 0.0, "B": 0.0, "C": 0.0}
    x_lo = _run_arm(tmp_path, "eq_uniform", (1 / 3, 1 / 3, 1 / 3), mu)
    x_hi = _run_arm(tmp_path, "eq_skewed", (0.7, 0.2, 0.1), mu)
    expected = np.full(3, 1 / 3)
    _check(x_lo, x_hi, expected, N_SIDE**3)

    # The discrimination is worth stating as an assertion rather than a comment:
    # the wrong answer is not "somewhere else", it is p_trial, and the arm that
    # STARTED there is the one that has to leave it.
    p_trial = np.array([0.7, 0.2, 0.1])
    got = x_hi[BURN:].mean(axis=0)
    assert np.abs(got - p_trial).max() > 0.2, (
        f"the composition sat at {got.round(4).tolist()}, which is p_trial "
        f"{p_trial.tolist()} -- the species-draw correction is not being applied"
    )


def test_unequal_activities_give_the_activity_fractions_exactly(tmp_path):
    mu = {"A": 0.0, "B": 0.0, "C": -T * math.log(2.0)}  # z_C = z_A / 2
    x_lo = _run_arm(tmp_path, "mu_uniform", (1 / 3, 1 / 3, 1 / 3), mu)
    x_hi = _run_arm(tmp_path, "mu_skewed", (0.1, 0.2, 0.7), mu)
    _check(x_lo, x_hi, np.array([0.4, 0.4, 0.2]), N_SIDE**3)
