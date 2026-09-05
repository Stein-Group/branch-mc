"""Gate S3, the statistical arm: each species is exchanged at its OWN activity.

WHY THIS FILE EXISTS SEPARATELY FROM THE ARBITER. The arbiter
(`test_species_bookkeeping_arbiter.py`) runs on species with EQUAL activities, so
the per-species activity term is identically zero there and the arbiter is blind
to it by construction. The reduction gate in `test_typed_energy.py` runs on ONE
species, where it is blind for the same reason. Between them they covered every
other part of the typed path and could not see this one.

The defect they missed: the exchange acceptance carried a single scalar
`beta * mu` -- species 0's chemical potential -- for the net particle a move
adds or removes, so every species was exchanged at the first one's activity.
`species.z` reached only the swap acceptance.

AND SWAP HID IT. With the identity swap active the composition came out right
anyway, because swap equilibrates composition at the correct activities and the
exchange's error is invisible in the stationary distribution it produces. Measured
before the fix, degenerate species at mu = [1.5, 0, -1.5]:

    swap active      x_s = [0.569, 0.311, 0.121]   looks correct
    swap suppressed  x_s = [0.323, 0.346, 0.332]   uniform: z_s never used

That is the second time in this stage that a passing gate meant nothing until an
equilibrating move was switched off. Every test below therefore runs with swap
SUPPRESSED, and `test_swap_masks_the_exchange` pins that fact so nobody
"simplifies" the suppression away and restores a test that cannot fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import MC_engine
from gcmc.proposals import moves
from gcmc.run import sim_setup
from gcmc.system.state import ChainState

pytestmark = pytest.mark.slow

_FILE = "ff/species_degenerate_reduced.yaml"


def _no_swap(rng, _n_species):
    """The three-way split whatever the species count: swap suppressed."""
    return str(rng.choice(moves.SINGLE_MOVES, p=moves.SINGLE_PROBS))


def _run(mus: dict, *, box: float, n_windows: int, suppress_swap: bool, seed: int = 11):
    cfg = sim_setup.from_params(
        {
            "potential": {"main": "lennard_jones", "T": 2.0, "species_file": _FILE},
            "sampler": {"type": "rosenbluth", "depth": 0, "trials": 20},
            "ensemble": {"mu": mus, "dimension": 3, "box": box},
            "run": {"sampling_window": 300},
        }
    )
    pot = cfg.build_potential()
    wgh = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    real = MC_engine.moves.draw_move
    if suppress_swap:
        MC_engine.moves.draw_move = _no_swap
    try:
        state = ChainState(
            positions=np.empty((0, 3)), rng=np.random.default_rng(seed), step=0
        )
        state.species = np.zeros(0, dtype=np.int64)
        counts = np.zeros(3)
        samples = 0
        for w in range(n_windows):
            MC_engine.advance(state, cfg, pot, wgh)
            assert state.species is not None
            if w >= n_windows // 3:
                counts += np.bincount(state.species, minlength=3)
                samples += 1
        return cfg, counts / max(samples, 1)
    finally:
        MC_engine.moves.draw_move = real


_SKEWED = {"A": -3.0, "B": -3.5, "C": -4.0}


def test_the_composition_tracks_the_activities_not_the_proposal() -> None:
    """x_s = z_s / sum(z_s), with swap suppressed so only the exchange can set it.

    `p_trial` is [0.7, 0.2, 0.1] in this file and the activities give
    [0.419, 0.327, 0.254], so the three candidate answers -- activities, proposal
    probabilities, and uniform -- are far apart and the measurement says which one
    the engine is producing.
    """
    cfg, mean_n = _run(_SKEWED, box=8.0, n_windows=60, suppress_swap=True)
    z = np.asarray(cfg.species.z)
    want = z / z.sum()
    got = mean_n / mean_n.sum()

    assert np.allclose(got, want, atol=0.03), (
        f"composition {np.round(got, 4)} against activities {np.round(want, 4)}.\n"
        f"  uniform would be   {np.round(np.full(3, 1 / 3), 4)}  (the exchange is "
        "using one scalar chemical potential for every species)\n"
        f"  p_trial is         {np.round(np.asarray(cfg.species.p_trial), 4)}  "
        "(the M21 correction is not cancelling the proposal)\n"
        "Which of those it lands on says which term is missing."
    )


def test_the_ideal_gas_limit_holds_per_species() -> None:
    """<N_s> ~ z_s V. The absolute scale, not just the ratios.

    A composition test alone cannot catch an activity that is uniformly wrong by
    a constant factor -- the ratios would still be right. This pins the scale.
    The tolerance is loose because the state point is not truly ideal: at
    <N> ~ 260 in this box the LJ attraction pulls the total a few percent below
    the ideal-gas value, which is physics rather than slack.
    """
    cfg, mean_n = _run(_SKEWED, box=8.0, n_windows=60, suppress_swap=True)
    want = np.asarray(cfg.species.z) * cfg.total_volume

    assert np.allclose(mean_n, want, rtol=0.12), (
        f"<N_s> = {np.round(mean_n, 1)} against z_s V = {np.round(want, 1)}. "
        "The ratios can be right while the scale is wrong, which is what this "
        "checks and the composition test cannot."
    )


def test_swap_masks_the_exchange() -> None:
    """The reason every test here suppresses swap, asserted rather than described.

    With swap ACTIVE the composition is correct whether or not the exchange uses
    per-species activities, because swap equilibrates it at the right ones. So a
    test of the exchange that leaves swap running cannot fail, and this asserts
    that the masking is real -- if it ever stops being real, the suppression above
    is unnecessary complexity and should go.

    Deliberately NOT a check that swap is broken. Swap is correct; the point is
    that a correct move can conceal an incorrect one, and which arm you measure in
    decides whether you find out.
    """
    _, masked = _run(_SKEWED, box=8.0, n_windows=40, suppress_swap=False)
    x = masked / masked.sum()
    assert np.allclose(x, np.asarray(_z_of()) / np.sum(_z_of()), atol=0.05), (
        "with swap active the composition should look right regardless; if this "
        "fails, swap itself has regressed and the arbiter should be run"
    )


def _z_of() -> np.ndarray:
    cfg = sim_setup.from_params(
        {
            "potential": {"main": "lennard_jones", "T": 2.0, "species_file": _FILE},
            "sampler": {"type": "rosenbluth", "depth": 0, "trials": 20},
            "ensemble": {"mu": _SKEWED, "dimension": 3, "box": 8.0},
            "run": {"sampling_window": 300},
        }
    )
    return np.asarray(cfg.species.z)


def test_a_repack_carries_an_activity_term() -> None:
    """N-conserving does not mean composition-conserving.

    A repack removes k atoms and inserts k, so the single-species engine correctly
    carries no activity term at all -- and that is exactly why the typed term is
    easy to omit there. The k that leave and the k that arrive need not be the
    same species, so the term is generally non-zero. Asserted on the function
    directly, because reaching a repack from a params file needs NVT and the point
    is about the algebra.
    """
    log_z = np.log(np.array([2.0, 1.0, 0.5]))
    pool_species = np.array([0, 1, 2, 0])
    # two members (rows 0, 1) leave; two other rows (2, 3) arrive: N conserved
    term = moves.species_activity_term(log_z, pool_species, [0, 1], [2, 3])
    expected = (log_z[2] + log_z[0]) - (log_z[0] + log_z[1])
    assert np.isclose(term, expected)
    assert not np.isclose(term, 0.0), (
        "a repack that swaps a B for a C changes the composition, so its activity "
        "term is not zero even though the particle count is unchanged"
    )


def test_a_reselected_member_contributes_nothing() -> None:
    """The third provenance case, shared with the M21 correction.

    A member reselected at its own row is the same particle with the same species
    in both configurations. It is neither inserted nor deleted and must cancel
    from both sums -- and it is not a rare case: member reselection was measured
    at 12 of 214 in one `hs_branch` window.
    """
    log_z = np.log(np.array([2.0, 1.0, 0.5]))
    spc = np.array([0, 1, 2])
    # row 1 is a member AND chosen: reselected, so it must not appear either side
    assert moves.species_activity_term(log_z, spc, [1], [1]) == 0.0
    # and with it reselected alongside a genuine insertion, only the insertion counts
    got = moves.species_activity_term(log_z, spc, [1], [1, 2])
    assert np.isclose(got, log_z[2])
