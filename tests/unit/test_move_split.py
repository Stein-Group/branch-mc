"""M8 of multispecies_plan.md: the move split and the swap destination draw.

The split is four numbers and a branch, which is exactly the sort of thing that
gets copied into a third loop and then drifts. These tests pin the three claims
the plan makes about it, and one of them (the uniform destination) is asserted
because M8 asks for it explicitly, so that nobody optimises it into a silent
bias later.

No chain, no physics, all sub-second.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import moves


def test_the_multi_species_split_is_the_single_one_scaled_uniformly():
    """The multi-species vector is 0.5/0.25/0.25 times (1 - p_swap), so EVERY
    ratio among the non-swap moves is preserved, not just add against remove.

    That stronger statement is the one worth pinning. The grand-canonical ratio
    needs P(add) == P(remove), and the translate share relative to them sets how
    much configurational relaxation happens between exchanges, so a split that
    shifted it would change the sampler's character as well as its algebra.

    Note the comment this replaces, in `mof_metropolis`, said the swap was carved
    out of the translate budget. It is not: translation pays 0.1 and add and
    remove pay 0.05 each. The invariant survives because all three shrink by the
    same FACTOR.
    """
    for m, p in (
        (moves.SINGLE_MOVES, moves.SINGLE_PROBS),
        (moves.MS_MOVES, moves.MS_PROBS),
    ):
        d = dict(zip(m, p))
        assert d["add"] == d["remove"]
        assert sum(p) == pytest.approx(1.0, abs=1e-15)

    p_swap = dict(zip(moves.MS_MOVES, moves.MS_PROBS))["swap"]
    scale = 1.0 - p_swap
    for single, ms in zip(moves.SINGLE_PROBS, moves.MS_PROBS[:3]):
        assert ms == pytest.approx(single * scale, rel=1e-15)
    assert moves.MS_MOVES[:3] == moves.SINGLE_MOVES


def test_the_single_species_split_consumes_the_stream_exactly_as_before():
    """The load-bearing test. `rng.choice` consumes different amounts of the
    stream for different argument shapes, so a four-way call with a zero swap
    probability would be a DIFFERENT CHAIN even though it can never return
    "swap". Every single-species reference trajectory depends on the guard taking
    the frozen three-way call.

    Checked by consuming a long run of draws and then comparing the RNG's
    remaining stream against a reference generator driven by the literal old
    call. Comparing the draws alone would not catch a call that returned the same
    labels while eating a different number of values.
    """
    a = np.random.default_rng(20260813)
    b = np.random.default_rng(20260813)
    got = [moves.draw_move(a, 1) for _ in range(300)]
    want = [
        str(b.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25]))
        for _ in range(300)
    ]
    assert got == want
    # and the streams are still in lockstep afterwards
    assert a.random(20).tolist() == b.random(20).tolist()


def test_a_single_species_run_can_never_draw_a_swap():
    rng = np.random.default_rng(1)
    assert "swap" not in {moves.draw_move(rng, 1) for _ in range(2000)}


def test_the_multi_species_split_hits_its_probabilities():
    """A frequency check with a tolerance from the binomial standard error, not a
    number tuned until it passed: for n draws at probability p the count has
    sd = sqrt(n p (1-p)), and 4 sd is a ~1e-4 false-failure rate per category."""
    n = 200_000
    rng = np.random.default_rng(7)
    draws = [moves.draw_move(rng, 3) for _ in range(n)]
    for name, p in zip(moves.MS_MOVES, moves.MS_PROBS):
        got = draws.count(name)
        sd = np.sqrt(n * p * (1.0 - p))
        assert abs(got - n * p) <= 4.0 * sd, f"{name}: {got} vs {n * p} +/- {sd:.0f}"


# ------------------------------------------------------- the destination draw


def test_the_swap_destination_is_uniform_over_the_others():
    """M8 asks for this to be asserted. Uniform over the other species is a
    SYMMETRIC proposal, so the acceptance ratio is z_s'/z_s exp(-beta dU) with no
    proposal correction. Drawing from p_trial instead would need a p_s/p_s'
    factor, which is not there -- so a well-meaning change to draw from p_trial
    would introduce a bias with nothing else in the code to catch it.
    """
    n, n_species = 120_000, 4
    rng = np.random.default_rng(3)
    for current in range(n_species):
        draws = [
            moves.draw_swap_target(rng, current, range(n_species)) for _ in range(n)
        ]
        assert current not in set(draws), "a swap returned the current species"
        p = 1.0 / (n_species - 1)
        for t in [t for t in range(n_species) if t != current]:
            got = draws.count(t)
            sd = np.sqrt(n * p * (1.0 - p))
            assert abs(got - n * p) <= 4.0 * sd, f"{current}->{t}: {got}"


def test_two_species_forces_the_destination_and_draws_nothing():
    """With one candidate there is no decision, so asking the RNG would consume a
    value for nothing -- and that value would shift every later draw."""
    a = np.random.default_rng(5)
    b = np.random.default_rng(5)
    assert moves.draw_swap_target(a, 0, [0, 1]) == 1
    assert moves.draw_swap_target(a, 1, [0, 1]) == 0
    assert a.random(10).tolist() == b.random(10).tolist()


def test_non_contiguous_species_ids_are_respected():
    """`mof_metropolis` derives its candidates from the state-point dict, so a run
    over a SUBSET of a force field's guests has non-contiguous ids. A helper that
    assumed range(n) would swap into a species the run does not contain."""
    rng = np.random.default_rng(2)
    ids = [0, 2, 5]
    got = {moves.draw_swap_target(rng, 2, ids) for _ in range(500)}
    assert got == {0, 5}


def test_a_swap_with_nothing_to_swap_into_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least two species"):
        moves.draw_swap_target(rng, 0, [0])


# ------------------------------------------------------------- the acceptance


def test_the_swap_ratio_is_the_activity_ratio_alone():
    """No n_s/(n_s'+1) combinatorial term: a specific labelled particle changes
    identity, so adding one would double count the uniform-particle selection and
    bias the mixture toward the minority species with total N conserved -- easy to
    mistake for physics."""
    log_z = np.log(np.array([0.2, 0.5, 0.1]))
    got = moves.swap_log_acceptance(log_z, 0, 1, beta=1.0 / 1.2, du=0.75)
    assert got == pytest.approx(np.log(0.5 / 0.2) - 0.75 / 1.2, rel=1e-15)


def test_the_swap_ratio_is_antisymmetric_at_zero_energy_change():
    """s -> s' and s' -> s must be exact negatives when dU is zero, or the swap
    does not satisfy detailed balance for any activity pair."""
    log_z = np.log(np.array([0.3, 0.9]))
    f = moves.swap_log_acceptance(log_z, 0, 1, beta=0.8, du=0.0)
    r = moves.swap_log_acceptance(log_z, 1, 0, beta=0.8, du=0.0)
    assert f == pytest.approx(-r, rel=1e-15)


def test_identical_activities_reduce_to_the_boltzmann_factor():
    log_z = np.log(np.array([0.4, 0.4, 0.4]))
    assert moves.swap_log_acceptance(log_z, 2, 0, beta=1.5, du=2.0) == pytest.approx(
        -3.0, rel=1e-15
    )


# ------------------------------- M21, the species-draw correction (unvalidated)
#
# These pin the STRUCTURE of the derived expression, not its correctness. The
# arbiter for correctness is the degenerate-species run of plan section 16, which
# needs the typed insert path first. What these catch is the expression drifting
# from what was derived, and the two mistakes most likely to be made when wiring
# it: charging a reselected member, and getting the sign backwards.


def _corr(p, spc, members, chosen):
    return moves.species_draw_correction(
        np.log(np.asarray(p, float)), np.asarray(spc, np.int64), members, chosen
    )


def test_a_reselected_member_is_charged_nothing():
    """The whole point of M21. A member the DP puts back at its own pool row is the
    same particle with the same species in both configurations, so it was drawn in
    neither direction. A per-insertion factor charges it p_s and biases composition
    at a rate set by how often the DP reselects -- measured at roughly 6% of
    apparently-stationary particles per window."""
    p = [0.2, 0.3, 0.5]
    spc = [0, 1, 2, 1]
    # rows 2 and 3 are the tombstoned members; the DP puts both back unchanged
    assert _corr(p, spc, [2, 3], [2, 3]) == pytest.approx(0.0, abs=1e-15)
    # and a pure repack that swaps which species occupies the body is NOT free
    assert _corr(p, spc, [2, 3], [0, 1]) != pytest.approx(0.0, abs=1e-12)


def test_a_pure_insertion_is_charged_the_forward_draw():
    """One trial atom of species 2 inserted, nothing removed."""
    p = [0.2, 0.3, 0.5]
    got = _corr(p, [0, 1, 2], [], [2])
    assert got == pytest.approx(-np.log(0.5), rel=1e-15)


def test_a_pure_deletion_is_charged_the_reverse_draw():
    """One member of species 0 removed and not put back. The reverse move has to
    regenerate it as a trial draw, so the term appears with the opposite sign to an
    insertion."""
    p = [0.2, 0.3, 0.5]
    got = _corr(p, [0, 1, 2], [0], [])
    assert got == pytest.approx(np.log(0.2), rel=1e-15)


def test_insertion_and_deletion_of_the_same_species_cancel():
    """A one-for-one exchange within a species changes nothing about the species
    bookkeeping, whatever p_s is. If this ever fails, the two branches are not
    reading the same probability vector."""
    p = [0.2, 0.3, 0.5]
    spc = [1, 1, 0, 2]
    assert _corr(p, spc, [0], [1]) == pytest.approx(0.0, abs=1e-15)


def test_at_uniform_p_the_correction_is_the_net_count_times_log_T():
    """A sign check that can be read off by hand. At p_s = 1/T the sum collapses to
    (n_deleted - n_inserted) log(1/T) = +dN log T. Adding a particle to a
    T-species mixture means the proposal had to guess among T species, so the
    forward proposal is T times less likely and the ratio gains a factor of T.

    This is the test that catches an inverted prefactor, which is the mistake this
    acceptance has already suffered once.
    """
    rng = np.random.default_rng(0)
    for T in (2, 3, 5):
        p = np.full(T, 1.0 / T)
        for _ in range(8):
            n_pool = 14
            spc = rng.integers(T, size=n_pool)
            members = rng.choice(n_pool, size=4, replace=False)
            chosen = rng.choice(n_pool, size=5, replace=False)
            n_del = len(set(members.tolist()) - set(chosen.tolist()))
            n_ins = len(set(chosen.tolist()) - set(members.tolist()))
            want = (n_ins - n_del) * np.log(T)
            assert _corr(p, spc, members, chosen) == pytest.approx(want, rel=1e-14)


def test_provenance_is_by_row_and_not_by_species():
    """Two pool rows may hold the same species; only the row decides whether an
    atom was drawn. A member of species 1 removed while a DIFFERENT trial atom of
    species 1 is inserted must cancel, and that is only true if both are found by
    row."""
    p = [0.4, 0.6]
    spc = [1, 1, 1, 0]
    assert _corr(p, spc, [2], [0]) == pytest.approx(0.0, abs=1e-15)
    # while removing species 1 and inserting species 0 does not cancel
    assert _corr(p, spc, [2], [3]) == pytest.approx(
        np.log(0.6) - np.log(0.4), rel=1e-15
    )


def test_an_empty_exchange_is_free():
    p = [0.4, 0.6]
    assert _corr(p, [0, 1], [], []) == 0.0
