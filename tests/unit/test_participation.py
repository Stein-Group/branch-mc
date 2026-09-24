"""The participation solver: one target in place of nine fitted numbers.

`solve_gamma` is the whole adaptive-gamma method in one pure function, and
because it runs inside the selection loop of a kernel that has to satisfy
detailed balance, what it must be is stricter than what it must compute. Three
properties matter more than accuracy:

  DETERMINISM. Same inputs, same float, always. No state, no RNG, no cache.
  PERMUTATION INVARIANCE. The engine permutes the pool, so the same physical
    set of trials arrives in different orders and must yield the identical
    gamma, bit for bit. `np.sum` is order-dependent; this is the gate that says
    the sorted reduction actually fixes that.
  TOTALITY. Every pool the sampler can hand it gets an answer. A single trial,
    flat weights, one dominant weight, an exhausted pool, a target nothing can
    reach: all of these occur, and none of them may raise inside a move.

Accuracy barely matters by comparison. The grid has 17 points and the answer is
one of them; a gamma that lands a grid step away from the ideal one changes the
variance of the estimator and nothing else, because the bias is divided back out
of the acceptance.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gcmc.sampling.participation import (
    CRITERIA,
    N_GRID,
    SOLVERS,
    ParticipationTarget,
    gamma_grid,
    participation_fraction,
    participation_ratio,
    perplexity,
    solve_gamma,
    solve_gamma_moments,
    spread_fraction,
)

GAMMA_MAX = 3.0


# ---- the participation ratio itself -----------------------------------------


def test_flat_weights_participate_completely() -> None:
    """Every trial equally likely: PR is the pool size and the fraction is 1.
    This is what gamma = 0 tends towards and what a target of 1.0 asks for."""
    for n in (1, 2, 30):
        assert participation_ratio(np.ones(n)) == pytest.approx(n)
        assert participation_fraction(np.ones(n)) == pytest.approx(1.0)


def test_one_dominant_weight_participates_barely() -> None:
    """The failure the knob exists to detect: one trial takes every pick, so the
    other 29 were generated for nothing."""
    w = np.full(30, 1e-12)
    w[7] = 1.0
    assert participation_ratio(w) == pytest.approx(1.0, abs=1e-6)


def test_a_single_trial_is_fully_participating_not_degenerate() -> None:
    """k = 1 is the metropolis limit, where the bias cancels identically. PR = 1
    of a pool of 1 is a fraction of 1, not a division by zero."""
    assert participation_ratio(np.array([0.37])) == pytest.approx(1.0)
    assert participation_fraction(np.array([0.37])) == pytest.approx(1.0)


def test_an_exhausted_pool_reduces_to_zero_rather_than_nan() -> None:
    """Every trial annihilated (deep overlap underflow). The caller is about to
    reject the move; until then the arithmetic has to stay finite, because a NaN
    would make the grid comparison undefined and the gamma a matter of luck."""
    assert participation_ratio(np.zeros(12)) == 0.0
    assert participation_fraction(np.zeros(12)) == 0.0
    assert participation_fraction(np.array([])) == 0.0


def test_the_ratio_is_invariant_under_a_common_scale() -> None:
    """PR reads the SHAPE of the weight distribution, not its size. This is what
    lets the solver work in log space with the maximum shifted out."""
    rng = np.random.default_rng(0)
    w = rng.exponential(1.0, 40)
    for scale in (1e-30, 1e-3, 1.0, 1e12):
        assert participation_ratio(w * scale) == pytest.approx(
            participation_ratio(w), rel=1e-12
        )


def test_the_ratio_is_bounded_by_one_and_the_pool_size() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = rng.exponential(1.0, rng.integers(1, 40))
        pr = participation_ratio(w)
        assert 1.0 - 1e-12 <= pr <= len(w) + 1e-12


def test_the_reduction_is_order_independent_bit_for_bit() -> None:
    """The reason for the sort. Summed in the given order, these weights give
    different totals for different orderings; sorted first, they cannot."""
    rng = np.random.default_rng(2)
    for _ in range(2000):
        w = rng.exponential(1.0, 30) * 10.0 ** rng.integers(-8, 8)
        shuffled = w[rng.permutation(len(w))]
        assert participation_ratio(shuffled) == participation_ratio(w)


# ---- the grid -----------------------------------------------------------------


def test_the_grid_starts_at_zero_and_ends_at_gamma_max() -> None:
    """Zero is a real answer, not a padding value: the fitted schedule used it
    at levels 7 through 9."""
    grid = gamma_grid(GAMMA_MAX, 17)
    assert len(grid) == 17
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(GAMMA_MAX)
    assert np.all(np.diff(grid) > 0)


def test_the_grid_resolves_the_small_gammas_the_fitted_table_used() -> None:
    """The fitted 3D numbers run from 0.0037 to 0.2903. A linear grid over
    [0, 3] would put two points in that range; the log grid puts eight."""
    grid = gamma_grid(GAMMA_MAX, 17)
    assert np.count_nonzero((grid >= 0.0037) & (grid <= 0.2903)) >= 6


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_gamma_max_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="gamma_max"):
        gamma_grid(bad)


# ---- the solver ---------------------------------------------------------------


def _anticorrelated(n: int = 30, seed: int = 5):
    """The construction from the plan: a bias that likes what the Boltzmann
    factor dislikes. PR rises with gamma here before falling, so there is a
    target with no root at all and one with two."""
    rng = np.random.default_rng(seed)
    log_base = rng.normal(0.0, 2.0, n)
    return np.exp(log_base), -log_base + rng.normal(0.0, 0.3, n)


def _correlated(n: int = 30, seed: int = 6):
    """The ordinary case: the bias reinforces the Boltzmann factor, so raising
    gamma concentrates the weights and PR falls."""
    rng = np.random.default_rng(seed)
    log_base = rng.normal(0.0, 1.0, n)
    return np.exp(log_base), log_base + rng.normal(0.0, 0.3, n)


def test_the_solver_is_permutation_invariant_bit_for_bit() -> None:
    """Gate S3. The engine hands the same pool in different orders; two orders
    must give one gamma, exactly. Ten thousand pools, targets rotating across
    the useful range, and `==`, not `approx`."""
    rng = np.random.default_rng(7)
    targets = (0.2, 0.4, 0.6, 0.8)
    for i in range(10_000):
        n = int(rng.integers(1, 31))
        base = np.exp(rng.normal(0.0, 3.0, n))
        logb = rng.normal(0.0, 2.0, n)
        perm = rng.permutation(n)
        f_target = targets[i % len(targets)]
        a = solve_gamma(base, logb, f_target, GAMMA_MAX)
        b = solve_gamma(base[perm], logb[perm], f_target, GAMMA_MAX)
        assert a == b


def test_the_solver_returns_a_grid_point_and_nothing_else() -> None:
    """No interpolation, no refinement. The answer being on the grid is what
    makes a run reproducible from its config alone."""
    grid = set(gamma_grid(GAMMA_MAX, 17).tolist())
    rng = np.random.default_rng(8)
    for _ in range(500):
        n = int(rng.integers(1, 31))
        base = np.exp(rng.normal(0.0, 3.0, n))
        logb = rng.normal(0.0, 2.0, n)
        f = float(rng.uniform(0.05, 1.0))
        assert solve_gamma(base, logb, f, GAMMA_MAX) in grid


def test_the_solver_hits_a_reachable_target_closely() -> None:
    """Accuracy, such as it is: with 17 grid points the achieved fraction should
    land within a grid step of a target that is actually reachable."""
    base, logb = _correlated()
    grid = gamma_grid(GAMMA_MAX, 17)
    reach = np.array([participation_fraction(base * np.exp(g * logb)) for g in grid])
    for f_target in (reach.min() + 0.02, reach.mean(), reach.max() - 0.02):
        g = solve_gamma(base, logb, f_target, GAMMA_MAX)
        got = participation_fraction(base * np.exp(g * logb))
        assert abs(got - f_target) <= np.max(np.abs(np.diff(reach))) + 1e-9


def test_an_unreachably_low_target_clamps_to_zero_when_pr_falls() -> None:
    """Gate S3's clamping case. With the bias reinforcing the Boltzmann factor,
    PR falls with gamma, so a target above the gamma = 0 fraction cannot be
    reached from either side and gamma = 0 is the closest point. Exactly 0.0,
    because that is the value the engine counts as a clamped level."""
    base, logb = _correlated()
    at_zero = participation_fraction(base)
    assert solve_gamma(base, logb, min(1.0, at_zero + 0.3), GAMMA_MAX) == 0.0


def test_the_non_monotone_case_answers_instead_of_raising() -> None:
    """An anti-correlated bias makes PR rise with gamma before falling, which is
    what rules out bisection. The solver has to return a defined grid point for
    a target with no root, two roots, or one on the rising branch."""
    base, logb = _anticorrelated()
    grid = gamma_grid(GAMMA_MAX, 17)
    reach = np.array([participation_fraction(base * np.exp(g * logb)) for g in grid])
    assert reach[-1] < reach.max() - 1e-6 or reach[0] < reach.max() - 1e-6
    for f_target in (0.05, 0.2, 0.5, 0.9, 1.0):
        g = solve_gamma(base, logb, f_target, GAMMA_MAX)
        assert g in set(grid.tolist())


def test_a_target_below_the_zero_fraction_need_not_clamp_when_pr_rises() -> None:
    """The corollary the plan's clamp rule would get wrong, recorded so nobody
    "fixes" the solver by adding the special case back. Here the fraction at
    gamma = 0 sits below the target and a nonzero gamma still gets closer, so
    returning 0.0 would be the worse answer."""
    base, logb = _anticorrelated()
    grid = gamma_grid(GAMMA_MAX, 17)
    reach = np.array([participation_fraction(base * np.exp(g * logb)) for g in grid])
    rising = reach.max() > reach[0] + 1e-6
    assert rising, "construction failed: PR does not rise with gamma here"
    f_target = float(reach.max())
    assert solve_gamma(base, logb, f_target, GAMMA_MAX) != 0.0


# ---- degenerate pools ---------------------------------------------------------


def test_flat_weights_solve_without_raising() -> None:
    """No bias to tune (`log_bias` all zero): every gamma gives the same
    fraction of 1.0, so the tie goes to the smallest gamma."""
    assert solve_gamma(np.ones(20), np.zeros(20), 0.4, GAMMA_MAX) == 0.0
    assert solve_gamma(np.ones(20), np.zeros(20), 1.0, GAMMA_MAX) == 0.0


def test_a_single_trial_solves_to_zero() -> None:
    """One trial always participates fully, so no gamma changes anything and the
    tie goes to zero. This is the metropolis limit and it must not raise."""
    assert solve_gamma(np.array([2.5]), np.array([1.3]), 0.4, GAMMA_MAX) == 0.0


def test_an_empty_pool_solves_to_zero() -> None:
    """The selection loop rejects an empty pool at the next `selection` call;
    the solve on the way there must not raise."""
    assert solve_gamma(np.array([]), np.array([]), 0.4, GAMMA_MAX) == 0.0


def test_an_exhausted_pool_solves_to_zero() -> None:
    assert solve_gamma(np.zeros(15), np.ones(15), 0.4, GAMMA_MAX) == 0.0


def test_a_huge_log_bias_does_not_overflow_into_nan() -> None:
    """`log_bias` summed over thirty neighbours reaches a few hundred, and
    `exp(gamma_max * that)` overflows a float. The solver works in log space
    with the maximum shifted out, so this has to return a grid point rather than
    a NaN-poisoned one."""
    rng = np.random.default_rng(9)
    base = np.exp(rng.normal(0.0, 5.0, 30))
    logb = rng.normal(400.0, 50.0, 30)
    g = solve_gamma(base, logb, 0.4, GAMMA_MAX)
    assert np.isfinite(g)
    assert g in set(gamma_grid(GAMMA_MAX, 17).tolist())


def test_a_dominant_weight_and_a_flat_bias_solve_to_zero() -> None:
    base = np.full(30, 1e-14)
    base[3] = 1.0
    assert solve_gamma(base, np.zeros(30), 0.4, GAMMA_MAX) == 0.0


# ---- refusals ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_target_outside_the_unit_interval_is_refused(bad: float) -> None:
    """`pr_target` is a fraction of the pool, not a gamma. Somebody will type
    the old 0.2903 into it."""
    with pytest.raises(ValueError, match="pr_target"):
        solve_gamma(np.ones(5), np.zeros(5), bad, GAMMA_MAX)


def test_misaligned_factors_are_refused_with_the_likely_cause_named() -> None:
    """The shape mismatch that actually happens: unreferenced trials dropped
    from one factor instead of contributing log_bias = 0."""
    with pytest.raises(ValueError, match="row-aligned"):
        solve_gamma(np.ones(10), np.zeros(7), 0.4, GAMMA_MAX)


def test_the_solver_is_stateless_across_repeated_calls() -> None:
    """Rule 1, asserted rather than assumed: no warm start, no cached bracket.
    A thousand interleaved calls on two different pools return the same two
    answers they returned first."""
    a = _correlated(seed=21)
    b = _anticorrelated(seed=22)
    first = (
        solve_gamma(*a, 0.4, GAMMA_MAX),
        solve_gamma(*b, 0.4, GAMMA_MAX),
    )
    for _ in range(500):
        assert solve_gamma(*a, 0.4, GAMMA_MAX) == first[0]
        assert solve_gamma(*b, 0.4, GAMMA_MAX) == first[1]


# ---- the second criterion ------------------------------------------------------
#
# Which spread to target is an empirical question, so perplexity is a peer of the
# participation ratio rather than a patch on it. What has to hold is that the two
# are interchangeable in the solver: same shape, same range, same meaning for
# `pr_target`. Where they differ is what they punish, and the last two tests here
# pin that difference so a future edit cannot quietly make them the same measure
# under two names.


def test_both_criteria_read_the_pool_size_on_flat_weights() -> None:
    """The property that lets one `pr_target` mean the same thing under either:
    both run from 1 to k, and both are k when every trial is equally likely."""
    for n in (1, 2, 30):
        assert perplexity(np.ones(n)) == pytest.approx(n)
        for crit in CRITERIA:
            assert spread_fraction(np.ones(n), crit) == pytest.approx(1.0)


def test_both_criteria_bottom_out_at_one_trial() -> None:
    w = np.full(30, 1e-12)
    w[7] = 1.0
    assert perplexity(w) == pytest.approx(1.0, abs=1e-6)


def test_perplexity_is_order_independent_bit_for_bit() -> None:
    """The same sorted reduction, for the same reason: the engine permutes the
    pool and two orders must give one number."""
    rng = np.random.default_rng(31)
    for _ in range(2000):
        w = rng.exponential(1.0, 30) * 10.0 ** rng.integers(-8, 8)
        assert perplexity(w[rng.permutation(len(w))]) == perplexity(w)


def test_perplexity_survives_zero_weights() -> None:
    """p log p -> 0 as p -> 0, so annihilated trials contribute nothing. Taken
    literally the expression is 0 * log(0) = NaN, which is why the zeros are
    dropped rather than evaluated."""
    w = np.concatenate([np.zeros(10), np.ones(5)])
    assert perplexity(w) == pytest.approx(5.0)
    assert perplexity(np.zeros(12)) == 0.0
    assert np.isfinite(perplexity(w))


def test_the_two_criteria_disagree_about_a_dominant_trial() -> None:
    """The reason there are two. The participation ratio is a second moment, so
    one trial holding most of the weight pulls it almost to 1 whatever the rest
    look like; perplexity is an entropy and keeps reading the tail. If these two
    ever return the same number on this pool, one of them has stopped being
    itself."""
    w = np.concatenate([[0.8], np.full(29, 0.2 / 29)])
    pr = participation_ratio(w)
    pp = perplexity(w)
    # PR has all but collapsed to the single dominant trial; perplexity still
    # counts the 29 others as worth something
    assert pr < 2.0, pr
    assert pp > 1.5 * pr, (pr, pp)


@pytest.mark.parametrize("crit", sorted(CRITERIA))
def test_the_solver_takes_either_criterion(crit: str) -> None:
    """Every property the solver has to have, under both measures: a grid point
    out, permutation invariance, and no raise on a degenerate pool."""
    grid = set(gamma_grid(GAMMA_MAX, 17).tolist())
    rng = np.random.default_rng(41)
    for _ in range(300):
        n = int(rng.integers(1, 31))
        base = np.exp(rng.normal(0.0, 3.0, n))
        logb = rng.normal(0.0, 2.0, n)
        perm = rng.permutation(n)
        g = solve_gamma(base, logb, 0.3, GAMMA_MAX, criterion=crit)
        assert g in grid
        assert g == solve_gamma(base[perm], logb[perm], 0.3, GAMMA_MAX, criterion=crit)
    assert solve_gamma(np.zeros(9), np.ones(9), 0.3, GAMMA_MAX, criterion=crit) == 0.0


def test_the_criterion_changes_the_answer() -> None:
    """Otherwise the config key would be decoration. The two measures read the
    same pool differently, so at one target they generally solve to different
    exponents."""
    base, logb = _correlated(seed=51)
    a = solve_gamma(base, logb, 0.3, GAMMA_MAX, criterion="participation")
    b = solve_gamma(base, logb, 0.3, GAMMA_MAX, criterion="perplexity")
    assert a != b, (a, b)


def test_an_unknown_criterion_is_refused_at_every_entry_point() -> None:
    """The solver, the fraction helper and the frozen target object. A typo that
    fell through to a default would silently run the other method."""
    with pytest.raises(ValueError, match="unknown criterion"):
        spread_fraction(np.ones(4), "entropy")
    with pytest.raises(ValueError, match="unknown criterion"):
        solve_gamma(np.ones(4), np.zeros(4), 0.3, GAMMA_MAX, criterion="entropy")
    with pytest.raises(ValueError, match="unknown criterion"):
        ParticipationTarget(0.3, criterion="entropy")


def test_the_target_object_carries_the_criterion_to_every_level() -> None:
    """The criterion is a constant of the RUN, not of the level: `solve` applies
    it at level 1 and at depth alike, which is what "PR at depth if chosen in
    input" means."""
    base, logb = _correlated(seed=61)
    for crit in sorted(CRITERIA):
        t = ParticipationTarget(0.3, criterion=crit, solver="grid")
        direct = solve_gamma(base, logb, 0.3, t.gamma_max, t.n_grid, crit)
        assert t.solve(base, logb) == direct
        # and the same under the default solver, against ITS free function:
        # the criterion reaches the pool either way, and which solver is in
        # force is the other constant of the run
        tm = ParticipationTarget(0.3, criterion=crit)
        assert tm.solver == "moments"
        assert tm.solve(base, logb) == solve_gamma_moments(
            base, logb, 0.3, tm.gamma_max, crit
        )
        # and it is frozen, so nothing can retune it mid-run: a criterion that
        # could be reassigned between levels would make the kernel a function
        # of chain history, which is rule 1 with a different knob
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.criterion = "participation"  # type: ignore[misc]


# --------------------------------------------------------------------------
# The closed form. `solve_gamma_moments` replaces the grid search, which was
# measured at 13% of move time on a production configuration (hard cores,
# k = 200, full_repacking): 17 grid points x k trials x the levels a repacking
# move walks came to roughly 1450 exponentials per move. The expansion answers
# the same question with two weighted moments and one square root.
#
# What these tests hold it to is what the grid was held to above -- determinism,
# permutation invariance bit for bit, totality -- and NOT accuracy beyond
# landing near the target. Detailed balance does not care which deterministic
# function of the pool gamma is; it cares that it is one.
# --------------------------------------------------------------------------


def test_the_closed_form_is_permutation_invariant_bit_for_bit() -> None:
    """The engine permutes the pool, so the same physical set of trials arrives
    in different orders. `np.sum` is pairwise and order-dependent, which is what
    `_exact_sums` exists to fix, by reducing with `math.fsum` -- exactly the
    float, not merely close, or the kernel depends on the RNG draw that produced
    the permutation. The pool here straddles `_VECTOR_FROM`, so both branches of
    that helper are exercised by the sweep over sizes below."""
    base, logb = _correlated(seed=31)
    first = solve_gamma_moments(base, logb, 0.4, GAMMA_MAX)
    rng = np.random.default_rng(99)
    for _ in range(25):
        perm = rng.permutation(len(base))
        assert solve_gamma_moments(base[perm], logb[perm], 0.4, GAMMA_MAX) == first


def test_the_closed_form_is_stateless_across_repeated_calls() -> None:
    """No cache, no warm start, no drift. A gamma that depended on what was
    solved before would turn a fixed kernel into an adaptive one."""
    base, logb = _correlated(seed=32)
    other, other_logb = _anticorrelated(seed=33)
    first = solve_gamma_moments(base, logb, 0.35, GAMMA_MAX)
    for _ in range(5):
        solve_gamma_moments(other, other_logb, 0.9, GAMMA_MAX)
        solve_gamma_moments(base, logb, 0.05, GAMMA_MAX)
        assert solve_gamma_moments(base, logb, 0.35, GAMMA_MAX) == first


def test_the_closed_form_ignores_the_grid_size() -> None:
    """There is no grid, and this is the test that says so rather than a comment
    claiming it. `n_grid` is the grid solver's only cost parameter, so a
    `ParticipationTarget` whose n_grid changes and whose gamma does not is the
    observable difference between the two solvers."""
    rng = np.random.default_rng(34)
    logb = rng.normal(0.0, 0.8, 40)
    base = np.ones(40)
    answers = {
        ParticipationTarget(0.7, n_grid=n).solve(base, logb) for n in (2, 17, 129)
    }
    assert len(answers) == 1
    # while the grid's answer does move with it, which is what makes the check
    # above meaningful
    grid_answers = {
        ParticipationTarget(0.7, n_grid=n, solver="grid").solve(base, logb)
        for n in (2, 17, 129)
    }
    assert len(grid_answers) > 1


def test_a_flat_base_reduces_to_one_variance_and_a_square_root() -> None:
    """Every admissible hard-core trial has the same gamma-free weight, so level
    1 of the production configuration lands in this case. With a_i constant, P
    and Q coincide, the linear coefficient vanishes and the whole solver is

        gamma = sqrt( log(1/f) / Var(log b) )

    which is worth pinning because it is the form the hot path actually runs.
    """
    rng = np.random.default_rng(35)
    logb = rng.normal(0.0, 0.8, 40)
    base = np.ones(40)
    var = float(np.var(logb))
    for f in (0.9, 0.6, 0.3):
        expect = math.sqrt(math.log(1.0 / f) / var)
        got = solve_gamma_moments(base, logb, f, GAMMA_MAX)
        assert got == pytest.approx(expect, rel=1e-10)


def test_the_closed_form_lands_as_near_the_target_as_the_grid() -> None:
    """The claim that justifies replacing the grid: across random pools the
    achieved fraction is no further from the target than the grid's, on average.
    Not exactly equal -- they are different functions -- and the expansion is
    often CLOSER, because the grid can only return one of its 17 points."""
    rng = np.random.default_rng(36)
    for crit in sorted(CRITERIA):
        for f in (0.9, 0.75, 0.5):
            miss_grid = []
            miss_closed = []
            for _ in range(60):
                n = int(rng.integers(5, 60))
                log_base = rng.normal(0.0, 1.0, n)
                base = np.exp(log_base)
                logb = log_base + rng.normal(0.0, 0.4, n)
                for solver, misses in (
                    (solve_gamma, miss_grid),
                    (solve_gamma_moments, miss_closed),
                ):
                    if solver is solve_gamma:
                        g = solve_gamma(base, logb, f, GAMMA_MAX, N_GRID, crit)
                    else:
                        g = solve_gamma_moments(base, logb, f, GAMMA_MAX, crit)
                    w = base * np.exp(g * logb)
                    misses.append(abs(spread_fraction(w, crit) - f))
            assert np.mean(miss_closed) <= np.mean(miss_grid) + 0.05


def test_the_closed_form_stays_inside_the_bracket_on_every_pool() -> None:
    """TOTALITY. Every pool the sampler can hand it gets a finite answer in
    [0, gamma_max], including the ones that make the quadratic degenerate: a
    pool of one, an annihilated pool, a bias with no spread, a target nothing
    can reach."""
    rng = np.random.default_rng(37)
    for crit in sorted(CRITERIA):
        for _ in range(200):
            n = int(rng.integers(1, 40))
            base = np.abs(rng.normal(1.0, 1.0, n))
            base[rng.random(n) < 0.2] = 0.0
            logb = rng.normal(0.0, rng.choice([0.0, 0.5, 5.0]), n)
            f = float(rng.uniform(0.01, 1.0))
            g = solve_gamma_moments(base, logb, f, GAMMA_MAX, crit)
            assert np.isfinite(g)
            assert 0.0 <= g <= GAMMA_MAX


@pytest.mark.parametrize(
    "base, logb",
    [
        (np.array([]), np.array([])),
        (np.array([2.0]), np.array([1.0])),
        (np.zeros(6), np.ones(6)),
        (np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 3.0])),
        (np.ones(8), np.zeros(8)),
    ],
    ids=["empty", "single", "annihilated", "one-survivor", "flat-bias"],
)
def test_degenerate_pools_solve_to_zero_rather_than_raising(base, logb) -> None:
    """Each of these occurs inside a move. An empty pool and a pool of one have
    no spread to tune; an annihilated pool is about to be rejected anyway; a
    flat bias cannot change the spread at any exponent, so the least biased
    answer is the honest one."""
    for crit in sorted(CRITERIA):
        assert solve_gamma_moments(base, logb, 0.4, GAMMA_MAX, crit) == 0.0


def test_the_closed_form_refuses_the_same_inputs_the_grid_refuses() -> None:
    """One solver replacing another must reject the same nonsense, or a params
    file that was refused becomes silently accepted."""
    base, logb = _correlated(seed=38)
    with pytest.raises(ValueError, match="pr_target"):
        solve_gamma_moments(base, logb, 0.0, GAMMA_MAX)
    with pytest.raises(ValueError, match="pr_target"):
        solve_gamma_moments(base, logb, 1.5, GAMMA_MAX)
    with pytest.raises(ValueError, match="gamma_max"):
        solve_gamma_moments(base, logb, 0.4, 0.0)
    with pytest.raises(ValueError, match="unknown criterion"):
        solve_gamma_moments(base, logb, 0.4, GAMMA_MAX, "entropy")
    with pytest.raises(ValueError, match="row-aligned"):
        solve_gamma_moments(np.ones(5), np.zeros(4), 0.4, GAMMA_MAX)


def test_an_unknown_solver_is_refused_on_the_target_object() -> None:
    """The config refuses it first (`sim_setup._GAMMA_SOLVERS`), but the object
    is constructible from a pilot script and must not accept a typo by falling
    through to the grid."""
    with pytest.raises(ValueError, match="unknown solver"):
        ParticipationTarget(0.3, solver="closed_form")


def test_the_default_solver_is_the_cheap_one() -> None:
    """Pinned, because the whole point of the field is which one production
    runs. A default of "grid" would put 13% of move cost back without any
    params file changing."""
    assert ParticipationTarget(0.3).solver == "moments"
    assert SOLVERS == ("moments", "grid")


@pytest.mark.parametrize("n", [3, 23, 24, 25, 200])
def test_the_closed_form_is_permutation_invariant_at_every_pool_size(n) -> None:
    """`_exact_sums` switches between a list comprehension and numpy products at
    `_VECTOR_FROM`, so invariance has to hold on both sides of the crossover and
    the two branches have to agree with each other. `math.fsum` is correctly
    rounded, so they do -- not approximately, identically."""
    rng = np.random.default_rng(1000 + n)
    log_base = rng.normal(0.0, 1.0, n)
    base = np.exp(log_base)
    logb = log_base + rng.normal(0.0, 0.4, n)
    for crit in sorted(CRITERIA):
        first = solve_gamma_moments(base, logb, 0.7, GAMMA_MAX, crit)
        for _ in range(12):
            perm = rng.permutation(n)
            got = solve_gamma_moments(base[perm], logb[perm], 0.7, GAMMA_MAX, crit)
            assert got == first


def test_the_two_summation_branches_give_the_same_float() -> None:
    """The crossover is a cost switch and must not be an answer switch."""
    from gcmc.sampling import participation as P

    rng = np.random.default_rng(1234)
    log_base = rng.normal(0.0, 1.0, 40)
    base, logb = np.exp(log_base), log_base + rng.normal(0.0, 0.4, 40)
    saved = P._VECTOR_FROM
    try:
        P._VECTOR_FROM = 0  # numpy products
        vector = solve_gamma_moments(base, logb, 0.7, GAMMA_MAX)
        P._VECTOR_FROM = 10**9  # list comprehension
        listed = solve_gamma_moments(base, logb, 0.7, GAMMA_MAX)
    finally:
        P._VECTOR_FROM = saved
    assert vector == listed


def test_a_pool_spanning_hundreds_of_orders_of_magnitude_does_not_overflow() -> None:
    """The one place this solver could overflow is `a * a`, and the rescale by
    the largest weight is what stops it. Boltzmann factors in a dense
    configuration really do span this much."""
    rng = np.random.default_rng(77)
    base = np.exp(rng.uniform(-300.0, 300.0, 30))
    logb = rng.normal(0.0, 1.0, 30)
    for crit in sorted(CRITERIA):
        g = solve_gamma_moments(base, logb, 0.5, GAMMA_MAX, crit)
        assert np.isfinite(g)
        assert 0.0 <= g <= GAMMA_MAX


def test_a_common_scale_on_the_base_cannot_move_the_answer() -> None:
    """Both criteria are scale-invariant in the gamma-free factor, so the
    rescale the overflow guard applies is free. If it were not, the solver would
    return a different exponent for the same physical pool at a different
    reference energy."""
    base, logb = _correlated(seed=78)
    ref = solve_gamma_moments(base, logb, 0.6, GAMMA_MAX)
    for c in (1e-8, 0.5, 1e9):
        assert solve_gamma_moments(base * c, logb, 0.6, GAMMA_MAX) == pytest.approx(
            ref, rel=1e-9
        )
