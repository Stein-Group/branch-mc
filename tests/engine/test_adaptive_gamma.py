"""Adaptive gamma in the two Rosenbluth passes: the gates that do not need a run.

The correctness of this feature rests on a handful of structural claims, and
what makes them worth testing separately from the statistics is that every one
of them fails SILENTLY. A gamma cached across passes, a reverse gamma solved
over the forward pool, two exponents at one reverse level: each produces a chain
that runs, reports plausible numbers, and samples the wrong distribution. The
long-run gates (G5, G6) catch them eventually and expensively. These catch them
in a second.

  G1 in miniature  -- `gamma_mode` absent changes nothing, including the level-1
                      weights, which is the one place the pool build differs.
  G2 (see also tests/unit/test_participation.py) -- the solve is deterministic.
  G3               -- the reverse pass is reproducible, and its gammas are NOT
                      the forward pass's.
  G7 in miniature  -- an unreachable target clamps to gamma = 0 and the run is
                      the bias-off run.
  the level cap    -- gone, which is what removing the nine fitted numbers buys.

The pools here are built directly rather than driven through `advance`, so a
failure names a pass and a level instead of a window summary.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc.forcefield.auxiliary import LJAux3D
from gcmc.proposals.trials import TrialSet
from gcmc.sampling.base import EnergyWeigher
from gcmc.sampling.participation import ParticipationTarget, gamma_grid
from gcmc.sampling.rosenbluth import select_forward, select_reverse

G3_CSV = "data/g3_analysis_data.csv"
SIDE, UPPER_CUTOFF, DIAMETER, BETA = 10.0, 2.5, 1.0, 1.0 / 0.85
N_TRIALS = 30
R_SQ = DIAMETER**2
#: the nine fitted 3D numbers, keyed as the engine keys them
FITTED = {1: 0.2903, 2: 0.1030, 3: 0.0332, 4: 0.0193, 5: 0.0037, 6: 0.0042}
FITTED.update({7: 0.0, 8: 0.0, 9: 0.0})


def _potential() -> LJAux3D:
    return LJAux3D.build(
        beta=BETA,
        upper_cutoff=UPPER_CUTOFF,
        diameter=DIAMETER,
        side=SIDE,
        n_trials=N_TRIALS,
        g3_csv=G3_CSV,
        min_g3=0.03,
    )


def _cloud(n: int, seed: int, *, split: bool) -> TrialSet:
    """A trial cloud in a liquid environment, weighted either the frozen way
    (bias folded in at gamma_1) or the adaptive way (bias unexponentiated in
    `log_bias`), exactly as `_weighted_trials` does it in each mode."""
    rng = np.random.default_rng(seed)
    step = SIDE / 8
    grid = np.stack(
        np.meshgrid(*(np.arange(8) * step,) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    env = (grid + rng.uniform(-0.15, 0.15, grid.shape)) % SIDE
    tree = cKDTree(env, boxsize=SIDE)
    cand = (grid + step / 2 + rng.uniform(-0.1, 0.1, grid.shape)) % SIDE
    cand = cand[rng.permutation(len(cand))]
    near, _ = tree.query(cand, k=1)
    coords = cand[near > 0.95][:n]
    assert len(coords) == n
    dists, idxs = tree.query(coords, k=10)
    pot = _potential()
    if split:
        base, log_bias, ref = pot.weight_bulk_split(
            dists, coords=coords, idxs=idxs, kdtree=tree
        )
        return TrialSet(coords, base, ref_neighbor_coord=ref, log_bias=log_bias)
    w, ref = pot.weight_bulk(
        dists, coords=coords, idxs=idxs, kdtree=tree, gamma=FITTED[1]
    )
    return TrialSet(coords, w, ref_neighbor_coord=ref)


def _weigher() -> EnergyWeigher:
    return EnergyWeigher(_potential())


def _rand_stream(seed: int):
    rng = np.random.default_rng(seed)
    return lambda: float(rng.random())


# ---- G1 in miniature ----------------------------------------------------------


def test_no_target_leaves_the_frozen_path_untouched() -> None:
    """Two calls with `target=None`, one of them on a pool that carries a
    populated `log_bias` column. The column must not be read: the frozen path
    folds gamma_1 in at pool-build time and reading it here would apply the bias
    twice."""
    weigher = _weigher()
    plain = _cloud(24, 3, split=False)
    with_column = _cloud(24, 3, split=False)
    with_column.log_bias = _cloud(24, 3, split=True).log_bias
    a = select_forward(
        plain,
        3,
        weigher,
        rand=_rand_stream(1),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
    )
    b = select_forward(
        with_column,
        3,
        weigher,
        rand=_rand_stream(1),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
    )
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[0].coords, b[0].coords)
    assert np.array_equal(a[0].energy_weight, b[0].energy_weight)


def test_the_frozen_path_records_no_gammas() -> None:
    """The out-parameter stays empty with no target, so a caller cannot read a
    solved exponent off a run that never solved one."""
    sink: list[float] = []
    select_forward(
        _cloud(20, 4, split=False),
        2,
        _weigher(),
        rand=_rand_stream(2),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        solved_gammas=sink,
    )
    assert sink == []


# ---- the solve happens, once per level ---------------------------------------


@pytest.mark.parametrize("n_atoms", [1, 2, 3, 5])
def test_one_gamma_is_solved_per_level(n_atoms: int) -> None:
    """Level 1 plus one per deeper level. Counting them is how a missing level-1
    solve (the bias silently off at the first level) would show up."""
    sink: list[float] = []
    chosen, _ = select_forward(
        _cloud(28, 5, split=True),
        n_atoms,
        _weigher(),
        rand=_rand_stream(3),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=ParticipationTarget(0.3),
        solved_gammas=sink,
    )
    assert chosen is not None
    assert len(sink) == n_atoms


class _Landmine(dict):
    """A gamma schedule that cannot be read. Reading one is the cap."""

    def __getitem__(self, key: object) -> float:
        raise AssertionError(f"the fitted schedule was consulted at level {key!r}")


def test_the_level_nine_cap_is_gone_because_the_table_is_never_read() -> None:
    """The point of replacing the nine fitted numbers, asserted structurally
    rather than by comparing values.

    The cap is not a separate rule to remove; it is what `gamma_schedule[min(i +
    1, 9)]` DOES when there is no tenth number. A twelve-level move on the
    adaptive path completes with a schedule that raises on any lookup, which
    says the table -- and therefore the cap -- is unreachable. Asserting it this
    way rather than by showing levels 10 to 12 solve to different numbers
    matters, because at this state point they all solve to 0.0 and a
    value-comparison test would fail while the code was right.
    """
    sink: list[float] = []
    chosen, sums = select_forward(
        _cloud(40, 6, split=True),
        12,
        _weigher(),
        rand=_rand_stream(4),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=_Landmine(),
        target=ParticipationTarget(0.3),
        solved_gammas=sink,
    )
    assert chosen is not None and sums is not None
    assert len(sink) == 12


def test_the_frozen_path_does_read_the_table_and_does_cap() -> None:
    """The control for the test above: with no target the schedule IS read, so
    the same landmine raises. Without this, a solver that quietly skipped the
    refresh entirely would pass the previous test."""
    with pytest.raises(AssertionError, match="fitted schedule was consulted"):
        select_forward(
            _cloud(40, 6, split=False),
            12,
            _weigher(),
            rand=_rand_stream(4),
            side=SIDE,
            r_squared=R_SQ,
            gamma_schedule=_Landmine(),
        )


# ---- G3: the reverse pass ----------------------------------------------------


def _reverse_pools(seed: int = 7, n_remove: int = 3, n_trial: int = 20):
    """(trial cloud, removed members), weighted the adaptive way. Two disjoint
    clouds, as the engine's are: the trials come from the insertion region and
    the members from the cavity being emptied."""
    trials = _cloud(n_trial, seed, split=True)
    members = _cloud(n_remove, seed + 100, split=True)
    return trials, members


def test_the_reverse_pass_is_reproducible_including_its_gammas() -> None:
    """G3, first half. The reverse pass has no RNG, so two calls on the same
    stored inputs must return the same arrays AND solve the same exponents. The
    inputs are copied because the pass mutates them, which is itself the thing
    that could make a second call differ."""
    target = ParticipationTarget(0.3)
    runs = []
    for _ in range(2):
        trials, members = _reverse_pools()
        sink: list[float] = []
        out = select_reverse(
            trials.copy(),
            members.copy(),
            len(members),
            _weigher(),
            side=SIDE,
            r_squared=R_SQ,
            gamma_schedule=FITTED,
            target=target,
            solved_gammas=sink,
        )
        runs.append((out, sink))
    (sum_a, e_a), g_a = runs[0]
    (sum_b, e_b), g_b = runs[1]
    assert np.array_equal(sum_a, sum_b)
    assert np.array_equal(e_a, e_b)
    assert g_a == g_b
    assert len(g_a) == 3


def test_the_reverse_gammas_are_not_the_forward_gammas() -> None:
    """G3, second half, and the one that catches a leak. The two passes see
    different level pools by construction -- the forward pool is the permuted
    merge of trials and members, the reverse level-1 pool is the members plus
    the surviving trials -- so their exponents must differ at some level. If
    they are always equal, gamma is being carried from one pass into the
    other."""
    target = ParticipationTarget(0.3)
    trials, members = _reverse_pools()
    fwd_pool = trials.copy().merge(members.copy())
    perm = np.random.default_rng(0).permutation(len(fwd_pool))
    fwd_pool = fwd_pool[perm]

    g_fwd: list[float] = []
    select_forward(
        fwd_pool,
        len(members),
        _weigher(),
        rand=_rand_stream(11),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=target,
        solved_gammas=g_fwd,
    )
    g_rev: list[float] = []
    select_reverse(
        trials.copy(),
        members.copy(),
        len(members),
        _weigher(),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=target,
        solved_gammas=g_rev,
    )
    assert g_fwd != g_rev, (
        "forward and reverse solved identical exponents at every level; the "
        "two passes have different level pools, so this is a leak"
    )


def test_the_reverse_pass_solves_one_gamma_per_level_over_the_union() -> None:
    """One exponent per level, not two. The reverse level pool is the union of
    the remaining members and the surviving trials, and solving the two halves
    separately would give two exponents at one level -- which is a different
    method, not a variant of this one. Counting the solves is what says it is
    one."""
    sink: list[float] = []
    trials, members = _reverse_pools(n_remove=4)
    select_reverse(
        trials.copy(),
        members.copy(),
        4,
        _weigher(),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=ParticipationTarget(0.3),
        solved_gammas=sink,
    )
    assert len(sink) == 4


def test_the_reverse_pass_solves_a_gamma_with_no_trial_cloud_left() -> None:
    """The fast path, where the whole shadow cloud overlapped. The members still
    carry an unexponentiated bias, so level 1 still has an exponent to solve --
    returning early without solving it would leave the deletion weight computed
    with the bias off while the insertion weight had it on."""
    _, members = _reverse_pools(n_remove=3)
    sink: list[float] = []
    sum_arr, e_arr = select_reverse(
        None,
        members.copy(),
        3,
        _weigher(),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=ParticipationTarget(0.3),
        solved_gammas=sink,
    )
    assert len(sink) == 1
    assert np.all(np.isfinite(sum_arr)) and np.all(np.isfinite(e_arr))
    # and the bias actually reached the weights: with gamma = 0 the members
    # would carry their bare Boltzmann factors
    bare = _reverse_pools(n_remove=3)[1]
    assert sink[0] == 0.0 or not np.array_equal(e_arr, bare.energy_weight[:3])


# ---- G7 in miniature ----------------------------------------------------------


def test_a_clamped_level_is_the_bias_off_level_exactly() -> None:
    """G7's mechanism, at the level it applies to.

    G7 as the plan states it -- "force f_PR unreachably high so the solver
    clamps to gamma = 0 everywhere" -- is NOT reachable through `pr_target`, and
    the reason is the non-monotonicity the plan itself documents two sections
    earlier. A high target clamps only where PR falls with gamma. Where the bias
    is anti-correlated with the Boltzmann factor, PR RISES, and a target of 1.0
    is best approached at the TOP of the bracket: measured here, `pr_target =
    1.0` solves [0.0, 3.0, 1.76] across three levels. No single number turns the
    bias off everywhere, because the fraction at gamma = 0 is a property of each
    pool and not of the target.

    What can be asserted is the mechanism: a level that solves to exactly 0.0
    carries no bias at all, so its weights are the bare Boltzmann factors the
    pool was built with. That is what the clamp counter in the window summary
    means, and it is the whole content of the no-op claim.
    """
    weigher = _weigher()
    pool = _cloud(28, 9, split=True)
    base_before = pool.energy_weight.copy()
    sink: list[float] = []
    # level 1 clamps at this target (the pool's unbiased participation is the
    # closest point on the grid); the deeper levels are free to do otherwise
    select_forward(
        pool,
        1,
        weigher,
        rand=_rand_stream(21),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=ParticipationTarget(1.0),
        solved_gammas=sink,
    )
    assert sink == [0.0], f"level 1 did not clamp at this target: {sink}"
    assert np.array_equal(pool.energy_weight, base_before)


def test_a_solved_level_is_not_the_bias_off_level() -> None:
    """The control. Without it the test above would pass on a solver that had
    stopped applying the bias altogether."""
    weigher = _weigher()
    pool = _cloud(28, 9, split=True)
    base_before = pool.energy_weight.copy()
    sink: list[float] = []
    select_forward(
        pool,
        1,
        weigher,
        rand=_rand_stream(21),
        side=SIDE,
        r_squared=R_SQ,
        gamma_schedule=FITTED,
        target=ParticipationTarget(0.15),
        solved_gammas=sink,
    )
    assert sink[0] > 0.0
    assert not np.array_equal(pool.energy_weight, base_before)


def test_the_solver_choice_reaches_the_level_loop() -> None:
    """The target object is built once per window and threaded to both passes,
    so a solver that were read anywhere else would be read at one call site and
    not the other. Two runs of the same pool under the two solvers must differ
    in the gammas and agree on nothing else being different."""
    gammas: dict[str, list[float]] = {}
    for solver in ("moments", "grid"):
        sink: list[float] = []
        chosen, _ = select_forward(
            _cloud(40, 11, split=True),
            4,
            _weigher(),
            rand=_rand_stream(3),
            side=SIDE,
            r_squared=R_SQ,
            gamma_schedule=FITTED,
            target=ParticipationTarget(0.4, solver=solver),
            solved_gammas=sink,
        )
        assert chosen is not None
        assert len(sink) == 4
        gammas[solver] = sink
    # the grid can only return a grid point; the closed form is continuous, so
    # a level that solved to anything other than 0 distinguishes them
    grid_points = set(gamma_grid(3.0, 17))
    assert all(g in grid_points for g in gammas["grid"])
    assert any(g not in grid_points for g in gammas["moments"])


def test_both_solvers_are_reproducible_across_repeated_passes() -> None:
    """Determinism at the level the engine sees it: the same window twice gives
    the same gammas, whichever solver is in force."""
    for solver in ("moments", "grid"):
        runs = []
        for _ in range(2):
            sink: list[float] = []
            select_forward(
                _cloud(30, 12, split=True),
                3,
                _weigher(),
                rand=_rand_stream(5),
                side=SIDE,
                r_squared=R_SQ,
                gamma_schedule=FITTED,
                target=ParticipationTarget(0.45, solver=solver),
                solved_gammas=sink,
            )
            runs.append(sink)
        assert runs[0] == runs[1]
