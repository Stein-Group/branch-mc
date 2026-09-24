"""The selection weight, taken apart at the seam gamma sits on.

`weight_bulk` and `pairwise` return `w_i * b_i**gamma` as one number: the
Boltzmann factor the acceptance cancels against the selection probability,
multiplied by the tabulated bias the acceptance divides back out. One number is
everything a gamma read from a schedule needs. A gamma SOLVED per level needs
the factors apart -- it has to see `w_i` and `log b_i` over the level pool
before any gamma exists -- so stage 2 of docs/decisions/adaptive_gamma_plan.md
adds a parallel pair of accessors that return them and leaves the combined path
untouched.

Untouched is the load-bearing word, and it is why the central test here is
composition rather than a table of expected numbers: `base * exp(gamma *
log_g3)` must reproduce `weight_bulk(gamma)` to round-off, at the gammas the
fitted 3D schedule actually uses. If it does, the split cannot be describing a
different weight than the sampler applies, which is the only way this refactor
can be wrong quietly.

The other half of the file is the no-reference rows. A trial whose nearest
neighbour is undefined carries bias 1 and therefore `log_g3 = 0`. Zero is not
"nothing to contribute": the participation ratio is taken over the level pool,
and that trial is in the pool. Dropping it makes the solver read a smaller pool
than the sampler selects from -- invisible at low density, biased at high, which
is the failure shape the plan singles out as worth a targeted assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc._types import FloatArray
from gcmc.forcefield.auxiliary import HardCoreAux2D, HardCoreAux3D, LJAux3D
from gcmc.forcefield.lennard_jones import LennardJones
from gcmc.proposals.trials import TrialSet
from gcmc.sampling.base import EnergyWeigher

G3_CSV = "data/g3_analysis_data.csv"
SIDE, UPPER_CUTOFF, DIAMETER, BETA = 10.0, 2.5, 1.0, 1.0 / 1.5
N_TRIALS = 30

# The gammas the frozen `_GAMMA_AUX_3D` schedule actually visits, plus 1.0 (the
# branch that skips the exponent entirely) and 0.0 (no bias at all). The
# composition has to hold at every one of them: a sign or floor error that
# cancels at gamma = 1 does not cancel at 0.0332.
GAMMAS = (0.0, 0.0332, 0.1030, 0.2903, 1.0, 2.0)


# Floors to sweep. One value cannot test the floor at all -- at any single floor
# a split that had frozen its own constant can agree by accident. The range
# matters too: 0.5 sits above the lowest g3 this geometry reaches, so the
# clamped part of the table actually enters the weights.
FLOORS = (0.03, 0.2, 0.5, 0.9)


def _liquid(seed: int = 3, n_side: int = 8, k: int = 10):
    """A jittered simple-cubic environment at rho ~ 0.5 and a cloud of trials
    in it. A liquid, not a gas: at low density every neighbour falls past the
    table's r range, g3 returns its out-of-range 1, every `log_g3` is 0, and
    every test below passes without evaluating the interpolator once."""
    rng = np.random.default_rng(seed)
    step = SIDE / n_side
    grid = np.stack(
        np.meshgrid(*(np.arange(n_side) * step,) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    env = (grid + rng.uniform(-0.15, 0.15, grid.shape)) % SIDE
    tree = cKDTree(env, boxsize=SIDE)
    # trials at the cube centres, jittered: in the interstices rather than on
    # the cores. A trial sitting on an environment particle has a Boltzmann
    # factor that underflows, and 0 == 0 is not a test of anything.
    cand = (grid + step / 2 + rng.uniform(-0.1, 0.1, grid.shape)) % SIDE
    cand = cand[rng.permutation(len(cand))]
    near, _ = tree.query(cand, k=1)
    trials = cand[near > 0.95][:N_TRIALS]
    assert len(trials) == N_TRIALS
    # k = 10, not 6: with six neighbours every g3 this geometry reaches is above
    # every floor swept below, so the clamped part of the table never enters a
    # weight and nothing here depends on the floor at all.
    dists, idxs = tree.query(trials, k=k)
    return trials, dists, idxs, tree


def _aux(min_g3: float | None = 0.03) -> HardCoreAux3D:
    return HardCoreAux3D.build(
        diameter=DIAMETER,
        upper_cutoff=UPPER_CUTOFF,
        side=SIDE,
        n_trials=N_TRIALS,
        g3_csv=G3_CSV,
        min_g3=min_g3,
    )


def _ljaux(min_g3: float | None = 0.03) -> LJAux3D:
    return LJAux3D.build(
        beta=BETA,
        upper_cutoff=UPPER_CUTOFF,
        diameter=DIAMETER,
        side=SIDE,
        n_trials=N_TRIALS,
        g3_csv=G3_CSV,
        min_g3=min_g3,
    )


def _pair_geometry(seed: int = 11, n: int = 40):
    """One deeper-level refresh: `n` survivors, their carried reference
    neighbours, and one just-chosen trial. Squared distances, as every depth > 0
    call site passes."""
    rng = np.random.default_rng(seed)
    chosen = np.array([5.0, 5.0, 5.0])
    survivors = chosen + rng.uniform(-2.2, 2.2, (n, 3))
    refs = survivors + rng.uniform(-1.8, 1.8, (n, 3))
    diff = survivors - chosen
    diff -= SIDE * np.round(diff / SIDE)
    sq = np.sum(diff**2, axis=1)
    keep = sq > DIAMETER**2  # overlapping survivors are filtered upstream
    return survivors[keep], refs[keep], chosen, sq[keep]


# ---- composition: the split describes the same weight -------------------------


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_bulk_weight_composes_back_to_the_combined_one(
    gamma: float,
) -> None:
    """Gate S2, first level, LJ + pmf. This is the configuration adaptive gamma
    is for, and the one where the two factors are genuinely different physics."""
    trials, dists, idxs, tree = _liquid()
    pot = _ljaux()
    combined, ref = pot.weight_bulk(
        dists, coords=trials, idxs=idxs, kdtree=tree, gamma=gamma
    )
    base, log_g3, ref_split = pot.weight_bulk_split(
        dists, coords=trials, idxs=idxs, kdtree=tree
    )
    np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)
    assert np.array_equal(ref, ref_split)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_bulk_weight_composes_back_for_the_bare_auxiliary(
    gamma: float,
) -> None:
    """Same for `aux3d` alone, where `base` is ones because the potential IS
    the bias. Kept separate so a bug in the LJ half cannot mask one in the aux
    half."""
    trials, dists, idxs, tree = _liquid()
    pot = _aux()
    combined, _ = pot.weight_bulk(
        dists, coords=trials, idxs=idxs, kdtree=tree, gamma=gamma
    )
    base, log_g3, _ = pot.weight_bulk_split(
        dists, coords=trials, idxs=idxs, kdtree=tree
    )
    assert np.array_equal(base, np.ones(len(trials)))
    np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_pair_refresh_composes_back_to_the_combined_one(
    gamma: float,
) -> None:
    """Gate S2, deeper levels. `pairwise`, not `weight_pair`, because that is
    what the weigher calls and it carries squared distances."""
    survivors, refs, chosen, sq = _pair_geometry()
    pot = _ljaux()
    kw = {
        "trial_coords": survivors,
        "ref_coords": refs,
        "chosen_coord": chosen,
    }
    combined = pot.pairwise(sq, gamma, **kw)
    base, log_g3 = pot.pairwise_split(sq, **kw)
    np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_pair_refresh_composes_back_for_the_bare_auxiliary(
    gamma: float,
) -> None:
    survivors, refs, chosen, sq = _pair_geometry()
    pot = _aux()
    kw = {
        "trial_coords": survivors,
        "ref_coords": refs,
        "chosen_coord": chosen,
    }
    combined = pot.pairwise(sq, gamma, **kw)
    base, log_g3 = pot.pairwise_split(sq, **kw)
    assert np.array_equal(base, np.ones(len(survivors)))
    np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)


def test_the_composition_is_tested_on_a_pool_that_is_actually_biased() -> None:
    """The guard on every test above. If the geometry were dilute enough that
    every neighbour fell past the table, `log_g3` would be all zeros and the
    compositions would hold trivially."""
    trials, dists, idxs, tree = _liquid()
    _, log_g3, _ = _ljaux().weight_bulk_split(
        dists, coords=trials, idxs=idxs, kdtree=tree
    )
    assert np.count_nonzero(log_g3) > len(trials) // 2
    survivors, refs, chosen, sq = _pair_geometry()
    _, log_pair = _ljaux().pairwise_split(
        sq, trial_coords=survivors, ref_coords=refs, chosen_coord=chosen
    )
    assert np.count_nonzero(log_pair) > len(survivors) // 2


def test_the_split_reads_the_same_floor_as_the_combined_path() -> None:
    """Sweep the floor and compose at every value. Stage 1's whole point is that
    the floor is a config constant rather than a function of k, and a split that
    had frozen its own copy of the old `1/n_trials` would still compose
    correctly at whichever floor happened not to bite."""
    trials, dists, idxs, tree = _liquid()
    survivors, refs, chosen, sq = _pair_geometry()
    kw = {
        "trial_coords": survivors,
        "ref_coords": refs,
        "chosen_coord": chosen,
    }
    gamma = 0.2903
    for floor in FLOORS:
        pot = _ljaux(floor)
        combined, _ = pot.weight_bulk(
            dists, coords=trials, idxs=idxs, kdtree=tree, gamma=gamma
        )
        base, log_g3, _ = pot.weight_bulk_split(
            dists, coords=trials, idxs=idxs, kdtree=tree
        )
        np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)
        pair_combined = pot.pairwise(sq, gamma, **kw)
        pair_base, pair_log = pot.pairwise_split(sq, **kw)
        np.testing.assert_allclose(
            pair_base * np.exp(gamma * pair_log), pair_combined, rtol=1e-14
        )


def test_the_floor_sweep_is_not_vacuous() -> None:
    """The guard on the sweep above: the floor has to actually reach the split's
    `log_g3`, at both levels, or that test proves nothing about the floor."""
    trials, dists, idxs, tree = _liquid()
    survivors, refs, chosen, sq = _pair_geometry()
    bulk, pair = [], []
    for floor in FLOORS:
        pot = _ljaux(floor)
        bulk.append(
            pot.weight_bulk_split(dists, coords=trials, idxs=idxs, kdtree=tree)[1]
        )
        pair.append(
            pot.pairwise_split(
                sq, trial_coords=survivors, ref_coords=refs, chosen_coord=chosen
            )[1]
        )
    for level in (bulk, pair):
        assert not any(np.array_equal(level[0], other) for other in level[1:])
    # and the swept range has to straddle the values this geometry produces, or
    # the floor is only ever reshaping parts of the table nobody reads. The low
    # end is inclusive: the reader returns tabulated entries verbatim, so where
    # the table was floored the geometry produces the floor EXACTLY. Under the
    # interpolator this was a strict inequality, because blending a floored node
    # with an unfloored neighbour always landed above the floor -- a difference
    # worth noticing rather than papering over, since it says the floor is now
    # reachable rather than approached.
    g3, _ = _aux(FLOORS[0])._g3_grid_bulk(dists, trials, idxs, tree)
    assert FLOORS[0] <= g3.min() < max(FLOORS)


# ---- the floor is what keeps the log finite -----------------------------------


def test_the_floor_keeps_every_log_finite() -> None:
    """Why `resolve_min_g3` refuses a floor of 0: the table contains exact
    zeros, and the solver reads the LOG of the floored value. A zero floor puts
    -inf in the array the participation ratio reduces over."""
    trials, dists, idxs, tree = _liquid()
    for floor in (None, 0.03, 1.0):
        _, log_g3, _ = _aux(floor).weight_bulk_split(
            dists, coords=trials, idxs=idxs, kdtree=tree
        )
        assert np.isfinite(log_g3).all()


# ---- the no-reference rows stay in the array ---------------------------------


def _mixed_reference_rows():
    """The same refresh, with a third of the survivors' reference neighbours
    marked undefined -- the NaN `weight_bulk` writes when a trial had no
    neighbour to fix a direction from."""
    survivors, refs, chosen, sq = _pair_geometry()
    refs = refs.copy()
    no_ref = np.zeros(len(survivors), dtype=bool)
    no_ref[::3] = True
    refs[no_ref] = np.nan
    return survivors, refs, chosen, sq, no_ref


def test_an_unreferenced_row_contributes_zero_and_is_not_dropped() -> None:
    """The targeted assertion stage 2 asks for. Three claims, and the third is
    the one that bites: the array is the full length of the pool."""
    survivors, refs, chosen, sq, no_ref = _mixed_reference_rows()
    base, log_g3 = _ljaux().pairwise_split(
        sq, trial_coords=survivors, ref_coords=refs, chosen_coord=chosen
    )
    assert len(log_g3) == len(survivors)
    assert len(base) == len(survivors)
    # exactly zero, not merely small: these rows have bias exactly 1
    assert np.all(log_g3[no_ref] == 0.0)
    # and not NaN, which is what an unmasked log of the NaN interpolation gives
    assert np.isfinite(log_g3).all()
    # the referenced rows are still biased, or the mask is swallowing everything
    assert np.count_nonzero(log_g3[~no_ref]) > 0


@pytest.mark.parametrize("gamma", GAMMAS)
def test_composition_holds_across_a_mix_of_referenced_and_not(gamma: float) -> None:
    """The mixed case is the one the two paths can disagree on: the combined
    path masks the interpolated value, the split path masks it before taking
    logs, and only equal results prove those are the same mask."""
    survivors, refs, chosen, sq, _ = _mixed_reference_rows()
    pot = _ljaux()
    kw = {
        "trial_coords": survivors,
        "ref_coords": refs,
        "chosen_coord": chosen,
    }
    combined = pot.pairwise(sq, gamma, **kw)
    base, log_g3 = pot.pairwise_split(sq, **kw)
    np.testing.assert_allclose(base * np.exp(gamma * log_g3), combined, rtol=1e-14)


def test_a_pool_with_no_references_at_all_returns_zeros_of_the_right_length() -> None:
    """The short-circuit branch. Still one row per survivor."""
    survivors, refs, chosen, sq = _pair_geometry()
    refs = np.full_like(refs, np.nan)
    base, log_g3 = _aux().pairwise_split(
        sq, trial_coords=survivors, ref_coords=refs, chosen_coord=chosen
    )
    assert np.array_equal(log_g3, np.zeros(len(survivors)))
    assert np.array_equal(base, np.ones(len(survivors)))


def test_the_empty_environment_splits_into_ones_and_zeros() -> None:
    """`kdtree is None` is the empty environment: no neighbours, so no triplets,
    so an empty product. Its log is 0, one per trial, and the reference
    neighbours come back NaN exactly as `weight_bulk` reports them."""
    trials, dists, idxs, _ = _liquid()
    for pot in (_aux(), _ljaux()):
        base, log_g3, ref = pot.weight_bulk_split(
            dists, coords=trials, idxs=idxs, kdtree=None
        )
        assert np.array_equal(log_g3, np.zeros(len(trials)))
        assert len(base) == len(trials)
        assert np.isnan(ref).all()
        combined, _ = pot.weight_bulk(
            dists, coords=trials, idxs=idxs, kdtree=None, gamma=0.5
        )
        np.testing.assert_allclose(base * np.exp(0.5 * log_g3), combined, rtol=1e-14)


# ---- row order cannot change a row's own bias --------------------------------


def test_log_g3_is_permutation_equivariant_bit_for_bit() -> None:
    """The engine permutes the pool before the passes read it
    (`pool[rng.permutation(len(pool))]`), so the same physical trial arrives at
    different rows. Its `log_g3` must be the same float, or the solver's
    permutation invariance (gate G2) is unreachable however carefully the
    reduction is written."""
    survivors, refs, chosen, sq = _pair_geometry()
    pot = _ljaux()
    _, log_g3 = pot.pairwise_split(
        sq, trial_coords=survivors, ref_coords=refs, chosen_coord=chosen
    )
    perm = np.random.default_rng(0).permutation(len(survivors))
    _, log_perm = pot.pairwise_split(
        sq[perm],
        trial_coords=survivors[perm],
        ref_coords=refs[perm],
        chosen_coord=chosen,
    )
    assert np.array_equal(log_perm, log_g3[perm])


def test_bulk_log_g3_is_permutation_equivariant_bit_for_bit() -> None:
    """Same at level 1. The neighbour axis is summed over, and a row
    permutation cannot reach it."""
    trials, dists, idxs, tree = _liquid()
    pot = _ljaux()
    _, log_g3, _ = pot.weight_bulk_split(dists, coords=trials, idxs=idxs, kdtree=tree)
    perm = np.random.default_rng(1).permutation(len(trials))
    _, log_perm, _ = pot.weight_bulk_split(
        dists[perm], coords=trials[perm], idxs=idxs[perm], kdtree=tree
    )
    assert np.array_equal(log_perm, log_g3[perm])


# ---- refusals ----------------------------------------------------------------


def test_a_potential_with_no_separable_bias_refuses_to_split() -> None:
    """Plain LJ has a Boltzmann factor and nothing else; the 2D aux is an
    analytic g(r) with no gamma table behind it. Neither can answer the
    question, and returning a zero `log_g3` would answer it wrongly -- the
    solver would read "gamma changes nothing here" and clamp, from a pool it
    never looked at. Stage 6 refuses these at config time; this is the
    backstop."""
    lj = LennardJones(beta=BETA, upper_cutoff=UPPER_CUTOFF, diameter=DIAMETER)
    aux2d = HardCoreAux2D.build(diameter=DIAMETER, upper_cutoff=UPPER_CUTOFF)
    for pot in (lj, aux2d):
        with pytest.raises(NotImplementedError, match="separable"):
            pot.pairwise_split(np.array([1.5, 2.0]))
        with pytest.raises(NotImplementedError, match="separable"):
            pot.weight_bulk_split(np.array([[1.5, 2.0]]))


# ---- the weigher layer -------------------------------------------------------
#
# The potentials answer what the two factors ARE; the weigher decides when they
# get committed to a pool. Those are separable bugs, so they are separable
# tests: below, the split methods have to commit exactly what the frozen
# `recompute*` methods commit, once the caller has composed them at a gamma.


def _weigher(min_g3: float | None = 0.03) -> EnergyWeigher:
    return EnergyWeigher(_ljaux(min_g3))


def _trial_pool(seed: int = 11) -> tuple[TrialSet, FloatArray, FloatArray]:
    """A survivor pool with arbitrary positive starting weights, the chosen
    trial, and the survivor -> chosen squared distances."""
    survivors, refs, chosen, sq = _pair_geometry(seed)
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 3.0, len(survivors))
    return TrialSet(survivors, w, ref_neighbor_coord=refs), chosen, sq


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_refresh_commits_what_recompute_commits(gamma: float) -> None:
    """The forward pool, both ways round: fold the increment in with the frozen
    method, or take it apart, compose it at the same gamma and apply it. Same
    weights out."""
    weigher = _weigher()
    frozen, chosen, sq = _trial_pool()
    split = frozen.copy()
    weigher.recompute(frozen, chosen, sq, gamma)
    base, log_g3 = weigher.recompute_split(split, chosen, sq)
    weigher.apply_increment(split, base * np.exp(gamma * log_g3))
    np.testing.assert_allclose(split.energy_weight, frozen.energy_weight, rtol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_trial_refresh_commits_what_recompute_trial_commits(
    gamma: float,
) -> None:
    """The reverse pass's trial half."""
    weigher = _weigher()
    frozen, chosen, sq = _trial_pool(seed=23)
    split = frozen.copy()
    weigher.recompute_trial(frozen, chosen, sq, gamma)
    base, log_g3 = weigher.recompute_trial_split(split, chosen, sq)
    weigher.apply_increment(split, base * np.exp(gamma * log_g3))
    np.testing.assert_allclose(split.energy_weight, frozen.energy_weight, rtol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_split_cavity_refresh_commits_what_recompute_cav_commits(
    gamma: float,
) -> None:
    """The reverse pass's cavity half, which is the one with an offset: only
    rows i+1.. are touched, and the split path must touch exactly the same
    ones. A split that folded its increment in from row 0 would double-count
    every member already re-selected."""
    weigher = _weigher()
    frozen, _, _ = _trial_pool(seed=31)
    split = frozen.copy()
    before = frozen.energy_weight.copy()
    i = 2
    chosen = frozen.coords[i]
    weigher.recompute_cav(frozen, i, chosen, gamma, SIDE)
    base, log_g3 = weigher.recompute_cav_split(split, i, chosen, SIDE)
    weigher.apply_increment_cav(split, i, base * np.exp(gamma * log_g3))
    np.testing.assert_allclose(split.energy_weight, frozen.energy_weight, rtol=1e-14)
    # rows up to and including i are untouched by both
    assert np.array_equal(split.energy_weight[: i + 1], before[: i + 1])
    assert len(base) == len(frozen) - (i + 1)


def test_the_cavity_increment_is_committed_through_the_view() -> None:
    """`recompute_cav` multiplies into a slice in place and `recompute` rebinds;
    the reverse pass depends on the first, because it holds `mol_in_cav` across
    levels and reads the updated weights back out of it. The split pair keeps
    that asymmetry rather than tidying it away."""
    weigher = _weigher()
    pool, _chosen, _sq = _trial_pool(seed=41)
    held = pool.energy_weight
    weigher.apply_increment_cav(pool, 1, np.full(len(pool) - 2, 2.0))
    assert pool.energy_weight is held

    held = pool.energy_weight
    snapshot = held.copy()
    weigher.apply_increment(pool, np.full(len(pool), 2.0))
    assert pool.energy_weight is not held
    assert np.array_equal(held, snapshot)


def test_the_weigher_forwards_the_first_level_split_to_its_potential() -> None:
    """`bulk_split` exists so the samplers never have to reach through the
    weigher for `.pot`. Its only job is to be the same arrays."""
    trials, dists, idxs, tree = _liquid()
    weigher = _weigher()
    via_weigher = weigher.bulk_split(dists, coords=trials, idxs=idxs, kdtree=tree)
    direct = _ljaux().weight_bulk_split(dists, coords=trials, idxs=idxs, kdtree=tree)
    for a, b in zip(via_weigher, direct, strict=True):
        assert np.array_equal(a, b, equal_nan=True)


def test_a_weigher_over_a_potential_with_no_bias_refuses_to_split() -> None:
    """Same refusal as the potential's, reached through the layer the sampler
    actually calls."""
    weigher = EnergyWeigher(
        LennardJones(beta=BETA, upper_cutoff=UPPER_CUTOFF, diameter=DIAMETER)
    )
    pool, chosen, sq = _trial_pool()
    with pytest.raises(NotImplementedError, match="separable"):
        weigher.recompute_split(pool, chosen, sq)
    with pytest.raises(NotImplementedError, match="separable"):
        weigher.recompute_cav_split(pool, 0, chosen, SIDE)


@pytest.mark.parametrize("gamma", [0.2903, 1.0])
def test_the_split_cavity_refresh_minimum_images_across_the_boundary(
    gamma: float,
) -> None:
    """The cavity geometry is the one place the split path forms its own
    separations rather than being handed them, so it is the one place it can
    disagree about WHICH IMAGE of the re-selected member the later members see.
    A cavity that straddles x = 0 is the test; the pool the other cavity test
    uses sits well inside the box, where dropping the minimum image changes
    nothing.
    """
    weigher = _weigher()
    rng = np.random.default_rng(5)
    # members clustered around x = 0, half of them wrapped to the far face
    coords = np.column_stack(
        [
            np.concatenate([rng.uniform(0.0, 0.8, 5), rng.uniform(9.2, 10.0, 5)]),
            rng.uniform(4.0, 6.0, 10),
            rng.uniform(4.0, 6.0, 10),
        ]
    )
    refs = coords + rng.uniform(-1.6, 1.6, coords.shape)
    frozen = TrialSet(
        coords, rng.uniform(0.1, 3.0, len(coords)), ref_neighbor_coord=refs
    )
    split = frozen.copy()
    i = 1
    chosen = frozen.coords[i]
    # the guard: without minimum imaging at least one separation is wrong
    raw = np.abs(frozen.coords[i + 1 :] - chosen)
    assert (raw > SIDE / 2).any()

    weigher.recompute_cav(frozen, i, chosen, gamma, SIDE)
    base, log_g3 = weigher.recompute_cav_split(split, i, chosen, SIDE)
    weigher.apply_increment_cav(split, i, base * np.exp(gamma * log_g3))
    np.testing.assert_allclose(split.energy_weight, frozen.energy_weight, rtol=1e-14)
