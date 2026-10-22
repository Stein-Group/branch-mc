"""The per-pair force field reaches the engine, and reduces when it should.

Two claims, and they are only worth anything together.

REDUCTION. A typed run whose species file has one species at sigma = epsilon = 1
must reproduce the untyped run BIT FOR BIT -- same counters, same positions, same
N at every window. If it does not, the typed path changed something it had no
business changing, and every existing single-species result is in question. This
is the same discipline `potentials/pair_table.py` applies to its own arithmetic,
carried up to the whole engine.

DISCRIMINATION. A reduction test alone is satisfied by a typed path that ignores
the table completely -- which is precisely the defect S3 found and this phase
fixes, where `build_potential` returned a single-sigma potential for typed runs
and every species interacted identically. So the second claim is that a species
file whose pairs DIFFER produces different physics from one whose pairs do not.
Without it, "reduces correctly" would be evidence of nothing.

SCOPE. The conventional move path only (`sampler.type='rosenbluth'`), which is
the fence `potentials/lj_table.py` sets for itself: the sweep samplers build
clique adjacency at one scalar floor and their intra-clique factor at
sigma = epsilon = 1, inside AST-locked kernels, and are refused rather than
silently flattened. `test_the_sweep_refuses_a_species_file` pins that refusal so
it cannot quietly become a silent flattening later.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics
from gcmc.forcefield.lennard_jones import LennardJones
from gcmc.forcefield.lj_table import LennardJonesTable
from gcmc.forcefield.pair_table import PairTable
from gcmc.run import sim_setup
from gcmc.system.state import ChainState

_BOX, _MU, _T, _N0 = 6.0, 1.0, 1.2, 30


def _params(species_file: str | None, sampler: str = "rosenbluth") -> dict:
    if species_file is None:
        pot = {"main": "lennard_jones", "T": _T, "tail": False}
        ens = {"mu": _MU, "dimension": 3, "box": _BOX}
    else:
        # no `tail` key: a species file forces the tail off, and stating it would
        # hide whether that rule is working
        pot = {"main": "lennard_jones", "T": _T, "species_file": species_file}
        ens = {"mu": {"A": _MU}, "dimension": 3, "box": _BOX}
    return {
        "potential": pot,
        "sampler": {"type": sampler, "trials": 20},
        "engine": {"depth": 1},
        "ensemble": ens,
        "run": {"sampling_window": 60},
    }


def _advance(species_file: str | None, n_windows: int = 4) -> tuple:
    cfg = sim_setup.from_params(_params(species_file))
    pot = cfg.build_potential()
    wgh = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    rng = np.random.default_rng(3)
    state = ChainState(
        positions=rng.uniform(0, _BOX, (_N0, 3)),
        rng=np.random.default_rng(4),
        step=0,
    )
    if species_file is not None:
        state.species = np.zeros(_N0, dtype=np.int64)
    summaries = [
        metropolis_dynamics.advance(state, cfg, pot, wgh) for _ in range(n_windows)
    ]
    return cfg, state, summaries


# ------------------------------------------------------------------ reduction


def test_one_species_at_unit_sigma_reduces_to_the_untyped_engine() -> None:
    """The whole engine, not just the pair energy: identical chain."""
    cfg_u, st_u, sum_u = _advance(None)
    cfg_t, st_t, sum_t = _advance("ff/species_one_reduced.yaml")

    assert isinstance(cfg_u.build_potential(), LennardJones)
    assert isinstance(cfg_t.build_potential(), LennardJonesTable), (
        "the typed run did not get the table-aware potential, so this test would "
        "pass for the wrong reason"
    )

    for i, (a, b) in enumerate(zip(sum_u, sum_t)):
        assert a["counters"] == b["counters"], (
            f"window {i}: counters diverged.\n  untyped {a['counters']}\n"
            f"  typed   {b['counters']}\n"
            "A degenerate typed run must be the untyped chain. A divergence here "
            "means the typed path consumed the rng differently or weighted a "
            "trial differently, and every single-species result is affected."
        )
    pu = np.asarray(st_u.positions)
    pt = np.asarray(st_t.positions)
    assert pu.shape == pt.shape and np.array_equal(pu, pt), (
        "final configurations differ; the reduction is not bit-exact"
    )


def test_a_species_file_forces_the_tail_off() -> None:
    """The mixture tail is composition-dependent and not implemented, so a typed
    run must not silently pick up the sigma=1 bulk tail its own force field
    forbids."""
    cfg = sim_setup.from_params(_params("ff/species_one_reduced.yaml"))
    assert cfg.lj_tail_pref == 0.0

    asked = _params("ff/species_one_reduced.yaml")
    asked["potential"]["tail"] = True
    with pytest.raises(ValueError, match="composition-dependent"):
        sim_setup.from_params(asked)


# -------------------------------------------------------------- it discriminates


def test_the_table_actually_reaches_the_weights() -> None:
    """A non-unit table must give different weights from the frozen potential.

    The unit-argument for this test: a typed path that ignored the table would
    pass every reduction check above, because at one unit species there is
    nothing to ignore. This is the assertion that fails if the table is present
    and unread -- which is exactly the state S3 found the engine in.
    """
    beta, uc = 1.0 / _T, 2.5
    unit = PairTable.from_reduced(("A",), 1, np.array([[1.0]]), np.array([[1.0]]), 0.9)
    wide = PairTable.from_reduced(("A",), 1, np.array([[1.3]]), np.array([[1.0]]), 0.9)

    d = np.array([[1.4, 2.0, np.inf]])
    tt = np.zeros(1, dtype=int)
    nt = np.zeros((1, 3), dtype=int)

    w_unit = LennardJonesTable(beta=beta, table=unit, upper_cutoff=uc).weight_cloud(
        d, tt, nt
    )
    w_wide = LennardJonesTable(beta=beta, table=wide, upper_cutoff=uc).weight_cloud(
        d, tt, nt
    )
    frozen = LennardJones(
        beta=beta, upper_cutoff=uc, diameter=1.0, volume=None
    ).weight_pair(d)

    assert np.array_equal(w_unit, frozen), "the unit table is not the frozen path"
    assert not np.allclose(w_wide, frozen), (
        "a sigma of 1.3 gave the same weight as sigma 1.0, so the pair table is "
        "not being read"
    )


def test_mixed_trial_types_are_weighted_per_pair() -> None:
    """Each trial in a cloud is weighted at ITS OWN type.

    The engine weights a whole trial cloud in one call and, after P3, each trial
    carries its own drawn species. A potential that applied one type to the whole
    cloud would pass a same-type test and be wrong here.
    """
    beta, uc = 1.0 / _T, 2.5
    sigma = np.array([[1.0, 1.15], [1.15, 1.3]])
    eps = np.array([[1.0, 1.0], [1.0, 1.0]])
    table = PairTable.from_reduced(("A", "B"), 2, sigma, eps, 0.9)
    lt = LennardJonesTable(beta=beta, table=table, upper_cutoff=uc)

    d = np.array([[1.5, 2.0], [1.5, 2.0]])  # identical distances
    nt = np.zeros((2, 2), dtype=int)  # identical neighbours (type A)
    w = lt.weight_cloud(d, np.array([0, 1]), nt)  # different TRIAL types

    assert not np.isclose(w[0], w[1]), (
        "an A trial and a B trial at identical distances against identical "
        "neighbours got the same weight, so the trial's own type is being ignored"
    )
    # and each row equals what a single-type instance would give for that type
    for t in (0, 1):
        one = LennardJonesTable(
            beta=beta, table=table, trial_type=t, upper_cutoff=uc
        ).weight_pair(d[t : t + 1], neighbour_types=nt[t : t + 1])
        assert np.allclose(w[t], one), f"row for type {t} disagrees with _pair_u"


def test_the_overlap_floor_is_per_pair() -> None:
    """A big species must exclude at a bigger radius than a small one.

    One scalar floor for a mixture is not a small error: it lets a large pair sit
    inside its own core, where the repulsion is enormous, and it excludes small
    pairs that were admissible. Both directions distort the density.
    """
    sigma = np.array([[1.0, 1.15], [1.15, 1.6]])
    eps = np.ones((2, 2))
    table = PairTable.from_reduced(("A", "B"), 2, sigma, eps, 0.9)

    from gcmc.forcefield.overlap import pair_overlaps

    d = np.array([1.2, 1.2])
    types = np.array([0, 1])
    got = pair_overlaps(d, types, types, table.core_floor)
    assert not bool(got[0]), "A-A at 1.2 is outside its 0.9 floor and admissible"
    assert bool(got[1]), "B-B at 1.2 is inside its 1.44 floor and must overlap"


# ------------------------------------------------------------------- the fence


def test_every_sampler_accepts_a_species_file() -> None:
    """All three samplers read the pair table. This is the completion of P3.

    `potential_type` collapses branch and channel into one tag, but they do not
    share a clique kernel, so support had to be established per SAMPLER and
    arrived in that order:

        rosenbluth      per-pair weights and per-pair overlap floors
        channel         the mixture overlap policy -- this already
                        existed for the MOF path; what was missing was
                        `make_sweep` selecting them
        branch          count_and_sample / ljbranch_weight_sum build adjacency at
                        the per-pair floor and take the intra-clique factor from
                        the table

    WHAT THIS TEST IS FOR NOW THAT NOTHING IS REFUSED. It asserts that every
    sampler receives the TABLE-AWARE potential, which is the thing that was
    silently false at the start of S3: `build_potential` returned a single-sigma
    `LennardJones` for typed runs, so every species interacted identically and
    the run produced an isotherm for a system nobody configured. A regression
    here does not raise -- it returns the wrong potential and keeps going. That is
    exactly why it is asserted by type rather than by absence of an error.

    A NON-degenerate file is used deliberately: a file whose pairs all share one
    sigma is represented exactly by the scalar kernels, so it would make this
    test vacuous.
    """
    ff = "ff/species_three_reduced.yaml"
    mus = {"A": 1.0, "B": 1.0, "C": 1.0}

    def cfg_for(sampler: str):
        params = _params(ff, sampler=sampler)
        params["ensemble"]["mu"] = mus
        if sampler == "channel":
            params["ensemble"]["box"] = 12.0
            params["sampler"]["channel_length"] = 4.0
            params["sampler"]["channel_width"] = 1.5
        return sim_setup.from_params(params)

    for sampler in ("rosenbluth", "channel", "branch"):
        pot = cfg_for(sampler).build_potential()
        assert isinstance(pot, LennardJonesTable), (
            f"{sampler} got {type(pot).__name__}, not the table-aware potential. "
            "A single-sigma potential on a typed run does not fail -- it flattens "
            "the force field and returns a plausible isotherm."
        )


def test_a_degenerate_file_is_allowed_everywhere() -> None:
    """One sigma, one epsilon, one core floor: the scalar kernels are exact.

    Not a loophole but the true condition, and it is load-bearing twice over --
    it is what lets the M21 arbiter run at all, and it is the only case in which
    the table path can be compared against the untyped chain.
    """
    for sampler in ("rosenbluth", "branch"):
        params = _params("ff/species_degenerate_reduced.yaml", sampler=sampler)
        params["ensemble"]["mu"] = {"A": 1.0, "B": 1.0, "C": 1.0}
        cfg = sim_setup.from_params(params)
        assert isinstance(cfg.build_potential(), LennardJonesTable)
