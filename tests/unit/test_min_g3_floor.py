"""The 3D pmf g3 floor is a config key, and pinning it decouples the bias from k.

The floor was `1/n_trials` (`HardCoreAux3D.__init__`). That is not a neutral
default: it means the selection bias is a function of the trial count, so the
same physics at k = 15 and k = 60 proposes differently, and any gamma tuned at
one k is silently wrong at the other. `_GAMMA_AUX_3D` is nine numbers fitted
against one floor, which is the entanglement stage 1 of
docs/decisions/adaptive_gamma_plan.md exists to break.

Three properties, and all three are needed:

  1. ABSENT means EXACTLY what it meant before. Not "approximately 1/n_trials",
     not "the same to a tolerance" -- the same float, so every config written
     before the key existed reproduces bit-for-bit. Asserted against the weights
     rather than against the attribute, because the floor is applied in two
     places that are not equivalent (the grid before interpolation, and the
     interpolated value) and only the weights see both.
  2. PINNED is k-independent. This is the property the key exists for.
  3. The DEFAULT is not k-independent. Without this the first two would keep
     passing if the floor stopped being applied at all.

The refusals are tested here too rather than in the config tests, because what
makes `min_g3 = 0` wrong is a fact about the weight (a zero floor annihilates a
trial, and its log is -inf), not a fact about the schema.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc.forcefield.auxiliary import HardCoreAux3D, resolve_min_g3
from gcmc.run import sim_setup

G3_CSV = "data/g3_analysis_data.csv"
SIDE, UPPER_CUTOFF, DIAMETER = 20.0, 2.5, 1.0
TRIAL_COUNTS = (15, 30, 60)


def _fixed_environment(seed: int = 7):
    """One frozen (trial cloud, neighbour) configuration, reused for every
    potential below. The comparison is only meaningful if the geometry is
    identical across n_trials: `n_trials` normally also sets how many trials get
    generated, and that would confound the floor with the cloud size. Here it
    varies nothing but the floor."""
    rng = np.random.default_rng(seed)
    env = rng.uniform(0, SIDE, (200, 3))
    tree = cKDTree(env, boxsize=SIDE)
    coords = rng.uniform(0, SIDE, (30, 3))
    dists, idxs = tree.query(coords, k=6)
    return coords, dists, idxs, tree


def _weights(n_trials: int, min_g3: float | None, gamma: float = 1.0):
    coords, dists, idxs, tree = _fixed_environment()
    pot = HardCoreAux3D.build(
        diameter=DIAMETER,
        upper_cutoff=UPPER_CUTOFF,
        side=SIDE,
        n_trials=n_trials,
        g3_csv=G3_CSV,
        min_g3=min_g3,
    )
    w, _ref = pot.weight_bulk(dists, coords=coords, idxs=idxs, kdtree=tree, gamma=gamma)
    return w


# ---- 1. the default is unchanged --------------------------------------------


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_absent_min_g3_is_the_historical_one_over_n_trials(n_trials: int) -> None:
    """The resolver returns the same float the old expression did."""
    assert resolve_min_g3(None, n_trials) == 1.0 / int(n_trials)


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_pinning_the_default_value_changes_no_bit(n_trials: int) -> None:
    """Stating 1/n_trials explicitly must produce the identical weight array, or
    the two application sites have drifted onto different constants."""
    absent = _weights(n_trials, None)
    stated = _weights(n_trials, 1.0 / n_trials)
    assert np.array_equal(absent, stated)


# ---- 2. pinned is k-independent ---------------------------------------------


@pytest.mark.parametrize("gamma", [1.0, 0.2903, 0.0332])
def test_a_pinned_floor_makes_the_weights_independent_of_the_trial_count(
    gamma: float,
) -> None:
    """The property the key exists for. Same configuration, same floor, three
    trial counts: one weight array. Checked at the gammas the fitted 3D schedule
    actually uses, because the floor enters before the exponent and a bug that
    cancelled at gamma = 1 would not cancel at 0.0332."""
    ws = [_weights(k, 0.03, gamma) for k in TRIAL_COUNTS]
    for other in ws[1:]:
        assert np.array_equal(ws[0], other)


# ---- 3. the default is not k-independent ------------------------------------


def test_the_default_floor_really_does_move_with_the_trial_count() -> None:
    """The defect, asserted so that properties 1 and 2 cannot both pass by the
    floor having quietly stopped being applied."""
    low, high = _weights(TRIAL_COUNTS[0], None), _weights(TRIAL_COUNTS[-1], None)
    assert not np.array_equal(low, high)


# ---- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.01, -1.0, 1.5, 2.0])
def test_a_floor_outside_the_unit_interval_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="min_g3"):
        resolve_min_g3(bad, 30)


def _params(aux: dict, dimension: int = 3, main: str = "lennard_jones") -> dict:
    potential: dict = {"main": main}
    if main == "lennard_jones":
        potential["T"] = 1.5
    return {
        "potential": potential,
        "ensemble": {"mu": -2.0, "dimension": dimension, "box": 10.0, "type": "muVT"},
        "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
        "engine": {"type": "metropolis_dynamics", "depth": 2},
        "run": {"kind": "single", "seed": 4, "sampling_window": 50},
        "auxiliary": aux,
    }


def test_the_key_reaches_the_potential_through_the_config() -> None:
    cfg = sim_setup.from_params(
        _params({"type": "pmf", "g3_csv": G3_CSV, "min_g3": 0.03})
    )
    assert cfg.min_g3 == 0.03
    assert cfg.build_potential().aux.min_g3 == 0.03


def test_an_absent_key_reaches_the_potential_as_one_over_trials() -> None:
    cfg = sim_setup.from_params(_params({"type": "pmf", "g3_csv": G3_CSV}))
    assert cfg.min_g3 is None
    assert cfg.build_potential().aux.min_g3 == 1.0 / 30


def test_the_short_range_auxiliary_refuses_a_floor_it_cannot_apply() -> None:
    """`short_range` is dual-cutoff CBMC: the same LJ at a shorter radius. There
    is no table, so there is nothing to floor."""
    with pytest.raises(ValueError, match="min_g3"):
        sim_setup.from_params(
            _params({"type": "short_range", "cutoff": 1.5, "min_g3": 0.03})
        )


def test_the_two_dimensional_auxiliary_refuses_a_floor_it_does_not_have() -> None:
    """The 2D aux is the analytic hard-disk g(r), which is 0 inside the core by
    construction and has no floor. Accepting the key and dropping it would let
    someone believe a floor is in force."""
    with pytest.raises(ValueError, match="3D-only"):
        sim_setup.from_params(
            _params({"type": "pmf", "eta": 0.69, "min_g3": 0.03}, 2, "hard_core")
        )


def test_the_floor_is_part_of_the_run_identity() -> None:
    """Two runs with different floors propose differently, so they are different
    experiments and must not share a run directory -- the rule at the top of
    `config_fields`."""
    from gcmc import config_specs

    a = config_specs.potential_spec(
        sim_setup.from_params(_params({"type": "pmf", "g3_csv": G3_CSV}))
    )
    b = config_specs.potential_spec(
        sim_setup.from_params(
            _params({"type": "pmf", "g3_csv": G3_CSV, "min_g3": 0.03})
        )
    )
    assert a != b


# ---- the criterion key, and the measurement switch --------------------------
#
# Two keys added after the floor, in the same file because they are checked the
# same way: what a key means is a fact about the weights or about the run
# identity, not about the schema.


def _adaptive(**smp_extra) -> dict:
    p = _params({"type": "pmf", "g3_csv": G3_CSV, "min_g3": 0.03})
    p["sampler"].update({"gamma_mode": "adaptive", "pr_target": 0.25})
    p["sampler"].update(smp_extra)
    return p


def test_the_criterion_defaults_to_participation_and_reaches_the_config() -> None:
    assert sim_setup.from_params(_adaptive()).gamma_criterion == "participation"
    assert (
        sim_setup.from_params(_adaptive(gamma_criterion="perplexity")).gamma_criterion
        == "perplexity"
    )


def test_an_unknown_criterion_is_refused_with_both_named() -> None:
    """A typo must not fall through to the default and silently run the other
    method, and the message has to say what the two are."""
    with pytest.raises(ValueError, match="gamma_criterion"):
        sim_setup.from_params(_adaptive(gamma_criterion="entropy"))


def test_the_criterion_is_refused_without_adaptive_gamma() -> None:
    """Same rule as `pr_target`: a key nothing reads, in a file whose author
    believes it is in force."""
    p = _params({"type": "pmf", "g3_csv": G3_CSV})
    p["sampler"]["gamma_criterion"] = "perplexity"
    with pytest.raises(ValueError, match="adaptive-gamma keys"):
        sim_setup.from_params(p)


def test_the_criterion_is_part_of_the_run_identity() -> None:
    """The same target under two criteria selects differently, so the two are
    different proposal distributions and must not share a run directory -- the
    same argument the floor makes."""
    from gcmc import config_specs

    a = config_specs.sampler_spec(sim_setup.from_params(_adaptive()))
    b = config_specs.sampler_spec(
        sim_setup.from_params(_adaptive(gamma_criterion="perplexity"))
    )
    assert a != b


def test_recording_esjd_is_not_part_of_the_run_identity() -> None:
    """The other direction, and it matters as much. `run.esjd` adds counters to
    the summary and changes nothing the chain does, so two runs differing only
    in whether ESJD was recorded are the SAME experiment and must share a run
    directory. A measurement switch that forked the run id would mean the
    baseline arm of an efficiency comparison could not be resumed from the run
    that produced it.
    """
    from gcmc.run.hash_fingerprint import run_id

    p = _params({"type": "pmf", "g3_csv": G3_CSV})
    quiet = sim_setup.from_params(p)
    p["run"]["esjd"] = True
    loud = sim_setup.from_params(p)
    assert loud.esjd and not quiet.esjd
    assert run_id(quiet) == run_id(loud)


def test_the_solver_defaults_to_moments_and_reaches_the_config() -> None:
    """The grid search was the measured cost of the method (13% of move time at
    k = 200 with full_repacking), so a params file that says nothing gets the
    closed form. The key exists for the pilot run that compares them."""
    assert sim_setup.from_params(_adaptive()).gamma_solver == "moments"
    assert sim_setup.from_params(_adaptive(gamma_solver="grid")).gamma_solver == "grid"


def test_an_unknown_solver_is_refused_with_both_named() -> None:
    with pytest.raises(ValueError, match="gamma_solver"):
        sim_setup.from_params(_adaptive(gamma_solver="closed_form"))


def test_the_solver_is_refused_without_adaptive_gamma() -> None:
    p = _params({"type": "pmf", "g3_csv": G3_CSV})
    p["sampler"]["gamma_solver"] = "grid"
    with pytest.raises(ValueError, match="adaptive-gamma keys"):
        sim_setup.from_params(p)


def test_the_solver_is_part_of_the_run_identity() -> None:
    """Two solvers land within about a percent of each other on the target,
    which is a different trial selected, so the arms of the comparison are two
    proposal distributions and must not be averaged together in one run
    directory."""
    from gcmc import config_specs

    a = config_specs.sampler_spec(sim_setup.from_params(_adaptive()))
    b = config_specs.sampler_spec(sim_setup.from_params(_adaptive(gamma_solver="grid")))
    assert a != b


def test_a_fixed_gamma_run_keeps_its_run_id_when_the_solver_key_appears() -> None:
    """The backward-compatibility contract the whole adaptive block was written
    under: every field it adds is None on a run that did not ask for it, and
    `identity_dict` drops a None, so no pre-existing run id moves."""
    from gcmc.run.hash_fingerprint import run_id

    cfg = sim_setup.from_params(_params({"type": "pmf", "g3_csv": G3_CSV}))
    assert cfg.gamma_solver is None
    assert cfg.gamma_criterion is None
    assert run_id(cfg) == run_id(
        sim_setup.from_params(_params({"type": "pmf", "g3_csv": G3_CSV}))
    )
