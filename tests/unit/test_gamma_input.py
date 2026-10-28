"""`auxiliary.gamma`: stating the selection-bias exponent instead of inheriting
the fitted schedule, and sweeping it.

The nine numbers in `_GAMMA_AUX_2D` / `_GAMMA_AUX_3D` were fitted at one trial
count and one box. This key makes them an input, which is what a gamma x
trial-box-side heatmap needs: the exponent is the axis, so it has to be
scannable, and each arm has to be able to say either "this exponent at every
level" or "these exponents at the first levels".

Two rules carry the design and both are tested here.

  PADDING IS ZERO, not repetition. An array shorter than the move is deep
    leaves the deeper levels UNBIASED. The fitted table already ends in three
    zeros, so this follows it, and it is the safe direction: a missing entry
    costs efficiency, an extrapolated one applies a bias nobody chose at a
    depth nobody measured.
  THE OUTER LIST IS THE SWEEP, which is the opposite of the `box` convention
    and deliberately so -- for gamma the common case is ten arms, for a box it
    is one anisotropic value. The cost is that a single array must be wrapped to
    be pinned, and the protection is that a nested list reaching a single run is
    refused rather than flattened.
"""

from __future__ import annotations

import pytest

from gcmc.run import sim_setup
from gcmc.run.hash_fingerprint import run_id
from gcmc.run.launcher import expand_sweep

G3_CSV = "data/g3_analysis_data.csv"
LEVELS = range(1, 10)


def _params(aux_extra=None, dimension=3, sampler_extra=None, aux_type="pmf"):
    aux = {"type": aux_type}
    if aux_type == "pmf":
        aux |= {"g3_csv": G3_CSV, "min_g3": 0.005} if dimension == 3 else {"eta": 0.69}
    else:
        aux |= {"cutoff": 1.5}
    if aux_extra:
        aux.update(aux_extra)
    smp = {"type": "rosenbluth", "trials": 40, "trial_box_side": 2.0}
    if sampler_extra:
        smp.update(sampler_extra)
    return {
        "potential": {"main": "hard_core"},
        "sampler": smp,
        "engine": {"type": "metropolis_dynamics", "depth": 2},
        "ensemble": {"mu": 2.0, "dimension": dimension, "box": 12.0},
        "auxiliary": aux,
        "run": {"kind": "single", "seed": 1, "sampling_window": 10},
    }


def _schedule(cfg):
    return [cfg.gamma_dict[i] for i in LEVELS]


# ---- the value ---------------------------------------------------------------


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_scalar_applies_at_every_level(dimension) -> None:
    """Both dimensions: the 2D analytic hard-disk g(r) and the 3D tabulated g3
    take their exponent from the same level table."""
    cfg = sim_setup.from_params(_params({"gamma": 0.1}, dimension=dimension))
    assert _schedule(cfg) == [0.1] * 9


@pytest.mark.parametrize("dimension", [2, 3])
def test_an_array_fills_the_first_levels_and_zeroes_the_rest(dimension) -> None:
    cfg = sim_setup.from_params(
        _params({"gamma": [0.29, 0.1, 0.03]}, dimension=dimension)
    )
    assert _schedule(cfg) == [0.29, 0.1, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_a_full_nine_array_is_taken_verbatim() -> None:
    """The boundary of the padding rule, and the only form that pins level 9,
    which is also what every level past 9 reads."""
    nine = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    cfg = sim_setup.from_params(_params({"gamma": nine}))
    assert _schedule(cfg) == nine
    assert sorted(cfg.gamma_dict) == list(LEVELS)


def test_a_scalar_and_a_padded_array_agree_where_the_array_reaches() -> None:
    """The two spellings are the same schedule over the levels the array
    covers, which is what makes a scalar arm and an array arm comparable in one
    sweep."""
    scalar = _schedule(sim_setup.from_params(_params({"gamma": 0.1})))
    array = _schedule(sim_setup.from_params(_params({"gamma": [0.1, 0.1, 0.1]})))
    assert scalar[:3] == array[:3]
    assert array[3:] == [0.0] * 6


def test_zero_is_a_legal_exponent_and_is_not_the_same_as_absent() -> None:
    """`gamma: 0` is the bias-OFF control that still pays the auxiliary's cost,
    which is the arm a cost/benefit sweep needs. It must not be mistaken for an
    absent key, which means the fitted schedule."""
    off = sim_setup.from_params(_params({"gamma": 0.0}))
    fitted = sim_setup.from_params(_params())
    assert _schedule(off) == [0.0] * 9
    assert _schedule(fitted) != [0.0] * 9
    assert run_id(off) != run_id(fitted)


# ---- refusals ----------------------------------------------------------------


def test_more_than_nine_values_is_refused_with_the_reason() -> None:
    """Ten values is most often a SWEEP written without the wrapping, so the
    message says so rather than silently dropping the tenth."""
    with pytest.raises(ValueError, match="Rosenbluth levels"):
        sim_setup.from_params(_params({"gamma": [0.1] * 10}))


def test_an_empty_list_is_refused() -> None:
    with pytest.raises(ValueError, match="empty list"):
        sim_setup.from_params(_params({"gamma": []}))


def test_a_nested_list_names_the_launcher() -> None:
    """The failure this prevents: a sweep that was never expanded, where one arm
    would run as if it were the whole experiment."""
    with pytest.raises(ValueError, match="launcher"):
        sim_setup.from_params(_params({"gamma": [[0.1], [0.2]]}))


def test_a_negative_exponent_is_refused() -> None:
    """Legal arithmetic, almost never the intent: it selects AGAINST what the
    auxiliary says is likely. Zero is how a level is switched off."""
    with pytest.raises(ValueError, match="negative exponent"):
        sim_setup.from_params(_params({"gamma": [0.1, -0.2]}))


def test_a_non_finite_exponent_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        sim_setup.from_params(_params({"gamma": float("inf")}))


def test_the_key_is_refused_on_the_short_range_auxiliary() -> None:
    with pytest.raises(ValueError, match="short_range"):
        sim_setup.from_params(_params({"gamma": 0.1}, aux_type="short_range"))


# ---- identity ----------------------------------------------------------------


def test_an_absent_key_is_the_fitted_schedule_and_moves_no_run_id() -> None:
    """The backward-compatibility contract: every params file written before
    this key keeps its numbers and its run directory."""
    cfg = sim_setup.from_params(_params())
    assert _schedule(cfg) == [0.2903, 0.103, 0.0332, 0.0193, 0.0037, 0.0042, 0, 0, 0]
    assert run_id(cfg) == run_id(sim_setup.from_params(_params()))


def test_two_exponents_are_two_run_identities() -> None:
    """They select different trials, so they are different proposal
    distributions and must not share a run directory -- the rule that keeps a
    heatmap's cells apart."""
    ids = {
        run_id(sim_setup.from_params(_params({"gamma": g})))
        for g in (0.0, 0.05, 0.1, [0.1], [0.1, 0.1], [0.2903, 0.103, 0.0332])
    }
    assert len(ids) == 6


# ---- sweeping ----------------------------------------------------------------


def _sweep_base(gamma, sides=(1.0, 1.5, 2.0)):
    p = _params()
    p["auxiliary"]["gamma"] = gamma
    p["sampler"]["trial_box_side"] = list(sides)
    return p


def test_the_outer_list_is_the_sweep_and_crosses_with_the_box_side() -> None:
    """The heatmap this axis exists for: 10 exponents x 8 sides = 80 runs, each
    with its own directory."""
    gammas = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    sides = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    combos, labels = expand_sweep(_sweep_base(gammas, sides))
    assert len(combos) == 80
    assert len(set(labels)) == 80
    assert {c["auxiliary"]["gamma"] for c in combos} == set(gammas)
    assert {c["sampler"]["trial_box_side"] for c in combos} == set(sides)


def test_an_arm_may_be_an_array_and_gets_a_readable_label() -> None:
    combos, labels = expand_sweep(_sweep_base([0.1, [0.29, 0.103, 0.0332]]))
    assert len(combos) == 6  # 2 gamma x 3 sides
    arms = [c["auxiliary"]["gamma"] for c in combos]
    assert 0.1 in arms and [0.29, 0.103, 0.0332] in arms
    assert any("gamma0.29x0.103x0.0332" in lbl for lbl in labels)


def test_a_wrapped_array_is_one_arm_not_a_sweep() -> None:
    """The price of the convention, pinned so it cannot drift: to fix a single
    per-level array, wrap it."""
    combos, _ = expand_sweep(_sweep_base([[0.29, 0.103, 0.0332]]))
    assert len(combos) == 3  # the three sides only
    assert all(c["auxiliary"]["gamma"] == [0.29, 0.103, 0.0332] for c in combos)


def test_a_scalar_arm_is_not_a_sweep_and_stays_out_of_the_label() -> None:
    combos, labels = expand_sweep(_sweep_base(0.1))
    assert len(combos) == 3
    assert all("gamma" not in lbl for lbl in labels)


def test_every_swept_arm_survives_from_params_in_both_dimensions() -> None:
    """The end-to-end contract of the sweep: what the launcher writes, the
    validator accepts. A refusal here would surface as 80 dead jobs."""
    for dimension in (2, 3):
        base = _params(dimension=dimension)
        base["auxiliary"]["gamma"] = [0.0, 0.1, [0.29, 0.103], [0.5] * 9]
        base["sampler"]["trial_box_side"] = [1.0, 2.0]
        combos, _ = expand_sweep(base)
        assert len(combos) == 8
        ids = {run_id(sim_setup.from_params(c)) for c in combos}
        assert len(ids) == 8


def test_a_long_array_gets_a_digest_tail_rather_than_a_ninety_character_name() -> None:
    """The run directory is how a human finds a run. A nine-level arm spelled
    out is about ninety characters of hyphenated decimals, so past a threshold
    the tail becomes a digest of the values -- still unique, still stable across
    relaunches, and still readable at the front."""
    from gcmc.run.launcher import _gamma_label

    fitted_2d = [
        0.193548,
        0.049948,
        0.022557,
        0.010187,
        0.007558,
        0.002438,
        0.00118,
        0.000495,
        0.000176,
    ]
    lbl = _gamma_label(fitted_2d)
    assert lbl.startswith("0.193548x9Lx")
    assert len(lbl) < 30
    assert _gamma_label(fitted_2d) == lbl  # stable, not salted
    # a different arm of the same length and first value still separates
    other = list(fitted_2d)
    other[4] = 0.5
    assert _gamma_label(other) != lbl
    # short arms are still spelled out
    assert _gamma_label([0.29, 0.103]) == "0.29x0.103"
    assert _gamma_label(0.1) == "0.1"
    # and nothing in a label may collide with the directory sanitiser, which
    # turns '-' into 'm' and '.' into 'p': a hyphen would read as a subtraction
    assert "-" not in lbl and "-" not in _gamma_label([0.29, 0.103])


# --------------------------------------------------------------------------
# auxiliary.g3_r_cut: dropping the neighbours that carry no ranking.
#
# The bias is a product over neighbours, so a pair with no angular structure
# left still multiplies in, costing a lookup and adding a dependence on the
# neighbour COUNT. On the shipped table that describes everything past r = 1.9.
# --------------------------------------------------------------------------


def test_the_cutoff_reaches_the_potential_and_defaults_to_absent() -> None:
    absent = sim_setup.from_params(_params({"gamma": 0.2}))
    cut = sim_setup.from_params(_params({"gamma": 0.2, "g3_r_cut": 1.8}))
    assert absent.g3_r_cut is None
    assert cut.g3_r_cut == 1.8
    assert absent.build_potential().r_cut is None
    assert cut.build_potential().r_cut == 1.8


def test_the_cutoff_is_part_of_the_run_identity() -> None:
    ids = {
        run_id(sim_setup.from_params(_params({"gamma": 0.2, **extra})))
        for extra in ({}, {"g3_r_cut": 1.8}, {"g3_r_cut": 1.9})
    }
    assert len(ids) == 3


def test_a_cutoff_at_or_inside_contact_is_refused() -> None:
    """Every pair would be dropped, so the bias would be identically 1 at every
    trial while still paying for the table. An auxiliary that does nothing is
    worth refusing by name rather than running."""
    with pytest.raises(ValueError, match="inside contact"):
        sim_setup.from_params(_params({"gamma": 0.2, "g3_r_cut": 1.0}))


def test_the_cutoff_is_refused_in_two_dimensions() -> None:
    with pytest.raises(ValueError, match="3D pmf key"):
        sim_setup.from_params(_params({"gamma": 0.2, "g3_r_cut": 1.8}, dimension=2))
