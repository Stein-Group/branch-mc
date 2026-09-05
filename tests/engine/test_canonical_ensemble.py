"""A mixture at fixed N_A and N_B: the canonical ensemble, and the only one here
that can hold a two-phase box.

WHY IT EXISTS, in one paragraph, because three cluster campaigns were spent
learning it. The other fixed-N ensemble, now called `semigrand`, lets the repack
draw the species of every particle it inserts, so total N is fixed and
composition is free. That samples a correct distribution and it cannot hold an
interface -- a single phase costs no interface, nothing constrains the
composition, and the box goes there. Measured at k = 0.8 from two slabs: nearly
one component within half a million cycles. `ensemble.type = "NVT"` with a
species file removes the escape route by removing the move that takes it:
translations and pair identity exchanges only, both of which conserve every
species count exactly.

WHY IT IS A NAME AND NOT A FLAG. It was `"NVT"` plus `fixed_composition: true`
for one stage, and that spelling made SEMI-GRAND the thing you got by forgetting
something -- which is exactly what happened, three times, under a params file
that read as canonical. The two ensembles now have two names, and a file that
still carries the flag is refused by name rather than reinterpreted.

The gates below are in the order they would fail if the move were wrong:

  1. composition EXACTLY constant, every window. The defining property, and the
     one the semi-grand path does not have.
  2. the move is REACHABLE and its dU is exactly zero for degenerate species --
     at k = 0 the two labels are one species, so every non-null proposal must
     accept, and the null rate is the analytic sum_s x_s^2.
  3. the SIGN of dU, which no accept-rate check can see: from one mixed start,
     low temperature demixes (alpha rises) and high temperature does not
     (alpha stays at zero). A sign error would demix the hot arm and mix the
     cold one.
  4. the validator refusals, at parse time.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.analysis import demixing
from gcmc.engine import MC_engine
from gcmc.run import runner, sim_setup
from gcmc.system.state import ChainState

BOX = 10.0
N = 600
MODEL_DEMIX = "ff/species_symmetric_k0.8.yaml"
MODEL_NULL = "ff/species_symmetric_k0.yaml"


def _start(tmp_path, model, arrangement="random"):
    from scripts.demixing import write_mixed_start

    return str(
        write_mixed_start(
            tmp_path / f"start_{arrangement}.extxyz",
            model,
            BOX,
            0.4,
            3,
            0,
            0.5,
            N,
            arrangement,
        )
    )


def _params(start, model, temp=1.0, window=4000, **run_over):
    p = {
        "potential": {"main": "lennard_jones", "T": temp, "species_file": model},
        # A canonical run has no trial cloud, so this is the only legal block.
        "sampler": {"type": "metropolis"},
        "ensemble": {
            "type": "NVT",
            "dimension": 3,
            "box": BOX,
            "N": N,
        },
        "run": {
            "kind": "single",
            "seed": 3,
            "traj_every": 0,
            "sampling_window": window,
            "target_cycles": window,
            "checkpoint_every": 5,
            "initial_config": start,
        },
    }
    p["run"].update(run_over)
    return p


def _drive(params, windows=1):
    """Advance the engine directly, returning (state, per-window summaries)."""
    cfg = sim_setup.from_params(params)
    pot = cfg.build_potential()
    weigher = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    state = ChainState.fresh(3, seed=3)
    state.positions = runner._init_config(cfg, state.rng)
    state.species = runner._init_species(cfg, len(state.positions))
    out = [MC_engine.advance(state, cfg, pot, weigher) for _ in range(windows)]
    return state, out


def test_every_species_count_is_exactly_constant(tmp_path):
    start = _start(tmp_path, MODEL_DEMIX)
    _state, summaries = _drive(_params(start, MODEL_DEMIX), windows=4)

    counts = [tuple(s["n_by_species"].values()) for s in summaries]
    assert len(set(counts)) == 1, (
        f"the composition moved across windows: {counts}. Fixed composition is "
        "the entire point of this ensemble -- a run whose counts drift is the "
        "semi-grand path under another name"
    )
    assert sum(counts[0]) == N
    assert {s["N"] for s in summaries} == {N}

    # ...and not constant because nothing happened, which is the reachable-value
    # half: both moves must actually be proposed.
    c = summaries[-1]["counters"]
    assert c["move_step"] > 0 and c["swap_step"] > 0, (
        f"a canonical window proposed {c['move_step']} translations and "
        f"{c['swap_step']} pair exchanges; it is not sampling"
    )


def test_degenerate_species_accept_every_pair_exchange(tmp_path):
    """At k = 0 the two labels are one species, so a pair exchange changes no
    energy at all and must accept unconditionally.

    The acceptance is compared against the ANALYTIC null rate rather than a
    threshold: a pair drawn uniformly is like-species with probability
    sum_s x_s^2, which is 1/2 at equal composition, and those proposals are
    counted and discarded. So the accepted fraction must be 1 - 1/2, and any
    real rejection would push it below.
    """
    start = _start(tmp_path, MODEL_NULL)
    _state, summaries = _drive(_params(start, MODEL_NULL), windows=2)
    c = summaries[-1]["counters"]
    like_fraction = 0.5  # x_A = x_B = 1/2 exactly, by construction of the start
    expected = 1.0 - like_fraction
    got = c["swap_accept"] / c["swap_step"]
    sd = np.sqrt(expected * (1 - expected) / c["swap_step"])
    assert abs(got - expected) < 4 * sd, (
        f"degenerate species accepted {got:.3f} of pair exchanges against the "
        f"{expected:.3f} the null rate alone predicts ({sd:.4f} sd). Anything "
        "below means a dU that should be identically zero is not"
    )


def test_the_sign_of_the_energy_change_is_right(tmp_path):
    """One mixed start, two temperatures, opposite outcomes.

    This is the gate no acceptance rate can stand in for. If the sign of dU were
    flipped the cold arm would refuse to demix and the hot arm would order, and
    both would still show a perfectly plausible acceptance fraction.
    """
    start = _start(tmp_path, MODEL_DEMIX)

    def alpha_after(temp):
        state, _ = _drive(
            _params(start, MODEL_DEMIX, temp=temp, window=8000), windows=3
        )
        return demixing.order_parameter_of(
            state.positions, state.species, BOX, 3, 2, demixing.SHELL_R_CUT
        )

    cold, hot = alpha_after(1.0), alpha_after(50.0)
    assert cold > 0.25, (
        f"at T* = 1.0 a mixture with eps_AB = 0.2 eps read alpha = {cold:.4f} "
        "after 24k proposals; it should be well on its way to separated"
    )
    assert abs(hot) < 0.1, (
        f"at T* = 50 the Boltzmann factor is ~1 for every exchange, so the "
        f"labels should be random; alpha = {hot:.4f}"
    )
    assert cold > hot + 0.2


@pytest.mark.parametrize(
    ("over", "message"),
    [
        # the sibling fixed-N ensemble DEMANDS the route this one forbids, which
        # is the whole reason they are two names
        ({"type": "semigrand"}, "exactly one state-point route"),
        # the activity ratio is the knob that would turn and do nothing here
        ({"mu": {"A": 0.0, "B": 0.0}}, "would do nothing"),
        # and the retired flag is refused by name rather than ignored
        ({"fixed_composition": True}, "fixed_composition is gone"),
    ],
)
def test_the_refusals(over, message, tmp_path):
    start = _start(tmp_path, MODEL_DEMIX)
    p = _params(start, MODEL_DEMIX)
    p["ensemble"].update(over)
    with pytest.raises(ValueError, match=message):
        sim_setup.from_params(p)


def test_untyped_nvt_is_not_canonical_and_needs_no_start_file(tmp_path):
    """With one species there is no composition to hold, so NVT means what it
    always meant and the canonical demands do not apply to it.

    This is the edge the first draft of the rename got wrong in three places at
    once: the validator, the engine and the log each decided "is this canonical?"
    from a different quantity (species file present / n_species > 1 / species
    NAMES present), so a one-species species file would have been canonical to
    one of them and semi-grand to another. They all read the species COUNT now.
    """
    p = {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "branch", "depth": 1, "trials": 10, "trial_box_side": 3.0},
        "ensemble": {"type": "NVT", "dimension": 3, "box": 12.0, "N": 100},
        "run": {"kind": "single", "radius": 0.5, "seed": 1},
    }
    cfg = sim_setup.from_params(p)  # no initial_config, and that is fine
    assert cfg.ensemble_type == "NVT" and cfg.n_species == 1


def test_a_canonical_run_without_a_start_file_is_refused_by_name(tmp_path):
    """The composition is an INPUT to a canonical run, so it has to be stated.
    Generated placement would make it whatever the draw produced -- a number the
    params file does not record and nobody can reproduce from it."""
    start = _start(tmp_path, MODEL_DEMIX)
    p = _params(start, MODEL_DEMIX)
    p["run"].pop("initial_config")
    with pytest.raises(ValueError, match=r"run\.initial_config"):
        sim_setup.from_params(p)


def test_the_log_of_a_canonical_run_carries_its_own_moves(tmp_path):
    start = _start(tmp_path, MODEL_DEMIX)
    params = tmp_path / "canon.json"
    params.write_text(
        json.dumps(_params(start, MODEL_DEMIX, window=2000, target_cycles=6000))
    )
    _state, paths = runner.run(str(params), root=str(tmp_path / "runs"))
    lines = [ln for ln in paths.log.read_text().splitlines() if ln.strip()]
    names = lines[0].lstrip("#").split()
    assert {"mv_acc", "mv_stp", "swp_acc", "swp_stp", "N_A", "N_B"} <= set(names)
    for absent in ("rpk_stp", "clqP_n", "add_stp"):
        assert absent not in names, (
            f"a canonical log declares {absent!r}, which it can never fill"
        )
    rows = [ln.split() for ln in lines[1:]]
    i_a, i_b = names.index("N_A"), names.index("N_B")
    assert len({(r[i_a], r[i_b]) for r in rows}) == 1, (
        "the composition changed between log rows of a fixed-composition run"
    )


def test_the_sampler_is_pinned_because_it_cannot_change_the_chain(tmp_path):
    """Every sampler gives the SAME canonical chain, so only one may be stated.

    This is the measurement behind the refusal, kept as a test because the
    refusal is only justified while it stays true. Five sampler blocks -- two
    channel geometries, rosenbluth, branch at depth 2, the cyclic rod -- produced
    the byte-identical final state from one seed, and five different run ids,
    because `sampler` is an identity field. One physical chain in five
    directories is not cosmetic: a sweep over `trial_density` would have read as
    five settings and been one measurement repeated.

    If a future move set makes the sampler live in this ensemble, this test fails
    first and the refusal above should go with it.
    """
    start = _start(tmp_path, MODEL_DEMIX)
    for block in (
        {
            "type": "channel",
            "trial_density": 10.0,
            "full_repacking": True,
            "channel_length": 5.0,
            "channel_width": 1.0,
        },
        {"type": "rosenbluth", "trials": 20, "trial_box_side": 3.0},
        {"type": "branch", "depth": 2, "trials": 20, "trial_box_side": 3.0},
        {
            "type": "channel_cyclic",
            "trial_density": 2.0,
            "channel_width": 1.2,
            "depth": 1,
        },
    ):
        p = _params(start, MODEL_DEMIX)
        p["sampler"] = block
        with pytest.raises(ValueError, match="runs no trial cloud"):
            sim_setup.from_params(p)


def test_a_canonical_run_does_not_enter_the_metropolis_loop(tmp_path):
    """`sampler.type='metropolis'` names the move set, not `_advance_metropolis`.

    The dispatch in `advance` was on the sampler alone, so pinning the sampler
    sent canonical runs into the single-species metropolis loop, which has no
    branch for the pair exchange and raises `unhandled move choice 'swap'` on the
    first one. It raised on the first run after the pin, which is the good case;
    the bad case would have been a loop that silently dropped the exchange.

    The ensemble decides which loop runs. This asserts the run completes AND that
    composition is still exactly conserved, because a run that reached the
    metropolis loop could do neither.
    """
    start = _start(tmp_path, MODEL_DEMIX)
    state, summaries = _drive(_params(start, MODEL_DEMIX, window=1500), windows=2)
    assert summaries, "the canonical window produced no summary"
    counts = np.bincount(np.asarray(state.species, dtype=np.int64), minlength=2)
    assert counts[0] == counts[1] == N // 2, (
        f"composition moved to {counts.tolist()}; a canonical window cannot "
        "change a species count, and reaching the metropolis loop is how it would"
    )
