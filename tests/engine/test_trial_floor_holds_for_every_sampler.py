"""The depth path's trial-count floor is a property of the PATH, not of the
sampler that happens to be named.

`advance` places a depth-d move by claiming `d + 1` trial slots --
`rng.choice(n_trials, size=n_remove, replace=False)` -- so a pool smaller than
that cannot be drawn from and numpy raises. The validator has always refused it
for `branch`, `channel` and `channel_cyclic`, and `rosenbluth` was exempt while
taking exactly the same path, so an accepted rosenbluth config could not survive
its first exchange proposal.

This is the second instance of one shape in this stage. The cyclic rod's floor
(S4 phase 2) was a count recomputed per move and floored at a literal 2; this one
is a validator check whose condition names three of the four samplers it covers.
Both crash in the same numpy call, and neither was reachable from the feature
matrix because every generated point carries a comfortable trial pool.

The pairing below is the test: for each sampler and depth, the validator's
verdict and the engine's survival must AGREE. A test that only asserted "the
validator refuses trials <= depth" would keep passing if the engine were changed
to tolerate a short pool, and the two would drift; asserting the agreement is
what makes it a gate on the relationship rather than on either side.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import MC_engine, config_specs, runner, sim_setup
from gcmc.state import ChainState

_BASE = {
    "potential": {"main": "lennard_jones", "T": 1.5},
    "ensemble": {"mu": -2.0, "dimension": 3, "box": 10.0, "type": "muVT"},
    "run": {
        "kind": "single",
        "seed": 4,
        "sampling_window": 300,
        "initial_config": "grid",
    },
}


def _params(sampler: str, depth: int, trials: int) -> dict:
    p = {k: dict(v) for k, v in _BASE.items()}
    p["sampler"] = {
        "type": sampler,
        "depth": depth,
        "trials": trials,
        "trial_box_side": 3.0,
    }
    if sampler in ("channel", "channel_cyclic"):
        p["sampler"]["channel_width"] = 1.5
        p["sampler"]["channel_length"] = 3.0
        p["sampler"].pop("trial_box_side")
    return p


@pytest.mark.parametrize("sampler", ["rosenbluth", "branch", "channel"])
@pytest.mark.parametrize(("depth", "trials"), [(1, 1), (2, 2), (2, 3), (3, 4)])
def test_the_validator_and_the_engine_agree_about_a_short_trial_pool(
    sampler, depth, trials
):
    params = _params(sampler, depth, trials)
    try:
        cfg = sim_setup.from_params(params)
    except ValueError as exc:
        assert "trial spheres" in str(exc), (
            f"{sampler} depth {depth} trials {trials} was refused for some other "
            f"reason: {exc}"
        )
        assert trials < depth + 1, (
            f"{sampler} refused a pool of {trials} for depth {depth}, which is "
            "large enough to place the move"
        )
        return

    assert trials >= depth + 1, (
        f"{sampler} depth {depth} accepted a pool of {trials}; a depth-{depth} "
        f"move claims {depth + 1} slots and numpy cannot draw them"
    )
    pot = cfg.build_potential()
    weigher = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    state = ChainState.fresh(3, seed=9)
    state.positions = runner._init_config(cfg, state.rng)
    state.species = runner._init_species(cfg, len(state.positions))
    for _ in range(3):
        MC_engine.advance(state, cfg, pot, weigher)
    assert len(state.positions) > 0


def test_the_message_names_the_sampler_that_was_asked_for():
    """It said "branch/channel" for every sampler, which is a poor thing to read
    under `rosenbluth` -- the reader's first move is to check whether the message
    is even about their run."""
    with pytest.raises(ValueError, match="rosenbluth needs at least"):
        sim_setup.from_params(_params("rosenbluth", 2, 2))


def test_a_short_pool_really_does_crash_the_engine_when_the_guard_is_bypassed():
    """The reachable-value check on the guard itself.

    Everything above would pass if the floor were merely conservative and the
    engine coped. It does not: building the same config with the trial count
    lowered AFTER validation reproduces the numpy error the guard exists to
    prevent, so the refusal is protecting against a real crash rather than
    stating a preference.
    """
    cfg = sim_setup.from_params(_params("rosenbluth", 1, 2))
    object.__setattr__(cfg, "no_of_trial_molecules", 1)
    pot = cfg.build_potential()
    weigher = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    state = ChainState.fresh(3, seed=9)
    state.positions = runner._init_config(cfg, state.rng)
    with pytest.raises(ValueError, match="larger sample than population"):
        for _ in range(6):
            MC_engine.advance(state, cfg, pot, weigher)
    assert np.isfinite(state.positions).all()
