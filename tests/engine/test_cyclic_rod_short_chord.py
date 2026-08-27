"""The cyclic rod's per-move trial count must hold the move it is proposing.

THE DEFECT THIS PINS. `ChannelBody.build` recomputes the trial count for the
wall-to-wall rod from the chord where the line meets the box, floored at a
literal 2. The floor was `depth + 1` at depth 1 and was never revisited. An
oblique line clipped to the box leaves an arbitrarily short chord -- the slab
intersection goes to zero as the line grazes a corner -- so at depth >= 2 the
count fell below the group size and the engine died in

    keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False
    ValueError: Cannot take a larger sample than population when replace is False

on roughly 3% of exchange proposals, in 2D and in 3D alike. Three percent is not
a tail: it is a crash within the first few dozen proposals, which means the
cyclic rod at depth >= 2 had never been run by anything. What let it hide is that
the validator's `n_trials >= depth + 1` check reads the CONFIGURED count, and
that check is the reason the plain depth path has no runtime guard of its own --
so the one sampler that recomputes the count per move was also the one sampler
whose guarantee had quietly stopped holding.

Two gates, because either alone is escapable. The first is arithmetic on the body
and needs no engine: it drives the geometry into short chords deliberately, which
random sampling would only reach a few percent of the time. The second runs the
engine long enough that those orientations are certain to occur -- because the
first would still pass if the engine stopped passing the right floor.
"""

from __future__ import annotations

import _harness as H
import numpy as np
import pytest

from gcmc import MC_engine, channel_move

_BOX = 12.0


def _short_chord_setups(dim: int, n: int = 4000):
    """(center, rotation) pairs biased hard toward grazing lines.

    Centres are pressed against a corner and orientations are drawn from the
    sampler's own `draw_rotation`, so the chord distribution here is the tail of
    the real one rather than a different distribution. Nothing is asserted about
    how short they get; the point is only that short ones are reached.
    """
    rng = np.random.default_rng(20260824)
    box = np.full(dim, _BOX)
    out = []
    for _ in range(n):
        # within 0.05 sigma of a corner, where the slab intersection is thinnest
        center = rng.uniform(0.0, 0.05, dim) + rng.integers(0, 2, dim) * (_BOX - 0.05)
        R = channel_move.draw_rotation(rng, dim, channel=True)
        out.append((np.clip(center, 0.0, _BOX), R, box))
    return out


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("group", (1, 2, 3, 4, 5))
def test_the_body_never_returns_fewer_trials_than_the_group_needs(dim, group) -> None:
    """`n_trials >= n_trials_min`, over thousands of grazing orientations.

    `group` spans depth 0 through depth 4 (the matrix's own depth axis), so the
    old literal floor of 2 is covered as the depth-1 case and fails here for
    every larger one.
    """
    worst = None
    reached_the_floor = 0
    for center, R, box in _short_chord_setups(dim):
        body = channel_move.ChannelBody.build(
            center,
            R,
            box,
            box,
            dim,
            cyclic=True,
            channel_length=0.0,  # the cyclic rod's length comes from the chord
            channel_width=1.2,
            trial_density=0.6,
            n_trials_base=12,
            n_trials_min=group,
        )
        if worst is None or body.n_trials < worst:
            worst = body.n_trials
        if body.n_trials == group:
            reached_the_floor += 1
        assert body.n_trials >= group, (
            f"{dim}D: a chord gave {body.n_trials} trial slots for a group of "
            f"{group}; the mirror needs one slot per removed molecule"
        )
    assert reached_the_floor > 0, (
        f"{dim}D group {group}: no setup was short enough to reach the floor, so "
        "this run proves nothing. The corner-biased generator has stopped "
        "generating grazing lines."
    )
    assert worst == group  # the floor is attained, not merely respected


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("depth", (2, 3))
def test_the_cyclic_rod_runs_at_depth_two_and_three(dim, depth) -> None:
    """The engine survives the orientations that used to kill it.

    Sized from the measured rate: at depth 2 about 3% of exchange proposals hit a
    chord too short for the old floor, so ~600 proposals put the probability of
    NOT hitting one at about 1e-8. A pass here is therefore evidence, not luck.
    It asserts nothing about the physics -- that is the EOS and detailed-balance
    suites' job -- only that a supported configuration completes.
    """
    params = {
        "potential": {"main": "lennard_jones", "T": 0.85},
        "sampler": {
            "type": "channel_cyclic",
            "depth": depth,
            "trial_density": 0.6,
            "channel_width": 1.2,
        },
        "ensemble": {"mu": -3.2, "dimension": dim, "box": _BOX},
        "run": {"sampling_window": 200},
    }
    n_seed = 120 if dim == 2 else 400
    pos = H.lattice_seed(_BOX, n_seed, dim, np.random.default_rng(7))
    cfg, potential, weigher, state = H.build(params, 7, positions=pos)
    exchanges = 0
    for _ in range(3):
        c = MC_engine.advance(state, cfg, potential, weigher)["counters"]
        exchanges += c["add_step"] + c["remove_step"] + c["repack_step"]
    assert exchanges >= 500, (
        f"{dim}D depth {depth}: only {exchanges} exchange proposals ran, too few "
        "for the ~3% short-chord rate to be probed. Raise sampling_window."
    )
