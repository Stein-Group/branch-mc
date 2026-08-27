"""The stage-A bit-identity oracle, and what is left of it after stage B.

`tests/ref/traj_*.npz` was generated from the untouched, pre-refactor engine.
Stage A changed WHICH data structure answered a neighbour query, never which
question was asked, which RNG value was drawn, or in what order, so every mode
reproduced those bytes exactly. That is no longer true of `gcmc`: stage B made
particle state index-based, and swap-remove changes which particle a given rng
draw selects. The chains differ; the distributions do not, and stage B's
equivalence is argued in `test_stageB_stats.py`.

So the oracle is now pointed at `tests/ref/engineA`, the frozen stage-A build
that `test_stageB_stats.py` compares against. If that build ever stops
reproducing the original trajectories, the statistical reference has rotted and
every stage-B parity number computed against it is worthless. This test is what
says otherwise.

Two things are still asserted of the CURRENT engine:

  * `metropolis` is bit-identical. Rule 2 keeps `_advance_metropolis` on its
    kd-tree and forbids touching it, precisely so it can be an independent
    reference for the B3 EOS gate. A divergence here means it was touched.
  * Every mode still completes its windows with consistent bookkeeping. Cheap,
    and it catches a mode that quietly stopped running at all.

A divergence in the frozen build has one benign cause and no others: a last-ulp
difference in a distance sitting within ~1e-13 of the overlap threshold, flipping
an accept. The failure message prints the worst-differing row so that claim can
be checked rather than assumed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import _harness as H
import numpy as np
import pytest

from gcmc import MC_engine

# R1: this module's oracle is the frozen stage-A engine, which lives in tag
# v2-engineA rather than in the tree. Skipped as a module when it is absent --
# tier 3 does the `git worktree add` and runs these for real.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _frozen

_frozen.skip_module_without(_frozen.ENGINEA)


def _check_against_reference(mode, engine):
    path = os.path.join(H.REF, f"traj_{mode}.npz")
    if not os.path.exists(path):
        pytest.skip(f"no reference for {mode}; run gen_reference_traj.py")
    ref = np.load(path, allow_pickle=False)

    eng = None if engine == "current" else H.load_engine(engine)
    advance = (MC_engine if eng is None else eng[1]).advance
    cfg, pot, wgh, st = H.make_mode(mode, engine=eng)
    for i in range(H.REF_WINDOWS):
        advance(st, cfg, pot, wgh)
        want = ref[f"w{i}"]
        got = np.asarray(st.positions, float)
        assert got.shape == want.shape, (
            f"{mode} [{engine}]: window {i} has N={got.shape[0]}, "
            f"reference N={want.shape[0]}"
        )
        if not np.array_equal(got, want):
            d = np.abs(got - want)
            k = int(np.argmax(d.max(axis=1)))
            raise AssertionError(
                f"{mode} [{engine}]: window {i} diverged; worst row {k}\n"
                f"  got  {got[k]}\n  want {want[k]}\n"
                f"  max |delta| = {d.max():.3e}"
            )


@pytest.mark.parametrize("mode", sorted(H.REF_MODES))
def test_frozen_stage_a_engine_still_matches_the_oracle(mode):
    """The stage-B statistical reference must still be the engine it claims."""
    _check_against_reference(mode, "stageA")


def test_metropolis_is_bit_identical_in_the_current_engine():
    """`_advance_metropolis` was never converted, and never may be."""
    _check_against_reference("metropolis", "current")


@pytest.mark.parametrize("mode", sorted(H.REF_MODES))
def test_current_engine_still_runs_every_mode(mode):
    cfg, pot, wgh, st = H.make_mode(mode)
    for _ in range(H.REF_WINDOWS):
        stats = MC_engine.advance(st, cfg, pot, wgh)
    assert stats["N"] == len(st.positions)
