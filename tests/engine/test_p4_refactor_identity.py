"""P4 of multispecies_plan.md: the channel extraction must move no bits.

`tests/ref/p4_snapshot.npz` is the current engine's own output, frozen on the
pre-refactor tree by `tests/scripts/gen_p4_snapshot.py`. The existing
`traj_*.npz` references cannot serve this purpose -- stage B made particle state
index-based and `test_stageA_traj_identity` records that only `metropolis` still
reproduces them -- so a refactor has to be checked against the engine as it was
immediately before, not against a reference it had already diverged from.

Both halves of the comparison matter. The final configuration says the chain
ended in the same place; the per-window counter stream says it took the same
route to get there. A change that alters which proposals were attempted while
landing on the same coordinates is not a refactor, and only the counters catch
it.

Fast enough for the default tier: fourteen modes, four windows each, about five
seconds in total. That is deliberate. A bit-identity gate is only useful if it
is cheap enough to run after every edit, and the whole point of P4 is to make a
sequence of edits to the hardest function in the codebase.
"""

from __future__ import annotations

import os

import _harness as H
import numpy as np
import pytest
from gen_p4_snapshot import N_WINDOWS, SNAPSHOT

from gcmc.engine import MC_engine


@pytest.fixture(scope="module")
def snapshot():
    if not os.path.exists(SNAPSHOT):
        pytest.skip(
            f"no {SNAPSHOT}; generate it on the pre-refactor tree with "
            "python tests/scripts/gen_p4_snapshot.py"
        )
    return np.load(SNAPSHOT, allow_pickle=False)


@pytest.mark.parametrize("mode", list(H.REF_MODES))
def test_the_chain_is_bit_identical_across_the_refactor(mode, snapshot):
    cfg, pot, wgh, st = H.make_mode(mode)
    keys = [str(k) for k in snapshot[f"{mode}__keys"]]
    rows = []
    for _ in range(N_WINDOWS):
        summary = MC_engine.advance(st, cfg, pot, wgh)
        rows.append([float(summary[k]) for k in keys])

    want_ctr = snapshot[f"{mode}__counters"]
    got_ctr = np.asarray(rows, float)
    assert got_ctr.shape == want_ctr.shape, f"{mode}: counter stream changed shape"
    if got_ctr.tobytes() != want_ctr.tobytes():
        w = int(np.argmax(np.abs(got_ctr - want_ctr).max(axis=1)))
        diff = {
            k: (want_ctr[w, i], got_ctr[w, i])
            for i, k in enumerate(keys)
            if want_ctr[w, i] != got_ctr[w, i]
        }
        pytest.fail(f"{mode}: counters diverge first in window {w}: {diff}")

    want_pos = snapshot[f"{mode}__positions"]
    got_pos = np.asarray(st.positions, float)
    assert got_pos.shape == want_pos.shape, (
        f"{mode}: final N is {len(got_pos)}, was {len(want_pos)}"
    )
    if got_pos.tobytes() != want_pos.tobytes():
        d = np.abs(got_pos - want_pos).max(axis=1)
        r = int(np.argmax(d))
        pytest.fail(
            f"{mode}: positions diverge, worst row {r} by {d[r]:.3e}: "
            f"{want_pos[r]} -> {got_pos[r]}"
        )
