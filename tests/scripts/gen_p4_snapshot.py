"""Freeze the CURRENT engine's chains, so P4 can be checked as a pure refactor.

P4 of multispecies_plan.md factors the channel proposal out of `advance` so that
`advance_cell` can invoke the same callable instead of carrying a second copy.
Its gate is bit-identity, and the existing `tests/ref/traj_*.npz` cannot serve:
those were generated before stage B made particle state index-based, and
`test_stageA_traj_identity` already records that only `metropolis` still
reproduces them. Comparing a refactor against a reference the code has already
diverged from proves nothing.

So the oracle is the engine comparing against ITSELF across the refactor. Run
this once on the pre-refactor tree, commit the output, refactor, and let
`tests/engine/test_p4_refactor_identity.py` require the same bytes. It is only
meaningful if the snapshot predates the change, which is why it is a separate
script and not a fixture that regenerates on demand.

    python tests/scripts/gen_p4_snapshot.py            # write the snapshot
    python tests/scripts/gen_p4_snapshot.py --verify   # re-run and compare

Every mode in `_harness.REF_MODES` is covered, not just the channel ones: a
refactor that moves shared code can break the branch, rosenbluth or metropolis
path just as easily, and the plan's gate names all of them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness as H
import numpy as np

from gcmc.engine import metropolis_dynamics

SNAPSHOT = os.path.join(H.REF, "p4_snapshot.npz")

# Enough windows to exercise accept AND reject on every branch of the move loop,
# few enough that the whole sweep is a couple of minutes. Fixed per mode so the
# snapshot is reproducible from this file alone.
N_WINDOWS = 4


def _run(mode: str) -> dict[str, np.ndarray]:
    """One mode, N_WINDOWS windows, returning the final configuration and the
    per-window counter stream.

    The counter stream matters as much as the coordinates: two chains can end on
    the same configuration having taken different routes there, and a refactor
    that changes which proposals were attempted is not a refactor.
    """
    cfg, pot, wgh, st = H.make_mode(mode)
    rows: list[list[float]] = []
    keys: list[str] | None = None
    for _ in range(N_WINDOWS):
        summary = metropolis_dynamics.advance(st, cfg, pot, wgh)
        if keys is None:
            keys = sorted(k for k, v in summary.items() if isinstance(v, (int, float)))
        rows.append([float(summary[k]) for k in keys])
    pos = np.asarray(st.positions, float)
    return {
        f"{mode}__positions": pos,
        f"{mode}__counters": np.asarray(rows, float),
        f"{mode}__keys": np.asarray(keys or [], dtype="U40"),
    }


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--verify",
        action="store_true",
        help="re-run and compare against the committed snapshot instead of writing",
    )
    ap.add_argument(
        "--modes",
        default=None,
        help="comma-separated subset of modes; default every mode in REF_MODES",
    )
    a = ap.parse_args(argv)

    modes = a.modes.split(",") if a.modes else list(H.REF_MODES)
    out: dict[str, np.ndarray] = {}
    for m in modes:
        out.update(_run(m))
        print(
            f"  {m:<20} N = {len(out[f'{m}__positions']):4d}"
            f"  pos {_digest(out[f'{m}__positions'])}"
            f"  ctr {_digest(out[f'{m}__counters'])}"
        )

    if not a.verify:
        np.savez(SNAPSHOT, **out)
        print(f"\nwrote {SNAPSHOT} ({len(modes)} modes, {N_WINDOWS} windows each)")
        return 0

    if not os.path.exists(SNAPSHOT):
        print(f"no snapshot at {SNAPSHOT}; run without --verify first")
        return 2
    ref = np.load(SNAPSHOT, allow_pickle=False)
    bad = []
    for key, got in out.items():
        if key not in ref:
            bad.append(f"{key}: absent from the snapshot")
            continue
        want = ref[key]
        if got.shape != want.shape or got.tobytes() != want.tobytes():
            bad.append(f"{key}: {want.shape} -> {got.shape}, bytes differ")
    if bad:
        print("\nP4 IDENTITY FAILED:")
        for line in bad:
            print("  " + line)
        return 1
    print(f"\nP4 identity holds for {len(modes)} modes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
