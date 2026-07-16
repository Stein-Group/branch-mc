"""Freeze per-window positions for every engine mode, from the UNTOUCHED code.

    python tests/gen_reference_traj.py            # write any missing refs
    python tests/gen_reference_traj.py --force    # rewrite (do not do this)

Writes tests/ref/traj_<mode>.npz with one `w<i>` array per window plus the final
counter dict. These are the stage-A oracles for `test_stageA_traj_identity.py`
and must never be regenerated after an engine edit: the whole point is that the
converted engine reproduces them bit for bit. The generator refuses to overwrite
an existing file unless --force is passed, so an accidental rerun is a no-op.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H

from gcmc import MC_engine


def run_mode(name):
    """Return (list of per-window positions, list of per-window stats)."""
    cfg, pot, wgh, st = H.make_mode(name)
    frames, stats = [], []
    for _ in range(H.REF_WINDOWS):
        s = MC_engine.advance(st, cfg, pot, wgh)
        frames.append(np.array(st.positions, dtype=np.float64, copy=True))
        stats.append(
            {
                "N": int(s["N"]),
                "n_cav": float(s["n_cav"]),
                "counters": dict(s["counters"]),
            }
        )
    return frames, stats


def main(argv):
    force = "--force" in argv
    os.makedirs(H.REF, exist_ok=True)
    for name in H.REF_MODES:
        path = os.path.join(H.REF, f"traj_{name}.npz")
        if os.path.exists(path) and not force:
            print(f"  keep   {name}")
            continue
        frames, stats = run_mode(name)
        np.savez(
            path, meta=json.dumps(stats), **{f"w{i}": f for i, f in enumerate(frames)}
        )
        print(
            f"  write  {name:18s} N per window: "
            + ", ".join(str(len(f)) for f in frames)
        )
    print(f"reference trajectories in {H.REF}")


if __name__ == "__main__":
    main(sys.argv[1:])
