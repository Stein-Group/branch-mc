"""Two launchers sharing a launch_dir must not hand each other's params over.

The snapshot used to be `params_<label>.json`, and the label names only the
swept axes. Two jobs from different base files at the same mu wrote the same
file, and whichever child read it after the second write ran the other job's
physics under its own --root.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gcmc.run import launcher


def _base(sampler: dict) -> dict:
    return {
        "potential": {"main": "hard_core"},
        "sampler": sampler,
        "engine": {"depth": 1},
        "ensemble": {"type": "muVT", "dimension": 3, "box": 8.0, "mu": [15.0]},
        "run": {
            "kind": "single",
            "radius": 0.5,
            "seed": 1,
            "traj_every": 0,
            "target_cycles": 10,
            "sampling_window": 5,
            "checkpoint_every": 1,
        },
    }


def test_same_label_different_base_gets_its_own_snapshot(tmp_path, monkeypatch):
    seen: list[str] = []

    class _Fake:
        def __init__(self, cmd, env=None):
            seen.append(cmd[cmd.index("run") + 1])

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    a = _base({"type": "branch", "trials": 30, "trial_box_side": 3.0})
    b = _base({"type": "branch", "trials": 60, "trial_box_side": 2.0})
    ldir = str(tmp_path / "launch")
    launcher.launch_sweep(a, 1, root="ra", launch_dir=ldir, pin=False, wait=False)
    launcher.launch_sweep(b, 1, root="rb", launch_dir=ldir, pin=False, wait=False)

    assert len(seen) == 2 and seen[0] != seen[1], seen
    got = [json.loads(Path(p).read_text())["sampler"]["trials"] for p in seen]
    assert got == [30, 60], f"a job was handed the other job's params: {got}"
