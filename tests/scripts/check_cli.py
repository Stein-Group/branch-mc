"""CLI checks: `python -m gcmc run` produces a complete run, and the launcher
fans a mu sweep into one distinct directory per mu.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.join(HERE, "..", "..")
sys.path.insert(0, PKG_PARENT)
from gcmc import launcher

ENV = dict(os.environ)
ENV["PYTHONPATH"] = PKG_PARENT + os.pathsep + ENV.get("PYTHONPATH", "")

BASE = {
    "potential": {"main": "hard_core"},
    "ensemble": {"mu": 3.0, "dimension": 3},
    "sampler": {"type": "rosenbluth", "trials": 20, "trial_box_side": 3.0, "depth": 0},
    "run": {
        "sampling_window": 200,
        "target_cycles": 400,
        "checkpoint_every": 1,
        "traj_every": 1,
        "seed": 3,
    },
}

with tempfile.TemporaryDirectory() as tmp:
    # 1. python -m gcmc run params.json
    pf = os.path.join(tmp, "params.json")
    with open(pf, "w") as f:
        json.dump(BASE, f)
    root1 = os.path.join(tmp, "r1")
    r = subprocess.run(
        [sys.executable, "-m", "gcmc", "run", pf, "--root", root1],
        env=ENV,
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"run failed:\n{r.stderr}"
    assert "done:" in r.stdout, r.stdout
    dirs = glob.glob(os.path.join(root1, "*"))
    assert len(dirs) == 1, dirs
    assert os.path.exists(os.path.join(dirs[0], "checkpoint.npz"))
    assert os.path.exists(os.path.join(dirs[0], "run.log"))
    print("  ok  `gcmc run` produced a complete run:", os.path.basename(dirs[0]))

    # 2. launcher sweep over three mu values -> three distinct directories
    import copy as _copy

    base2 = _copy.deepcopy(BASE)
    # `ensemble.sweep` was the OLD way to name a swept axis. `expand_sweep` now
    # sweeps any list-valued axis directly, and `from_params` rejects a params
    # dict carrying both, so the old key is gone rather than merely unused.
    base2["ensemble"] = {"dimension": 3}
    bf = os.path.join(tmp, "base.json")
    with open(bf, "w") as f:
        json.dump(base2, f)
    root2 = os.path.join(tmp, "r2")
    ldir = os.path.join(tmp, "launch")
    # run launcher in-process so it inherits PYTHONPATH for the children
    os.environ["PYTHONPATH"] = ENV["PYTHONPATH"]
    # API, as of the modernization: the swept axis lives IN the params dict --
    # any list-valued axis is expanded by `expand_sweep` -- and the second
    # positional argument is the CORE BUDGET, not a value list. check_cli drove
    # the old signature and had been xfailed ever since; updated in S1.
    #
    # Three cores so the three mu values run concurrently, which is also what
    # exercises the scheduler rather than just the expansion.
    base2["ensemble"]["mu"] = [3.0, 3.25, 3.5]
    rcs = launcher.launch_sweep(
        base2, 3, root=root2, launch_dir=ldir, pin=False, wait=True
    )
    assert rcs == [0, 0, 0], f"sweep return codes {rcs}"
    run_dirs = [d for d in glob.glob(os.path.join(root2, "*")) if os.path.isdir(d)]
    assert len(run_dirs) == 3, f"expected 3 run dirs, got {len(run_dirs)}"
    for d in run_dirs:
        assert os.path.exists(os.path.join(d, "checkpoint.npz")), d
    # the three per-mu params files were written
    # `params_<label>.json`, where the label is the short axis suffix the run
    # directory also carries (e.g. `mu3p5`). The old `params_mu_<value>.json`
    # convention went with the old sweep API.
    snapshots = glob.glob(os.path.join(ldir, "params_*.json"))
    assert len(snapshots) == 3, f"expected 3 params snapshots, got {snapshots}"
    print(
        "  ok  launcher produced 3 distinct runs:",
        sorted(os.path.basename(d) for d in run_dirs),
    )

print("\nALL CLI/LAUNCHER TESTS PASSED")
