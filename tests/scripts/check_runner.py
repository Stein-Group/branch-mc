"""Runner checks: a full run writes its files and hits the target; re-entering a
finished run is a no-op; a one-shot run and a checkpointed restart reach an
identical final configuration (the RNG-continuity test); the lock refuses a
second concurrent holder.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.run import hash_fingerprint as identity
from gcmc.run import io, runner
from gcmc.run import sim_setup as config


def base(target):
    return {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 3.0, "dimension": 3},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
        },
        "engine": {"depth": 0},
        "run": {
            "sampling_window": 200,
            "target_cycles": target,
            "checkpoint_every": 1,
            "traj_every": 1,
            "seed": 7,
        },
    }


with tempfile.TemporaryDirectory() as tmp:
    # 1. a full run writes its files and reaches the target
    root1 = os.path.join(tmp, "r1")
    state, paths = runner.run(base(1200), root=root1)
    for f in (paths.checkpoint, paths.log, paths.trajectory, paths.params):
        assert os.path.exists(f), f"missing {f}"
    assert state.step >= 1200, "did not reach target"
    assert not os.path.exists(paths.lock), "lock not released"
    print("  ok  full run writes files and reaches target, step =", state.step)

    # 2. re-entering a finished run is a no-op
    step_before = state.step
    state2, _ = runner.run(base(1200), root=root1)
    assert state2.step == step_before, "finished run should not advance"
    print("  ok  finished run re-entry is a no-op")

    # 3. RNG continuity: one-shot to 1200 vs 600-then-1200 restart -> identical
    one = os.path.join(tmp, "one")
    s_one, p_one = runner.run(base(1200), root=one)

    two = os.path.join(tmp, "two")
    runner.run(base(600), root=two)  # first leg, checkpoints at 600
    s_two, p_two = runner.run(base(1200), root=two)  # resume to 1200

    assert s_one.step == s_two.step == 1200, (s_one.step, s_two.step)
    assert s_one.positions.shape == s_two.positions.shape, (
        f"N differs: {s_one.positions.shape} vs {s_two.positions.shape}"
    )
    assert np.allclose(s_one.positions, s_two.positions), (
        "restart did not reproduce the one-shot configuration"
    )
    print("  ok  restart reproduces one-shot exactly (N =", len(s_one.positions), ")")

    # 4. the restarted log matches the one-shot log row for row
    log_one, log_two = io.read_log(p_one.log), io.read_log(p_two.log)
    assert log_one.keys() == log_two.keys()
    for k in log_one:
        if k != "wall_s":
            assert np.array_equal(log_one[k], log_two[k]), f"{k} differs after restart"
    print("  ok  log continuous across restart")

    # 5. lock refuses a second holder
    root5 = os.path.join(tmp, "r5")
    cfg = config.from_params(base(1200))
    p5 = identity.run_paths(cfg, root5)
    os.makedirs(p5.directory, exist_ok=True)
    runner.acquire_lock(p5)
    try:
        runner.acquire_lock(p5)
        raise SystemExit("lock should have refused a second holder")
    except RuntimeError:
        print("  ok  lock refuses a second holder")
    runner.release_lock(p5)

print("\nALL RUNNER TESTS PASSED")
