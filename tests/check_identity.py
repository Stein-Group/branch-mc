"""Identity property checks: determinism, the right sensitivity, and layout.

No original to diff against (identity is new), so this asserts the contract:
identity-bearing params change the id, execution-only params do not, and the
run directory is laid out as expected.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gcmc import sim_setup as config, hash_fingerprint as identity  # noqa: E402

BASE = {
    "potential": {"main": "hard_core"},
    "ensemble": {"mu": 4.0, "dimension": 2, "box": 75.0},
    "sampler": {"type": "rosenbluth", "trial_density": 400, "trial_box_side": 3.0, "depth": 0},
    "run": {"target_cycles": 1e6, "move_factor": 0.25},
}

LJ_BASE = {
    "potential": {"main": "lennard_jones", "T": 0.85},
    "ensemble": {"mu": -4.0, "dimension": 3},
    "sampler": {"type": "rosenbluth", "trial_density": 400, "trial_box_side": 3.0},
}


def cid(mut, base=BASE):
    import copy
    p = copy.deepcopy(base)
    for path, val in mut.items():
        sec, key = path.split(".")
        p.setdefault(sec, {})[key] = val
    return identity.run_id(config.from_params(p))


base_id = cid({})
print("base id:", base_id)

assert cid({}) == base_id, "id not deterministic"
print("  ok  deterministic")

# identity-bearing params must change the id
for path, val in [("ensemble.mu", 3.5), ("ensemble.box", 25.0),
                  ("sampler.trial_density", 500), ("sampler.trial_box_side", 4.0),
                  ("sampler.depth", 1), ("sampler.type", "branch"),
                  ("run.move_factor", 0.3)]:
    assert cid({path: val}) != base_id, f"id should change with {path}"
    print(f"  ok  id changes with {path}")

# temperature is identity-bearing on an LJ run
assert cid({"potential.T": 0.9}, base=LJ_BASE) != cid({}, base=LJ_BASE), "T should change id"
print("  ok  id changes with potential.T (LJ)")

# execution-only params must NOT change the id
for path, val in [("run.target_cycles", 9e9), ("run.n_workers", 16),
                  ("run.kind", "parallel")]:
    import copy
    p = copy.deepcopy(BASE)            # 2D base: parallel is valid, box pinned
    sec, key = path.split(".")
    p.setdefault(sec, {})[key] = val
    p0 = copy.deepcopy(BASE)
    assert identity.run_id(config.from_params(p)) == identity.run_id(config.from_params(p0)), \
        f"id should NOT change with {path}"
    print(f"  ok  id stable under {path}")

# directory layout
cfg = config.from_params(BASE)
rp = identity.run_paths(cfg, root="runs")
assert rp.directory.startswith(os.path.join("runs", base_id)), rp.directory
assert rp.checkpoint.endswith("checkpoint.npz")
assert rp.trajectory.endswith("trajectory.extxyz")
assert rp.params.endswith("params.json")
print("  ok  run_paths layout:", rp.directory)

print("\nALL IDENTITY TESTS PASSED")
