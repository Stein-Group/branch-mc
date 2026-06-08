"""Identity property checks: determinism, the right sensitivity, and layout.

No original to diff against (identity is new), so this asserts the contract:
identity-bearing params change the id, execution-only params do not, and the
run directory is laid out as expected.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gcmc import config, identity  # noqa: E402

BASE = {
    "potential": {"type": "lj", "T": 0.85},
    "ensemble": {"mu": -4.0, "dimension": 3},
    "trials": {"trial_density": 400, "cavity_radius": 1.5},
    "algorithm": {"depth": 0, "branch": False},
    "execution": {"mode": "serial", "n_workers": 1, "target_cycles": 1e6},
    "move": {"move_factor": 0.25},
}


def cid(mut):
    import copy
    p = copy.deepcopy(BASE)
    for path, val in mut.items():
        sec, key = path.split(".")
        p[sec][key] = val
    return identity.run_id(config.from_params(p))


base_id = cid({})
print("base id:", base_id)

# determinism
assert cid({}) == base_id, "id not deterministic"
print("  ok  deterministic")

# identity-bearing params must change the id
for path, val in [("ensemble.mu", -3.5), ("ensemble.box", 25.0),
                  ("trials.trial_density", 500), ("trials.cavity_radius", 2.0),
                  ("algorithm.depth", 1), ("algorithm.branch", True),
                  ("move.move_factor", 0.3), ("potential.T", 0.9)]:
    assert cid({path: val}) != base_id, f"id should change with {path}"
    print(f"  ok  id changes with {path}")

# execution-only params must NOT change the id
for path, val in [("execution.target_cycles", 9e9), ("execution.n_workers", 16),
                  ("execution.mode", "parallel")]:
    # mode=parallel would change the default box, so pin box to isolate mode
    import copy
    p = copy.deepcopy(BASE)
    p["ensemble"]["box"] = 20.0
    sec, key = path.split(".")
    p[sec][key] = val
    p0 = copy.deepcopy(BASE)
    p0["ensemble"]["box"] = 20.0
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
