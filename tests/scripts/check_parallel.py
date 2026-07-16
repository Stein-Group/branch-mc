"""Parallel-engine checks.

The parallel path cannot be traced exactly against the original (its cavities
come from an unseedable default_rng), so it is validated by the physical
invariant a decomposition must preserve: at a fixed chemical potential the
equilibrium packing fraction is intensive, so the domain-decomposed run and the
single-box serial run must converge to the same eta despite different box sizes,
different cell counts and different starting densities. A bug in acceptance,
cell-local wrapping or the cell-area volume term would shift that plateau.

Also checks that the engine runs and stays populated, and that 3D parallel
raises, matching the original.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from gcmc import MC_engine as engine
from gcmc import parallel
from gcmc.sim_setup import from_params
from gcmc.state import ChainState, zero_counters

MU, K, CAVR, WIN, NW = 1.0, 20, 2.5, 400, 42


def _state(n, hi):
    rng = np.random.default_rng(0)
    return ChainState(
        positions=rng.uniform(2, hi, size=(n, 2)),
        step=0,
        rng=np.random.default_rng(7),
        counters=zero_counters(),
    )


def run_parallel(n_win):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": 2, "box": 70.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": K,
            "trial_box_side": 2 * CAVR,
            "depth": 0,
        },
        "run": {
            "kind": "parallel",
            "n_workers": 1,
            "sampling_window": WIN,
            "target_cycles": 1e9,
            "seed": 1,
        },
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = parallel.make_weigher(cfg, pot)
    st = _state(150, 68)
    etas, last = [], None
    for _ in range(n_win):
        last = parallel.advance_parallel(st, cfg, pot, w)
        etas.append(last["packing_fraction"])
    return etas, last


def run_serial(n_win):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": 2, "box": 40.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": K,
            "trial_box_side": 2 * CAVR,
            "depth": 0,
        },
        "run": {
            "kind": "single",
            "sampling_window": WIN,
            "target_cycles": 1e9,
            "seed": 1,
        },
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = engine.make_weigher(cfg, pot)
    st = _state(60, 38)
    etas = []
    for _ in range(n_win):
        etas.append(engine.advance(st, cfg, pot, w)["packing_fraction"])
    return etas


# 1. runs and stays populated; gridded into >1 cell
ep, last = run_parallel(NW)
assert last["n_cells"] > 1, "expected the box to grid into several cells"
assert last["N"] > 0, "parallel run depopulated"
print(f"  ok  parallel-2D runs over {last['n_cells']} cells, stays populated")

# 2. equilibrium packing fraction agrees with the serial path
es = run_serial(NW)
pmean = float(np.mean(ep[-12:]))
smean = float(np.mean(es[-12:]))
diff = abs(pmean - smean)
print(
    f"  parallel equilib eta = {pmean:.4f}   serial equilib eta = {smean:.4f}   diff = {diff:.4f}"
)
assert diff < 0.02, f"equilibrium densities disagree by {diff:.4f}"
print("  ok  parallel and serial reach the same equilibrium density")

# 3. 3D parallel is rejected up front by the schema (parallel is 2D only)
p3 = {
    "potential": {"main": "hard_core"},
    "ensemble": {"mu": 1.0, "dimension": 3, "box": 50.0},
    "sampler": {"type": "rosenbluth", "trials": 20, "trial_box_side": 3.0},
    "run": {
        "kind": "parallel",
        "n_workers": 1,
        "sampling_window": 100,
        "target_cycles": 1e9,
    },
}
try:
    from_params(p3)
    raise SystemExit("FAIL: 3D parallel did not raise")
except ValueError as e:
    assert "2D only" in str(e), str(e)
    print("  ok  3D parallel rejected at config:", e)

# 4. a non-rosenbluth sampler under parallel is rejected by the engine guard
p4 = {
    "potential": {"main": "hard_core"},
    "ensemble": {"mu": MU, "dimension": 2, "box": 40.0},
    "sampler": {"type": "branch", "trials": K, "trial_box_side": 2 * CAVR},
    "run": {
        "kind": "parallel",
        "n_workers": 1,
        "sampling_window": 50,
        "target_cycles": 1e9,
    },
}
cfg4 = from_params(p4)
st4 = _state(40, 38)
try:
    parallel.advance_parallel(
        st4,
        cfg4,
        cfg4.build_potential(),
        parallel.make_weigher(cfg4, cfg4.build_potential()),
    )
    raise SystemExit("FAIL: parallel+branch did not raise")
except NotImplementedError as e:
    print("  ok  parallel+branch raises NotImplementedError:", str(e)[:60])

print("\nALL PARALLEL TESTS PASSED")
