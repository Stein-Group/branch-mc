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
sys.path.insert(0, os.path.join(HERE, ".."))

from gcmc.config import from_params           # noqa: E402
from gcmc import parallel, engine             # noqa: E402
from gcmc.state import ChainState, zero_counters  # noqa: E402

MU, K, CAVR, WIN, NW = 1.0, 20, 2.5, 400, 42


def _state(n, hi):
    rng = np.random.default_rng(0)
    return ChainState(positions=rng.uniform(2, hi, size=(n, 2)), step=0,
                      rng=np.random.default_rng(7), counters=zero_counters())


def run_parallel(n_win):
    p = {"potential": {"type": "hard_sphere"},
         "ensemble": {"mu": MU, "dimension": 2, "box": 70.0},
         "trials": {"k": K, "cavity_radius": CAVR}, "algorithm": {"depth": 0},
         "execution": {"mode": "parallel", "n_workers": 1, "sampling_window": WIN,
                       "target_cycles": 1e9, "seed": 1}}
    cfg = from_params(p); pot = cfg.build_potential()
    w = parallel.make_weigher(cfg, pot); st = _state(150, 68)
    etas, last = [], None
    for _ in range(n_win):
        last = parallel.advance_parallel(st, cfg, pot, w)
        etas.append(last["packing_fraction"])
    return etas, last


def run_serial(n_win):
    p = {"potential": {"type": "hard_sphere"},
         "ensemble": {"mu": MU, "dimension": 2, "box": 40.0},
         "trials": {"k": K, "cavity_radius": CAVR}, "algorithm": {"depth": 0},
         "execution": {"mode": "serial", "sampling_window": WIN,
                       "target_cycles": 1e9, "seed": 1}}
    cfg = from_params(p); pot = cfg.build_potential()
    w = engine.make_weigher(cfg, pot); st = _state(60, 38)
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
print(f"  parallel equilib eta = {pmean:.4f}   serial equilib eta = {smean:.4f}   diff = {diff:.4f}")
assert diff < 0.02, f"equilibrium densities disagree by {diff:.4f}"
print("  ok  parallel and serial reach the same equilibrium density")

# 3. 3D parallel raises, as in the original
p3 = {"potential": {"type": "lj"}, "ensemble": {"mu": -4.0, "dimension": 3, "box": 50.0},
      "trials": {"k": 20, "cavity_radius": 1.5},
      "execution": {"mode": "parallel", "n_workers": 1, "sampling_window": 100,
                    "target_cycles": 1e9}}
# lj is 3d; build a hard_sphere 3d instead to hit the guard cleanly
p3["potential"] = {"type": "hard_sphere"}; p3["ensemble"]["mu"] = 1.0
cfg3 = from_params(p3)
st3 = ChainState(positions=np.random.default_rng(0).uniform(2, 48, size=(40, 3)),
                 step=0, rng=np.random.default_rng(1), counters=zero_counters())
try:
    parallel.advance_parallel(st3, cfg3, cfg3.build_potential(),
                              parallel.make_weigher(cfg3, cfg3.build_potential()))
    raise SystemExit("FAIL: 3D parallel did not raise")
except NotImplementedError:
    print("  ok  3D parallel raises NotImplementedError")

print("\nALL PARALLEL TESTS PASSED")
