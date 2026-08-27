"""Parallel-engine checks.

The parallel path cannot be traced exactly against the original (its cavities
come from an unseedable default_rng), so it is validated by the physical
invariant a decomposition must preserve: at a fixed chemical potential the
equilibrium packing fraction is intensive, so the domain-decomposed run and the
single-box serial run must converge to the same eta despite different box sizes,
different cell counts and different starting densities. A bug in acceptance,
cell-local wrapping or the cell-area volume term would shift that plateau.

Also checks that the engine runs and stays populated, that 3D parallel runs for
every supported sampler (the original's "parallel is 2D only" guard was dropped
by decision: cell_unify_plan.md U0, GO 2026-07-14; 3D physics is validated on
the cluster in U3), and that the mixes which stay serial-only are still
rejected at config time.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from gcmc import MC_engine as engine
from gcmc import config_specs, parallel
from gcmc.sim_setup import from_params
from gcmc.state import ChainState, zero_counters

# One cell is one core, so cfg.n_workers is the cell count; without this
# hook a 9- or 27-cell case would spawn that many processes inside a test.
parallel.IN_PROCESS = True

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
            # manual geometry ON PURPOSE: this section's eta parity is pinned to
            # measured numbers, so the decomposition must stay exactly what it
            # was when they were measured (the old cell_length/buffer defaults).
            "cell_length": 30.0,
            "buffer": 2.0,
            "sampling_window": WIN,
            "target_cycles": 1e9,
            "seed": 1,
        },
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = parallel.make_weigher(config_specs.kernel_spec(cfg), pot)
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
    w = engine.make_weigher(config_specs.kernel_spec(cfg), pot)
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


# 3. 3D parallel is supported (2D-only constraint dropped: cell_unify_plan.md
#    U0, Dipesh GO 2026-07-14). metropolis, rosenbluth, and branch all build
#    in 3D and run a live window over several cells without depopulating.
def _run3d(sampler_block, n_win=4):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 2.0, "dimension": 3, "box": 24.0},
        "sampler": sampler_block,
        "run": {
            "kind": "parallel",
            "sampling_window": 60,
            "target_cycles": 1e9,
            "seed": 1,
            "cell_length": 6.0,
            "buffer": 2.0,
        },
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = parallel.make_weigher(config_specs.kernel_spec(cfg), pot)
    rng = np.random.default_rng(0)
    st = ChainState(
        positions=rng.uniform(1.0, 23.0, size=(120, 3)),
        step=0,
        rng=np.random.default_rng(7),
        counters=zero_counters(),
    )
    last = None
    for _ in range(n_win):
        last = parallel.advance_parallel(st, cfg, pot, w)
    return last


for name, blk in [
    ("metropolis", {"type": "metropolis"}),
    (
        "rosenbluth",
        {"type": "rosenbluth", "trials": K, "trial_box_side": 2 * CAVR, "depth": 0},
    ),
    ("branch", {"type": "branch", "trials": K, "trial_box_side": 2 * CAVR, "depth": 1}),
]:
    last = _run3d(blk)
    assert last["n_cells"] > 1, f"{name}: box did not grid into several cells"
    assert last["N"] > 0, f"{name}: 3D parallel depopulated"
    print(f"  ok  3D parallel {name} runs over {last['n_cells']} cells, N={last['N']}")


# 4. mixes that stay serial-only must still be rejected at config time.
def _rejected(label, p):
    try:
        from_params(p)
    except ValueError as e:
        print(f"  ok  {label} rejected:", str(e)[:55])
        return
    raise SystemExit(f"FAIL: {label} was accepted under parallel")


_rejected(
    "reweight+parallel",
    {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": 2, "box": 40.0},
        "sampler": {"type": "reweight"},
        "run": {"kind": "parallel", "n_cell": 9, "sampling_window": 50},
    },
)
_rejected(
    "channel+parallel",
    {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 2.0, "dimension": 3, "box": 24.0},
        "sampler": {
            "type": "channel",
            "trial_density": 2.0,
            "depth": 1,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        "run": {
            "kind": "parallel",
            "sampling_window": 50,
            "cell_length": 6.0,
            "buffer": 2.0,
        },
    },
)
_rejected(
    "confined+parallel",
    {
        "potential": {"main": "hard_core"},
        "ensemble": {
            "type": "confined",
            "mu": MU,
            "dimension": 3,
            "box": 24.0,
            "pore": {"geometry": "slit", "H": 6.0},
        },
        "sampler": {"type": "metropolis"},
        "run": {
            "kind": "parallel",
            "sampling_window": 50,
            "cell_length": 6.0,
            "buffer": 2.0,
        },
    },
)

print("\nALL PARALLEL TESTS PASSED")
