"""The parallel engine agrees with the serial one, and its cell list stays honest.

U4 of cell_unify_plan.md. Two things had no pytest-level coverage before:

  * 3D parallel. It runs deliberately now (the original's "parallel is 2D only"
    guard was dropped by decision in U0), so the burden of proof moved here and
    onto the cluster ladder in U3. The heavy statistical validation is U3's;
    this is the fast rung that keeps CI honest between cluster runs.
  * The cell list `advance_cell` now carries (U1). A tombstone that was never
    reactivated, or a slot array that drifted out of step with the coordinates,
    would not necessarily change the density enough to fail a parity check --
    but it would break an invariant immediately. So the invariants are asserted
    directly, through the DEBUG_CELL_LIST hook, on the paths that mutate them.

Parity here is statistical with deliberately generous bars: the two engines do
not share a trajectory (different box, different cell count, different rng
stream), only a stationary distribution. Tight tolerances on short runs would
buy flakiness, not sensitivity. The precise version lives in U3.
"""

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import MC_engine as engine
from gcmc.engine import parallel
from gcmc.run.sim_setup import from_params
from gcmc.system.state import ChainState, zero_counters

# One cell is one core, so cfg.n_workers is the cell count; without this
# hook a 9- or 27-cell case would spawn that many processes inside a test.
parallel.IN_PROCESS = True

MU = 1.0


def _cfg(dim, kind, box, window):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": dim, "box": box},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 5.0,
            "depth": 0,
        },
        "run": {
            "kind": kind,
            "sampling_window": window,
            "target_cycles": 1e9,
            "seed": 1,
        },
    }
    if kind == "parallel":
        p["run"]["cell_length"] = box / 2.0
        p["run"]["buffer"] = 2.0
    return from_params(p)


def _state(n, dim, hi, seed=0):
    return ChainState(
        positions=np.random.default_rng(seed).uniform(1.0, hi, size=(n, dim)),
        step=0,
        rng=np.random.default_rng(7),
        counters=zero_counters(),
    )


def _equilibrium_eta(advance, cfg, st, n_win, n_tail):
    """Run n_win windows, average eta over the last n_tail, and return the mean
    with a block-standard-error over those windows (the B3.2 lesson: compare on
    honest bars, never on a flat tolerance)."""
    pot = cfg.build_potential()
    w = (
        parallel.make_weigher(config_specs.kernel_spec(cfg), pot)
        if advance is parallel.advance_parallel
        else engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    )
    etas = []
    for _ in range(n_win):
        out = advance(st, cfg, pot, w)
        etas.append(out["packing_fraction"])
    tail = np.asarray(etas[-n_tail:])
    return float(tail.mean()), float(tail.std(ddof=1) / np.sqrt(len(tail)))


@pytest.mark.parametrize(
    "dim",
    [
        2,
        pytest.param(3, marks=pytest.mark.slow),
    ],
)
def test_parallel_matches_serial_equilibrium(dim):
    """eta(mu) is intensive, so the decomposed run and the single-box run must
    settle on the same plateau from different starting densities. A bug in the
    acceptance algebra, the cell-torus wrap, or the cell-volume term moves it.

    3D is marked slow for a measured reason, not a guess: its box holds ~8x the
    volume of the 2D one at the same cell size, and the approach to the plateau
    is correspondingly slower. From N=60 it climbs 0.040 -> 0.129 (window 10)
    -> 0.170 (window 16) -> 0.177 (window 30), against a serial plateau of
    ~0.186. Cutting it short at 10 windows reports 0.045 and looks like a
    physics failure; it is only an unequilibrated chain. Equilibration windows
    are what the `slow` marker is for.
    """
    box_p, box_s = (30.0, 18.0) if dim == 2 else (12.0, 10.0)
    n_win, n_tail = (14, 6) if dim == 2 else (30, 8)
    n0_p, n0_s = (60, 20) if dim == 2 else (60, 30)
    window = 250 if dim == 2 else 200

    cfg_p = _cfg(dim, "parallel", box_p, window=window)
    cfg_s = _cfg(dim, "single", box_s, window=window)

    eta_p, err_p = _equilibrium_eta(
        parallel.advance_parallel, cfg_p, _state(n0_p, dim, box_p - 1.0), n_win, n_tail
    )
    eta_s, err_s = _equilibrium_eta(
        engine.advance, cfg_s, _state(n0_s, dim, box_s - 1.0, seed=3), n_win, n_tail
    )

    # honest bars plus a floor for the residual drift of a finite run; the
    # precise comparison, with block averaging and long runs, is U3's job
    bar = 3.0 * np.hypot(err_p, err_s) + 0.02
    assert abs(eta_p - eta_s) < bar, (
        f"{dim}D parallel eta={eta_p:.4f}+/-{err_p:.4f} vs serial "
        f"eta={eta_s:.4f}+/-{err_s:.4f}; gap {abs(eta_p - eta_s):.4f} > {bar:.4f}"
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "sampler",
    [
        {"type": "metropolis"},
        {"type": "rosenbluth", "trials": 20, "trial_box_side": 3.0, "depth": 0},
        {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
    ],
    ids=["metropolis", "rosenbluth", "branch"],
)
def test_cell_list_invariants_hold(dim, sampler, monkeypatch):
    """Every proposal leaves the cell list describing exactly the live
    configuration -- including the rejected ones, which is where a missed
    reactivate would hide. The hook asserts inside the `finally`, so every early
    exit in the move loop is covered too."""
    monkeypatch.setattr(parallel, "DEBUG_CELL_LIST", True)
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 2.0, "dimension": dim, "box": 24.0},
        "sampler": sampler,
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
    st = _state(60 if dim == 2 else 200, dim, 23.0)
    for _ in range(3):
        out = parallel.advance_parallel(st, cfg, pot, w)
    assert out["N"] > 0


def test_lj_cell_list_invariants_hold(monkeypatch):
    """Same, for LJ: the border must stay in the list and stay unflagged, since
    `is_matrix` is what would silently drop it from the energy sum."""
    monkeypatch.setattr(parallel, "DEBUG_CELL_LIST", True)
    p = {
        "potential": {"main": "lennard_jones", "T": 2.0, "tail": False},
        "ensemble": {"mu": -2.0, "dimension": 3, "box": 24.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "run": {
            "kind": "parallel",
            "sampling_window": 50,
            "target_cycles": 1e9,
            "seed": 1,
            "cell_length": 6.0,
            "buffer": 2.6,
        },
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = parallel.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = _state(150, 3, 23.0)
    for _ in range(3):
        out = parallel.advance_parallel(st, cfg, pot, w)
    assert out["N"] > 0
