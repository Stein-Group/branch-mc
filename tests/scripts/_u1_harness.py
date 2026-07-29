"""U1 validation harness: capture per-cell advance_cell trajectories at fixed
seeds and a tree-share benchmark, so the CellList refactor can be checked for
trajectory identity (or documented statistical parity) against the pre-U1 code.

Not a committed part of gcmc/. Lives under tests/scripts/ as a runnable oracle.
Usage:
    python tests/scripts/_u1_harness.py capture  out.npz
    python tests/scripts/_u1_harness.py compare  ref.npz cur.npz
    python tests/scripts/_u1_harness.py bench
"""

from __future__ import annotations

import sys
import time

import numpy as np

from gcmc import parallel
from gcmc.sim_setup import from_params

# One cell is one core, so cfg.n_workers is the cell count; without this
# hook a 9- or 27-cell case would spawn that many processes inside a test.
parallel.IN_PROCESS = True

# A fixed set of (label, params) cells covering the 2D matrix U1 must preserve.
_CASES = {
    "hs_rosenbluth_d0": {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 1.5, "dimension": 2, "box": 40.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.0,
            "seed": 1,
        },
    },
    "hs_branch_d1": {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 2.0, "dimension": 2, "box": 40.0},
        "sampler": {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.0,
            "seed": 1,
        },
    },
    "hs_metropolis": {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 1.5, "dimension": 2, "box": 40.0},
        "sampler": {"type": "metropolis"},
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.0,
            "seed": 1,
        },
    },
    "lj_rosenbluth_d0": {
        "potential": {"main": "lennard_jones", "T": 2.0, "tail": False},
        "ensemble": {"mu": -2.0, "dimension": 2, "box": 40.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.6,
            "seed": 1,
        },
    },
    "lj_branch_d1": {
        "potential": {"main": "lennard_jones", "T": 2.0, "tail": False},
        "ensemble": {"mu": -1.0, "dimension": 2, "box": 40.0},
        "sampler": {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.6,
            "seed": 1,
        },
    },
    "lj_metropolis": {
        "potential": {"main": "lennard_jones", "T": 2.0, "tail": False},
        "ensemble": {"mu": -2.0, "dimension": 2, "box": 40.0},
        "sampler": {"type": "metropolis"},
        "run": {
            "kind": "parallel",
            "sampling_window": 300,
            "cell_length": 8.0,
            "buffer": 2.6,
            "seed": 1,
        },
    },
}


def _one_cell(label, seed=12345):
    """Build a deterministic single cell (main + border) and run advance_cell
    once. Returns (final_global_positions, counters_tuple)."""
    cfg = from_params(_CASES[label])
    pot = cfg.build_potential()
    weigher = parallel.make_weigher(cfg, pot)
    dim = 2
    L_c = cfg.cell_length
    rng_geom = np.random.default_rng(seed)
    # cell at origin; main particles inside the cell cube, border in the ring
    origin = np.full(dim, 16.0)  # mid-box: cell [16, 16+L_c], border ring stays > 0
    n_main = 18
    main = rng_geom.uniform(0.5, L_c - 0.5, size=(n_main, dim)) + origin
    # border: a frozen ring just outside the cell, within buffer, global coords
    n_border = 12
    b = []
    while len(b) < n_border:
        p = rng_geom.uniform(-cfg.buffer, L_c + cfg.buffer, size=dim) + origin
        inside = np.all((p - origin >= 0) & (p - origin <= L_c))
        if not inside:
            b.append(p)
    border = np.array(b)
    rng = np.random.default_rng(777)
    out, c = parallel.advance_cell(
        main,
        border,
        origin,
        cfg.sampling_window,
        cfg,
        pot,
        weigher,
        rng,
        n_offset=len(border),
    )
    ckeys = sorted(c)
    return (
        np.asarray(out, float),
        np.array([c[k] for k in ckeys], dtype=np.int64),
        ckeys,
    )


def capture(path):
    data = {}
    for label in _CASES:
        out, counters, ckeys = _one_cell(label)
        data["__ckeys"] = np.array(ckeys)
        data[f"{label}__pos"] = out
        data[f"{label}__cnt"] = counters
        print(f"  captured {label:20s} N={len(out):3d}  counters={counters.tolist()}")
    np.savez(path, **data)
    print(f"saved -> {path}")


def compare(ref_path, cur_path):
    ref = np.load(ref_path)
    cur = np.load(cur_path)
    all_ok = True
    for label in _CASES:
        rp, cp = ref[f"{label}__pos"], cur[f"{label}__pos"]
        rc, cc = ref[f"{label}__cnt"], cur[f"{label}__cnt"]
        same_shape = rp.shape == cp.shape
        pos_ident = same_shape and np.array_equal(rp, cp)
        cnt_ident = np.array_equal(rc, cc)
        if pos_ident and cnt_ident:
            print(f"  IDENTICAL  {label}")
        else:
            all_ok = False
            note = []
            if not same_shape:
                note.append(f"shape {rp.shape}!={cp.shape}")
            elif not pos_ident:
                md = np.abs(rp - cp).max() if rp.size else 0.0
                note.append(f"pos max|d|={md:.2e}")
            if not cnt_ident:
                note.append(f"cnt {rc.tolist()} vs {cc.tolist()}")
            print(f"  DIFFERS    {label}: {'; '.join(note)}")
    print("ALL IDENTICAL" if all_ok else "SOME DIFFER")
    return all_ok


def bench():
    """Instrumented tree-share: fraction of a real 2D parallel window spent
    building/querying the per-move cKDTree, plus raw per-move build cost at
    several N. Numbers are container numbers; record alongside on the dev box."""
    from scipy.spatial import cKDTree

    print("--- per-move cKDTree build (3D) ---")
    for n in (1000, 5000, 18000):
        pts = np.random.default_rng(0).uniform(0, 30, size=(n, 3))
        t = time.perf_counter()
        reps = 200
        for _ in range(reps):
            cKDTree(pts, boxsize=[30, 30, 30])
        dt = (time.perf_counter() - t) / reps * 1e3
        print(f"  N={n:6d}   build = {dt:.3f} ms")

    print("--- live CellList query_point + move (3D) ---")
    from gcmc.neighbors import CellList

    for n in (1000, 5000, 18000):
        pts = np.random.default_rng(0).uniform(0, 30, size=(n, 3))
        cl = CellList.from_positions(pts, 30.0, 3, 1.0)
        q = pts[0]
        t = time.perf_counter()
        reps = 2000
        for _ in range(reps):
            cl.query_point(q, 1.0)
        dq = (time.perf_counter() - t) / reps * 1e6
        t = time.perf_counter()
        for i in range(reps):
            cl.move(cl.active_slots()[0], pts[i % n])
        dm = (time.perf_counter() - t) / reps * 1e6
        print(f"  N={n:6d}   query_point = {dq:.1f} us   move = {dm:.1f} us")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "capture"
    if cmd == "capture":
        capture(sys.argv[2] if len(sys.argv) > 2 else "/tmp/u1_ref.npz")
    elif cmd == "compare":
        ok = compare(sys.argv[2], sys.argv[3])
        sys.exit(0 if ok else 1)
    elif cmd == "bench":
        bench()
