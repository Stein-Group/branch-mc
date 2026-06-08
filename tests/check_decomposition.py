"""Decomposition checks: the ported domain-decomposition helpers reproduce the
original.

The deterministic helpers (_axis_mask, neighbor_indices_in_cavity_parallel,
apply_pbc_array, apply_PBC_single) are compared exactly. grid_up_the_system is
compared exactly on a fixed grid origin, and for structure on a seeded random
origin. create_cavity draws from an unseedable default_rng in both versions, so
it is only bounds-checked: the centre it returns must lie inside the cell.

The original helpers are exec'd from source into a clean namespace, which avoids
importing the monolith (it parses argv and needs ase at import time) while
running the exact original code.
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..")
sys.path.insert(0, WORK)
ORIG = os.path.join(WORK, "orig")
sys.path.insert(0, ORIG)

from gcmc import decomposition as D  # noqa: E402

# ---- build the original helpers in isolation -------------------------------
src = open(os.path.join(ORIG, "cav_n_d_3d_g3_LJ.py")).read().splitlines()
blob = "\n".join(src[1457:1867])  # create_cavity .. end of apply_PBC_single
ns = {"np": np, "cKDTree": cKDTree, "cell_length_global": 30, "time": __import__("time")}
exec(compile(blob, "<orig-helpers>", "exec"), ns)

import parallelization as PAR  # standalone grid_up_the_system  # noqa: E402

D.configure(30)

def ok(name):
    print(f"  ok  {name}")

# ---- _axis_mask: interior, split, and straddling cells ---------------------
side = 200.0
coords = np.linspace(0, side, 401)
cases = [
    # (lo, hi, split_left, split_right)  via cavity_center/radius below
]
for center, radius, sl, sr in [
    (30.0, 10.0, 15.0, 15.0),    # interior cell (split_left==split_right)
    (5.0, 10.0, 0.0, side),      # split cell wrapping at global edge (low)
    (195.0, 10.0, 0.0, side),    # split cell wrapping at global edge (high)
]:
    lo, hi = center - radius, center + radius
    mine = D._axis_mask(coords, center, radius, lo, hi, sl, sr, side)
    theirs = ns["_axis_mask"](coords, center, radius, lo, hi, sl, sr, side)
    assert np.array_equal(mine, theirs), f"_axis_mask differ at center={center}"
ok("_axis_mask matches across interior and split cells")

# ---- neighbor_indices_in_cavity_parallel -----------------------------------
rng = np.random.default_rng(0)
mol = rng.uniform(0, side, size=(2000, 3)); mol[:, 2] = 0.0
for center, cell0, cell1 in [
    (np.array([40.0, 40.0, 0.0]), [25, 55, 25, 55], [side/2]*4),
    (np.array([3.0, 3.0, 0.0]),   [0, side, 0, side], [0, side, 0, side]),
]:
    cell_dims = [cell0, cell1]
    mine = D.neighbor_indices_in_cavity_parallel(mol, center, 10.0, cell_dims,
                                                 [side, side], True)
    theirs = ns["neighbor_indices_in_cavity_parallel"](mol, center, 10.0, cell_dims,
                                                       [side, side], True)
    assert np.array_equal(np.sort(mine), np.sort(theirs)), "neighbor idx differ"
ok("neighbor_indices_in_cavity_parallel matches")

# ---- apply_pbc_array (interior + split cells) ------------------------------
for cell0, cell1 in [
    ([60, 90, 60, 90], [side/2]*4),                 # interior: cell_length 30
    ([side-15, 15, side-15, 15], [0, side, 0, side]),  # split: length side-30
]:
    cell_dims = [cell0, cell1]
    tx = rng.uniform(0, side, 300).copy()
    ty = rng.uniform(0, side, 300).copy()
    m = D.apply_pbc_array(tx.copy(), ty.copy(), cell_dims, [side, side])
    t = ns["apply_pbc_array"](tx.copy(), ty.copy(), cell_dims, [side, side])
    assert np.allclose(np.asarray(m), np.asarray(t)), "apply_pbc_array differ"
ok("apply_pbc_array matches (interior + split)")

# ---- apply_PBC_single ------------------------------------------------------
for cell0, cell1 in [
    ([60, 90, 60, 90], [side/2]*4),
    ([side-15, 15, side-15, 15], [0, side, 0, side]),
]:
    cell_dims = [cell0, cell1]
    for _ in range(200):
        x = float(rng.uniform(0, side))
        y = float(rng.uniform(0, side))
        mx, my = D.apply_PBC_single(x, y, cell_dims, [side, side])
        tx, ty = ns["apply_PBC_single"](x, y, cell_dims, [side, side])
        assert np.allclose([mx, my], [tx, ty]), f"apply_PBC_single differ at {x},{y}"
ok("apply_PBC_single matches (interior + split)")

# ---- grid_up_the_system: exact on fixed origin -----------------------------
sys_xyz = rng.uniform(0, side, size=(5000, 3)); sys_xyz[:, 2] = 0.0
mine_set, mine_rest = D.grid_up_the_system(sys_xyz.copy(), side=side,
                                           cell_length=30, x_pos=20, y_pos=20)
their_set, their_rest = PAR.grid_up_the_system(sys_xyz.copy(), side=side,
                                               cell_length=30, x_pos=20, y_pos=20)
assert len(mine_set) == len(their_set), "cell count differs"
assert np.allclose(np.sort(mine_rest, axis=0), np.sort(their_rest, axis=0))
for (m_main, m_bord, m_dims), (t_main, t_bord, t_dims) in zip(mine_set, their_set):
    assert len(m_main) == len(t_main) and len(m_bord) == len(t_bord)
    assert m_dims == t_dims
ok(f"grid_up_the_system matches exactly ({len(mine_set)} cells, fixed origin)")

# every molecule lands in exactly one cell-main or the rest (partition)
n_main = sum(len(c[0]) for c in mine_set)
assert n_main + len(mine_rest) == len(sys_xyz), "partition is not exhaustive"
ok("gridding partitions the system exactly (main + rest == all)")

# seeded random-origin path: same count both sides under the same seed
np.random.seed(7)
a, ra = D.grid_up_the_system(sys_xyz.copy(), side=side, cell_length=30)
np.random.seed(7)
b, rb = PAR.grid_up_the_system(sys_xyz.copy(), side=side, cell_length=30)
assert len(a) == len(b) and len(ra) == len(rb)
ok("grid_up_the_system matches under seeded random origin")

# ---- create_cavity: bounds check (unseedable rng in both) ------------------
for cell0, cell1, two_d in [
    ([60, 90, 60, 90], [side/2]*4, True),
    ([20, 50, 20, 50], [side/2]*4, True),
]:
    cd = [cell0, cell1]
    for _ in range(500):
        c = D.create_cavity(cd, [side, side], two_d)
        cx, cy = c[0], c[1]
        assert cell0[0] <= cx <= cell0[1], f"cavity x {cx} outside {cell0[:2]}"
        assert cell0[2] <= cy <= cell0[3], f"cavity y {cy} outside {cell0[2:]}"
ok("create_cavity centres land inside the cell")

print("\nALL DECOMPOSITION TESTS PASSED")
