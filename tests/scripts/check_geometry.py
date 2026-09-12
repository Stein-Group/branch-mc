"""Characterize geometry.py against the original, one mode per process.

  python check_geometry.py <mode>     (cwd anywhere; orig is found by path)

Checks:
  1. indices_in_cavity vs original neighbor_indices_in_cavity, with centres that
     exercise the interior and both wrapped branches on each axis.
  2. find_neighbors (raw) + potential.weight_bulk vs original find_neighbors_fast,
     comparing surviving coords, the weight, the carried keep_mask, and (aux3d)
     the reference neighbour coords.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

MODE = sys.argv[1]
ARGV = {
    "lj3d": [
        "--mu",
        "-4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--LJ_pot",
        "--T",
        "0.85",
    ],
    "hs2d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--2d",
        "--T",
        "1.0",
    ],
    "hs3d": ["--mu", "4", "--rho_k", "400", "--d", "0", "--r", "1.5", "--T", "1.0"],
    "aux2d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--2d",
        "--aux_pot",
        "--T",
        "1.0",
    ],
    "aux3d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--aux_pot",
        "--T",
        "1.0",
    ],
}
HERE = os.path.dirname(os.path.abspath(__file__))
from _frozen import add_orig_to_path

ORIG = str(add_orig_to_path())  # R1: resolved from $ORIG_ROOT or the tag
sys.argv = ["orig"] + ARGV[MODE]
if MODE == "aux3d":
    os.chdir(ORIG)  # g3_analysis_data.csv is loaded relative to cwd at import
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.forcefield.auxiliary import HardCoreAux2D, HardCoreAux3D, aux_constants_2d
from gcmc.forcefield.lennard_jones import LennardJones
from gcmc.proposals.trials import TrialSet
from gcmc.space import geometry


def eq(a, b, tag):
    a, b = np.asarray(a), np.asarray(b)
    if not (
        a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
    ):
        raise AssertionError(f"[FAIL] {tag}")
    print(f"  ok  {tag}")


side = orig.side_of_box
two_d = orig.two_dim_flag
dim = 2 if two_d else 3
rg = np.random.default_rng(7)

# ---- 1. cavity index search ----
mols = rg.uniform(0, side, size=(400, dim))
radius = 2.0
lo = radius * 0.4  # center < radius  -> lo<0 branch
hi = side - radius * 0.4  # center+radius>side -> hi>side branch
mid = side * 0.5
for cx in (lo, mid, hi):
    for cy in (lo, mid, hi):
        if two_d:
            center = np.array([cx, cy])
        else:
            center = np.array([cx, cy, mid])
        a = orig.neighbor_indices_in_cavity(mols, center, radius, side)
        b = geometry.indices_in_cavity(mols, center, radius, side, two_d)
        if not np.array_equal(np.sort(a), np.sort(b)):
            raise AssertionError(f"[FAIL] cavity center={center}")
print("  ok  indices_in_cavity (9 centres, all branches)")

# ---- 2. find_neighbors + weighting vs original find_neighbors_fast ----
system = rg.uniform(0, side, size=(300, dim))
box = [side] * dim
tree = cKDTree(system, boxsize=box)
trials = rg.uniform(0, side, size=(60, dim))
keep = rg.uniform(size=60) < 0.5
lower = orig.diameter_of_disk
upper = orig.upper_cutoff
nn = orig.no_neigh
N_glob = len(orig.molecules_xyz)
g1 = orig.gamma_dict[1]

o_set, o_km = orig.find_neighbors_fast(
    tree, trials, lower, upper, nn, keep_mask=keep.copy(), num_mols=N_glob, choice="add"
)
assert o_set is not None, "reference produced no surviving trials; adjust inputs"

raw = geometry.find_neighbors(tree, trials, lower, upper, nn, keep_mask=keep.copy())
coords, dists, idxs, km, _spc = raw

if MODE == "lj3d":
    pot = LennardJones(
        beta=orig.BETA, upper_cutoff=upper, diameter=lower, volume=orig.Total_volume
    )
    w = pot.weight_bulk(dists, n_particles=N_glob, choice="add")
    ts = TrialSet(coords, w)
elif MODE in ("hs2d", "hs3d"):
    w = np.ones(len(coords))
    ts = TrialSet(coords, w)
elif MODE == "aux2d":
    pot = HardCoreAux2D(
        diameter=lower, upper_cutoff=upper, params=aux_constants_2d(eta=0.69)
    )
    w = pot.weight_bulk(dists, gamma=g1)
    ts = TrialSet(coords, w)
elif MODE == "aux3d":
    pot = HardCoreAux3D.build(
        diameter=lower,
        upper_cutoff=upper,
        side=side,
        n_trials=orig.n_trial_molecules,
        g3_csv=os.path.join(ORIG, "g3_analysis_data.csv"),
    )
    w, ref = pot.weight_bulk(dists, coords=coords, idxs=idxs, kdtree=tree, gamma=g1)
    ts = TrialSet(coords, w, ref_neighbor_coord=ref)

eq(ts.coords, o_set.coords, f"{MODE} find_neighbors coords")
eq(ts.energy_weight, o_set.energy_weight, f"{MODE} find_neighbors weight")
eq(km, o_km, f"{MODE} find_neighbors keep_mask")
if MODE == "aux3d":
    eq(ts.ref_neighbor_coord, o_set.ref_neighbor_coord, "aux3d ref_neighbor_coord")

print(f"PASS {MODE}")
