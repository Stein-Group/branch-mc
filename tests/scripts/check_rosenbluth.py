"""Diff select_forward against the original rosenbluth_weight_new, per mode.

  python check_rosenbluth.py <mode>

Builds a real trial pool with the original find_neighbors_fast, seeds the legacy
RNG identically before each call, and compares the chosen coordinates, the
per-level sum_weight, and the chosen weights. Identical RNG + identical logic
must give a bit-for-bit identical selection.
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
ORIG = os.path.join(HERE, "..", "..", "orig")
sys.argv = ["orig"] + ARGV[MODE]
sys.path.insert(0, ORIG)
if MODE == "aux3d":
    os.chdir(ORIG)
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.potentials.aux import Aux2D, Aux3D, aux_constants_2d
from gcmc.potentials.lennard_jones import LennardJones
from gcmc.samplers.base import EnergyWeigher, UniformWeigher
from gcmc.samplers.rosenbluth import select_forward
from gcmc.trials import TrialSet


def eq(a, b, tag):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (
        a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
    ):
        raise AssertionError(f"[FAIL] {tag}  shapes {a.shape} {b.shape}")
    print(f"  ok  {tag}")


side = orig.side_of_box
two_d = orig.two_dim_flag
dim = 2 if two_d else 3
lower, upper, nn = orig.diameter_of_disk, orig.upper_cutoff, orig.no_neigh
rg = np.random.default_rng(11)

# trials inside a cavity (as the engine generates them); sparse system so most
# survive. Confining to the cavity keeps min-image separations small, which the
# original's 2D path asserts.
system = rg.uniform(0, side, size=(30, dim))
tree = cKDTree(system, boxsize=[side] * dim)
center = np.full(dim, side / 2.0)
rcav = orig.cavity_radius
trials = np.mod(rg.uniform(-rcav, rcav, size=(300, dim)) + center, side)
pool0 = orig.find_neighbors_fast(
    tree, trials, lower, upper, nn, num_mols=len(orig.molecules_xyz), choice="add"
)[0]
assert pool0 is not None and len(pool0) > 20, "need a large pool for this test"

# snapshot the pool arrays so both sides start from identical input
A = (
    pool0.coords.copy(),
    pool0.energy_weight.copy(),
    pool0.cossum.copy(),
    pool0.sinsum.copy(),
    pool0.counts.copy(),
    pool0.ref_neighbor_coord.copy(),
)

if MODE in ("hs2d", "hs3d"):
    weigher = UniformWeigher()
elif MODE == "lj3d":
    weigher = EnergyWeigher(
        LennardJones(beta=orig.BETA, upper_cutoff=upper, diameter=lower)
    )
elif MODE == "aux2d":
    weigher = EnergyWeigher(
        Aux2D(diameter=lower, upper_cutoff=upper, params=aux_constants_2d(eta=0.69))
    )
elif MODE == "aux3d":
    weigher = EnergyWeigher(
        Aux3D.build(
            diameter=lower,
            upper_cutoff=upper,
            side=side,
            n_trials=orig.no_of_trial_molecules,
            g3_csv=os.path.join(ORIG, "g3_analysis_data.csv"),
        )
    )

r_sq = orig.r_squared
for n_atoms in (1, 2, 4):
    SEED = 1234 + n_atoms

    np.random.seed(SEED)
    o_chosen, o_sum = orig.rosenbluth_weight_new(pool0.copy(), n_atoms, "add")

    np.random.seed(SEED)
    my_pool = TrialSet(*[a.copy() for a in A])
    my_chosen, my_sum = select_forward(
        my_pool,
        n_atoms,
        weigher,
        rand=np.random.rand,
        side=side,
        r_squared=r_sq,
        gamma_schedule=orig.gamma_dict,
    )

    assert (o_chosen is None) == (my_chosen is None), f"None mismatch n={n_atoms}"
    if o_chosen is None:
        print(f"  ok  n_atoms={n_atoms} both rejected")
        continue
    eq(my_chosen.coords, o_chosen.coords, f"{MODE} chosen coords n={n_atoms}")
    eq(my_sum, o_sum, f"{MODE} sum_weight n={n_atoms}")
    eq(
        my_chosen.energy_weight,
        o_chosen.energy_weight,
        f"{MODE} chosen weight n={n_atoms}",
    )

print(f"PASS {MODE}")
