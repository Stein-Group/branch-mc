"""Diff select_reverse against the original rosenbluth_weight_current, per mode.

  python check_reverse.py <mode>

The reverse weight has no random draw, so this is a direct deterministic diff of
the per-level sum_weight and the per-molecule weight, for both the with-trial
path and the trial-is-None fast path. The cavity molecules and trial pool are
built with the original find_neighbors_fast so the inputs are identical.
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
    os.chdir(ORIG)
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.forcefield.aux import HardCoreAux2D, HardCoreAux3D, aux_constants_2d
from gcmc.forcefield.lennard_jones import LennardJones
from gcmc.proposals.trials import TrialSet
from gcmc.sampling.base import EnergyWeigher, UniformWeigher
from gcmc.sampling.rosenbluth import select_reverse


def eq(a, b, tag):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (
        a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
    ):
        raise AssertionError(f"[FAIL] {tag}")
    print(f"  ok  {tag}")


side = orig.side_of_box
two_d = orig.two_dim_flag
dim = 2 if two_d else 3
lower, upper, nn = orig.diameter_of_disk, orig.upper_cutoff, orig.no_neigh
r_sq, gdict = orig.r_squared, orig.gamma_dict
rg = np.random.default_rng(23)

# sparse whole-box system: leaves the cavity clear so candidate cavity
# molecules and trials survive the overlap filter
center = np.full(dim, side / 2.0)
system = rg.uniform(0, side, size=(30, dim))
tree = cKDTree(system, boxsize=[side] * dim)
rcav = orig.cavity_radius


def fn(coords):
    return orig.find_neighbors_fast(
        tree,
        coords,
        lower,
        upper,
        nn,
        num_mols=len(orig.molecules_xyz),
        choice="remove",
    )[0]


cav_pts = np.mod(rg.uniform(-rcav, rcav, size=(80, dim)) + center, side)
trial_pts = np.mod(rg.uniform(-rcav, rcav, size=(150, dim)) + center, side)
cav_full = fn(cav_pts)
trial_full = fn(trial_pts)
assert cav_full is not None and len(cav_full) >= 3, "need cavity molecules"
assert trial_full is not None and len(trial_full) > 5, "need a trial pool"

if MODE in ("hs2d", "hs3d"):
    weigher = UniformWeigher()
elif MODE == "lj3d":
    weigher = EnergyWeigher(
        LennardJones(beta=orig.BETA, upper_cutoff=upper, diameter=lower)
    )
elif MODE == "aux2d":
    weigher = EnergyWeigher(
        HardCoreAux2D(
            diameter=lower, upper_cutoff=upper, params=aux_constants_2d(eta=0.69)
        )
    )
elif MODE == "aux3d":
    weigher = EnergyWeigher(
        HardCoreAux3D.build(
            diameter=lower,
            upper_cutoff=upper,
            side=side,
            n_trials=orig.no_of_trial_molecules,
            g3_csv=os.path.join(ORIG, "g3_analysis_data.csv"),
        )
    )


def snap(s):
    return (
        s.coords.copy(),
        s.energy_weight.copy(),
        s.cossum.copy(),
        s.sinsum.copy(),
        s.counts.copy(),
        s.ref_neighbor_coord.copy(),
    )


for n_atoms in (1, 2, 3):
    cav = cav_full[:n_atoms]
    C = snap(cav)
    T = snap(trial_full)

    # with-trial path
    o_sum, o_e = orig.rosenbluth_weight_current(
        orig.TrialMoleculeSet(*[a.copy() for a in T]),
        orig.TrialMoleculeSet(*[a.copy() for a in C]),
        n_atoms,
        "remove",
    )
    my_sum, my_e = select_reverse(
        TrialSet(*[a.copy() for a in T]),
        TrialSet(*[a.copy() for a in C]),
        n_atoms,
        weigher,
        side=side,
        r_squared=r_sq,
        gamma_schedule=gdict,
    )
    eq(my_sum, o_sum, f"{MODE} reverse sum_weight n={n_atoms}")
    eq(my_e, o_e, f"{MODE} reverse per-mol weight n={n_atoms}")

    # trial-is-None fast path
    o_sum2, o_e2 = orig.rosenbluth_weight_current(
        None, orig.TrialMoleculeSet(*[a.copy() for a in C]), n_atoms, "remove"
    )
    my_sum2, my_e2 = select_reverse(
        None,
        TrialSet(*[a.copy() for a in C]),
        n_atoms,
        weigher,
        side=side,
        r_squared=r_sq,
        gamma_schedule=gdict,
    )
    eq(my_sum2, o_sum2, f"{MODE} reverse(None) sum_weight n={n_atoms}")
    eq(my_e2, o_e2, f"{MODE} reverse(None) per-mol weight n={n_atoms}")

print(f"PASS {MODE}")
