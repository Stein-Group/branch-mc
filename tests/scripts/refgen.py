"""Generate reference outputs from the ORIGINAL code, one mode per process.

Run as:  python refgen.py <mode>   with cwd = the orig/ folder.
Writes refs/<mode>.npz. The original parses argv at import, so we set argv
before importing it. Inputs are generated from a fixed seed and saved alongside
the outputs so the characterization test feeds identical arrays.
"""

import os
import sys

import numpy as np

MODE = sys.argv[1]
REFDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "refs")
os.makedirs(REFDIR, exist_ok=True)

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

# 3D aux needs the g3 table at import time; provide a small synthetic grid that
# the characterization test will load identically.
if MODE == "aux3d" and not os.path.exists("g3_analysis_data.csv"):
    rg = np.linspace(1.0, 2.5, 12)
    cg = np.linspace(-1.0, 1.0, 9)
    rows = []
    for r in rg:
        for c in cg:
            g = 0.5 + 0.4 * np.cos(3 * r) * (1 + c) / 2 + 0.3
            rows.append([r, 0.0, c, 0.0, max(g, 0.01)])
    hdr = "r,dummy,cos,dummy2,g3"
    np.savetxt(
        "g3_analysis_data.csv", np.array(rows), delimiter=",", header=hdr, comments=""
    )

sys.argv = ["orig"] + ARGV[MODE]
sys.path.insert(
    0, os.getcwd()
)  # orig/ is cwd; lets the original + parallelization import
import cav_n_d_3d_g3_LJ as orig

rng = np.random.default_rng(0)
out = {}

if MODE in ("lj3d", "hs2d", "hs3d", "aux2d"):
    if MODE == "aux2d":
        # domain: diameter <= r <= upper_cutoff, with inf padding in column 0 allowed
        lo, hi = orig.diameter_of_disk, orig.upper_cutoff
        r2 = rng.uniform(lo + 1e-3, hi - 1e-3, size=(20, 6))
        r1 = rng.uniform(lo + 1e-3, hi - 1e-3, size=15)
        gammas = [1.0, 0.5]
    elif MODE == "lj3d":
        r2 = rng.uniform(0.95, 3.0, size=(20, 6))
        r2[::4, -1] = np.inf
        r1 = rng.uniform(0.95, 3.0, size=15)
        gammas = [1.0]
    else:  # hard sphere
        r2 = rng.uniform(0.5, 4.0, size=(20, 6))
        r1 = rng.uniform(0.5, 4.0, size=15)
        gammas = [1.0]

    out["r2"] = r2
    out["r1"] = r1
    for g in gammas:
        out[f"w2_g{g}"] = np.asarray(orig.calculate_interaction_energy(r2.copy(), g))
        out[f"w1_g{g}"] = np.asarray(orig.calculate_interaction_energy(r1.copy(), g))

    if MODE == "lj3d":
        out["lj_tail_pref"] = orig.lj_tail_pref
        out["BETA"] = orig.BETA
        out["Total_volume"] = orig.Total_volume
        out["upper_cutoff"] = orig.upper_cutoff
        out["diameter"] = orig.diameter_of_disk
    if MODE == "aux2d":
        for name in [
            "A",
            "B",
            "C",
            "mu",
            "alpha",
            "omega",
            "delta",
            "kappa",
            "sigma",
            "r_m",
        ]:
            out[f"const_{name}"] = getattr(orig, name)
        out["upper_cutoff"] = orig.upper_cutoff
        out["diameter"] = orig.diameter_of_disk

elif MODE == "aux3d":
    side = orig.side_of_box
    box = [side, side, side]
    n_sys = 200
    system = rng.uniform(0, side, size=(n_sys, 3))
    from scipy.spatial import cKDTree

    tree = cKDTree(system, boxsize=box)
    coords = rng.uniform(0, side, size=(8, 3))
    dists, idxs = tree.query(
        coords, k=orig.no_neigh, distance_upper_bound=orig.upper_cutoff
    )
    # keep only trials whose neighbour set is fully finite within the first few,
    # mirroring what the live code feeds (finite k>=2 columns used by g3)
    finite2 = np.isfinite(dists[:, :3]).all(axis=1)
    coords, dists, idxs = coords[finite2], dists[finite2], idxs[finite2]
    g_r, ref = orig.calculate_g3_3d(coords, dists, idxs, tree, 1.0)
    out["coords"] = coords
    out["dists"] = dists
    out["idxs"] = idxs
    out["tree_data"] = tree.data
    out["side"] = side
    out["n_trials"] = orig.n_trial_molecules
    out["upper_cutoff"] = orig.upper_cutoff
    out["diameter"] = orig.diameter_of_disk
    out["g3_bulk"] = g_r
    out["g3_ref"] = ref
    # pairwise
    tset = orig.TrialMoleculeSet(
        coords=coords, energy_weight=np.ones(len(coords)), ref_neighbor_coord=ref
    )
    chosen = coords[0] + np.array([0.7, 0.3, -0.2])
    delta = coords - chosen
    delta -= side * np.round(delta / side)
    sq = np.sum(delta**2, axis=1)
    g3p = orig.calculate_g3_pairwise(tset, chosen, sq, 1.0)
    out["pair_chosen"] = chosen
    out["pair_sq"] = sq
    out["g3_pair"] = g3p

np.savez(os.path.join(REFDIR, f"{MODE}.npz"), **out)
print(f"{MODE}: wrote {len(out)} arrays")
