"""Characterize the new potentials against the original code.

For each mode it loads refs/<mode>.npz (produced by refgen.py from the original)
and asserts the new potential reproduces the original numbers. Plain asserts,
no test framework, so the output is short.
"""
import os
import numpy as np
from scipy.spatial import cKDTree

from gcmc.potentials import make_potential
from gcmc.potentials.aux import aux_constants_2d, Aux2D, Aux3D
from gcmc.potentials.lennard_jones import LennardJones
from gcmc.potentials.hard_sphere import HardSphere

REFDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs")


def load(mode):
    return dict(np.load(os.path.join(REFDIR, f"{mode}.npz"), allow_pickle=True))


def same(a, b, tag, rtol=1e-9, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        d = np.nanmax(np.abs(a - b))
        raise AssertionError(f"[FAIL] {tag}: max abs diff {d:.3e}")
    print(f"  ok  {tag}")


def test_lj3d():
    r = load("lj3d")
    pot = LennardJones(beta=float(r["BETA"]), upper_cutoff=float(r["upper_cutoff"]),
                       diameter=float(r["diameter"]), volume=float(r["Total_volume"]))
    same(pot.weight_pair(r["r2"]), r["w2_g1.0"], "lj3d weight_pair 2D")
    same(pot.weight_pair(r["r1"]), r["w1_g1.0"], "lj3d weight_pair 1D")
    same(pot.tail_pref, float(r["lj_tail_pref"]), "lj3d tail_pref")
    # tail weight against the original formula
    beta, tp = float(r["BETA"]), float(r["lj_tail_pref"])
    for N in (0, 5, 37):
        for choice, dU in (("add", tp * (2 * N + 1)), ("remove", -tp * (2 * N - 1))):
            same(pot.tail_weight(N, choice), np.exp(-beta * dU), f"lj3d tail {choice} N={N}")


def test_hs(mode):
    r = load(mode)
    pot = HardSphere(diameter=1.0)
    same(pot.weight_pair(r["r2"]), r["w2_g1.0"], f"{mode} weight_pair 2D (count)")
    same(pot.weight_pair(r["r1"]), r["w1_g1.0"], f"{mode} weight_pair 1D (indicator)")


def test_aux2d():
    r = load("aux2d")
    c = aux_constants_2d(eta=0.69)
    for name in ["A", "B", "C", "mu", "alpha", "omega", "delta", "kappa", "sigma", "r_m"]:
        same(c[name], float(r[f"const_{name}"]), f"aux2d const {name}")
    pot = Aux2D(diameter=float(r["diameter"]), upper_cutoff=float(r["upper_cutoff"]), params=c)
    for g in (1.0, 0.5):
        same(pot.weight_pair(r["r2"], g), r[f"w2_g{g}"], f"aux2d weight_pair 2D gamma={g}")
        same(pot.weight_pair(r["r1"], g), r[f"w1_g{g}"], f"aux2d weight_pair 1D gamma={g}")


def test_aux3d():
    r = load("aux3d")
    side = float(r["side"])
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orig", "g3_analysis_data.csv")
    pot = Aux3D.build(diameter=float(r["diameter"]), upper_cutoff=float(r["upper_cutoff"]),
                      side=side, n_trials=int(r["n_trials"]), g3_csv=csv)
    tree = cKDTree(r["tree_data"], boxsize=[side, side, side])
    w, ref = pot.weight_bulk(r["dists"], coords=r["coords"], idxs=r["idxs"], kdtree=tree, gamma=1.0)
    same(w, r["g3_bulk"], "aux3d g3 bulk weight")
    same(ref, r["g3_ref"], "aux3d g3 reference coords")
    g3p = pot.weight_pair(r["pair_sq"], trial_coords=r["coords"], ref_coords=r["g3_ref"],
                          chosen_coord=r["pair_chosen"])
    same(g3p, r["g3_pair"], "aux3d g3 pairwise weight")


if __name__ == "__main__":
    print("lj3d");  test_lj3d()
    print("hs2d");  test_hs("hs2d")
    print("hs3d");  test_hs("hs3d")
    print("aux2d"); test_aux2d()
    print("aux3d"); test_aux3d()
    print("\nALL POTENTIAL CHARACTERIZATION TESTS PASSED")
