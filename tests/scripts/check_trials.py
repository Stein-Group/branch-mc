"""Diff TrialSet against the original TrialMoleculeSet on identical op sequences.

Imports the original in a benign mode (hs3d, no data file needed) purely to get
its TrialMoleculeSet class, then runs the same operations on both and compares
every field. Plain asserts.
"""

import os
import sys

import numpy as np

sys.argv = [
    "orig",
    "--mu",
    "4",
    "--rho_k",
    "400",
    "--d",
    "0",
    "--r",
    "1.5",
    "--T",
    "1.0",
]
from _frozen import add_orig_to_path

ORIG = str(add_orig_to_path())  # R1: resolved from $ORIG_ROOT or the tag
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(ORIG, ".."))
from gcmc.proposals.trials import TrialSet

FIELDS = ["coords", "energy_weight", "cossum", "sinsum", "counts", "ref_neighbor_coord"]


def cmp(a_orig, b_new, tag):
    for f in FIELDS:
        oa = getattr(a_orig, "_" + f)
        nb = getattr(b_new, f)
        if not np.array_equal(np.asarray(oa), np.asarray(nb)):
            raise AssertionError(f"[FAIL] {tag}: field {f} differs")
    if len(a_orig) != len(b_new):
        raise AssertionError(f"[FAIL] {tag}: length differs")
    print(f"  ok  {tag}")


def make(n, dim, seed):
    rg = np.random.default_rng(seed)
    coords = rg.uniform(0, 10, size=(n, dim))
    ew = rg.uniform(0.1, 2.0, size=n)
    cs = rg.uniform(-1, 1, size=n)
    sn = rg.uniform(-1, 1, size=n)
    ct = rg.uniform(1, 5, size=n)
    ref = rg.uniform(0, 10, size=(n, dim))
    return coords, ew, cs, sn, ct, ref


def build_both(args):
    o = orig.TrialMoleculeSet(*[a.copy() for a in args])
    n = TrialSet(*[a.copy() for a in args])
    return o, n


for dim in (2, 3):
    print(f"dim={dim}")
    args = make(12, dim, seed=dim)
    o, n = build_both(args)
    cmp(o, n, f"construct dim={dim}")

    # integer getitem returns the 5-tuple
    to, tn = o[3], n[3]
    assert len(to) == len(tn) == 5, "int getitem must be a 5-tuple"
    for k in range(5):
        assert np.array_equal(np.asarray(to[k]), np.asarray(tn[k])), (
            f"int getitem elem {k}"
        )
    print("  ok  int getitem 5-tuple")

    # first()
    fo, fn = o.first(), n.first()
    for k in range(5):
        assert np.array_equal(np.asarray(fo[k]), np.asarray(fn[k])), f"first elem {k}"
    print("  ok  first()")

    # mask
    m = np.array(
        [True, False, True, True, False, False, True, True, False, True, False, True]
    )
    cmp(o.mask(m), n.mask(m), f"mask dim={dim}")

    # copy then mutate-delete on the copy
    oc, nc = o.copy(), n.copy()
    oc.delete(4)
    nc.delete(4)
    cmp(oc, nc, f"delete (mutates) dim={dim}")
    # original untouched by the copy's delete
    cmp(o, n, f"copy isolation dim={dim}")

    # merge two sets
    args2 = make(5, dim, seed=dim + 100)
    o2, n2 = build_both(args2)
    cmp(o.merge(o2), n.merge(n2), f"merge dim={dim}")

    # array getitem (permutation)
    perm = np.array([11, 0, 5, 3, 9, 1, 2, 8, 7, 4, 10, 6])
    cmp(o[perm], n[perm], f"array getitem dim={dim}")

print("\nALL CONTAINER CHARACTERIZATION TESTS PASSED")
