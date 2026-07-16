"""Diff branch.py against the original count_and_sample / count_and_sample_branch.

The clique code reads no module globals, so one import in any mode suffices.
Both count and the returned sample are deterministic, so this is a direct diff
over a range of k and several random configurations, in 2D and 3D.
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
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "..", "..", "orig")
sys.path.insert(0, ORIG)
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.samplers import branch

fails = 0
checks = 0
for dim in (2, 3):
    for seed in range(6):
        rg = np.random.default_rng(100 * dim + seed)
        # a modest box with radius 0.5; mix of overlapping and not so cliques vary
        box = 6.0
        n = rg.integers(8, 22)
        centers = rg.uniform(0, box, size=(n, dim))
        for k in (1, 2, 3, 4, 5):
            oc, osamp = orig.count_and_sample(centers, k, 0.5, box_lengths=box)
            mc, msamp = branch.count_and_sample(centers, k, 1.0, box_lengths=box)
            checks += 1
            if oc != mc or osamp != msamp:
                fails += 1
                print(
                    f"[FAIL] count_and_sample dim={dim} seed={seed} k={k}: "
                    f"orig=({oc},{osamp}) mine=({mc},{msamp})"
                )
            # branch variant: k and k+1 in one build
            ock, ock1, osk, osk1 = orig.count_and_sample_branch(
                centers, k, 0.5, box_lengths=box
            )
            mck, mck1, msk, msk1 = branch.count_and_sample_branch(
                centers, k, 1.0, box_lengths=box
            )
            checks += 1
            if (ock, ock1, osk, osk1) != (mck, mck1, msk, msk1):
                fails += 1
                print(f"[FAIL] branch dim={dim} seed={seed} k={k}")

print(f"\n{checks - fails}/{checks} checks matched")
if fails == 0:
    print("ALL BRANCH CHARACTERIZATION TESTS PASSED")
else:
    raise SystemExit(1)
