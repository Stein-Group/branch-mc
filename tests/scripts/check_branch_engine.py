"""BRANCH engine check.

BRANCH counts insertion configurations by exact clique enumeration instead of
sequential Rosenbluth insertion. The clique counter itself is verified exactly
in check_branch.py; this checks the wiring into the acceptance. BRANCH and
Rosenbluth are two estimators of the same grand-canonical ensemble, so at a
fixed chemical potential they must reach the same equilibrium packing fraction.
A wrong k_trials term or a mis-scaled configuration count would shift that
plateau.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gcmc import MC_engine as engine
from gcmc import config_specs
from gcmc.sim_setup import from_params
from gcmc.state import ChainState

NW = 36


def run(branch, nwin=NW):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 3.0, "dimension": 3, "box": 20.0},
        "sampler": {
            "type": ("branch" if branch else "rosenbluth"),
            "trials": 30,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "run": {"sampling_window": 1000, "seed": 1},
    }
    cfg = from_params(p)
    pot = cfg.build_potential()
    w = engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(3, seed=1)
    st.positions = np.random.default_rng(2).uniform(1, 19, size=(40, 3))
    return [engine.advance(st, cfg, pot, w)["packing_fraction"] for _ in range(nwin)]


eb = run(True)
assert eb[-1] > eb[0], "branch run did not grow toward equilibrium"
print("  ok  BRANCH path runs and equilibrates")

en = run(False)
mb = float(np.mean(eb[-12:]))
mn = float(np.mean(en[-12:]))
diff = abs(mb - mn)
print(
    f"  branch equilib eta = {mb:.4f}   rosenbluth equilib eta = {mn:.4f}   diff = {diff:.4f}"
)
assert diff < 0.02, f"BRANCH and Rosenbluth disagree by {diff:.4f}"
print("  ok  BRANCH and Rosenbluth reach the same equilibrium density")

print("\nALL BRANCH-ENGINE TESTS PASSED")
