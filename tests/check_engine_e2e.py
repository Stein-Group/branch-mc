"""End-to-end: mean packing fraction, engine vs the original main(), per mode.

  python check_engine_e2e.py <mode> <mc_cycles>

Runs the original main() for mc_cycles (it takes the count as an argument),
parses the packing-fraction column from its log, then runs the new engine at
the same parameters for the same number of windows. The exact trajectories
differ (the RNG sources differ by design), so the test compares the
equilibrated mean packing fraction within statistical error.
"""
import os
import sys
import glob
import time
import tempfile
import numpy as np

MODE = sys.argv[1]
MC = int(float(sys.argv[2]))
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "..", "orig")
G3 = os.path.join(ORIG, "g3_analysis_data.csv")

ARGV = {
    "hs3d": ["--mu", "3.0", "--k", "30", "--d", "0", "--r", "1.5", "--T", "1.0"],
    "lj3d": ["--mu", "-3.5", "--k", "30", "--d", "0", "--r", "1.5", "--LJ_pot", "--T", "1.2"],
    "hs2d": ["--mu", "1.0", "--k", "30", "--d", "0", "--r", "1.5", "--2d", "--T", "1.0"],
    "aux2d": ["--mu", "2.0", "--k", "30", "--d", "0", "--r", "1.5", "--2d", "--aux_pot", "--T", "1.0"],
    "aux3d": ["--mu", "2.0", "--k", "30", "--d", "0", "--r", "1.5", "--aux_pot", "--T", "1.0"],
}
PARAMS = {
    "hs3d": {"potential": {"main": "hard_core"}, "ensemble": {"mu": 3.0, "dimension": 3},
             "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
             "run": {"sampling_window": 1000}},
    "lj3d": {"potential": {"main": "lennard_jones", "T": 1.2}, "ensemble": {"mu": -3.5, "dimension": 3},
             "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
             "run": {"sampling_window": 1000, "lower_cutoff": 1.0}},   # original floors LJ at the diameter
    "hs2d": {"potential": {"main": "hard_core"}, "ensemble": {"mu": 1.0, "dimension": 2},
             "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
             "run": {"sampling_window": 1000}},
    "aux2d": {"potential": {"main": "hard_core"}, "auxiliary": {"type": "pmf"},
              "ensemble": {"mu": 2.0, "dimension": 2},
              "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
              "run": {"sampling_window": 1000}},
    "aux3d": {"potential": {"main": "hard_core"}, "auxiliary": {"type": "pmf", "g3_csv": G3},
              "ensemble": {"mu": 2.0, "dimension": 3},
              "sampler": {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0},
              "run": {"sampling_window": 1000}},
}
# rough equilibrium N to seed both runs from, removing the fill-up transient
SEED_N = {"hs3d": 3000, "lj3d": 970, "hs2d": 2600, "aux2d": 3000, "aux3d": 3900}
DIM = {m: PARAMS[m]["ensemble"]["dimension"] for m in PARAMS}


def seed_config(n, box, dim, seed=999):
    rg = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:                     # crude non-overlapping placement
        cand = rg.uniform(0, box, size=(n, dim))
        for p in cand:
            if not pts:
                pts.append(p); continue
            d = np.array(pts) - p
            d -= box * np.round(d / box)
            if np.all(np.sum(d * d, axis=1) >= 1.0):
                pts.append(p)
            if len(pts) >= n:
                break
    return np.array(pts[:n])

# ---- original ----
sys.argv = ["orig"] + ARGV[MODE]
sys.path.insert(0, ORIG)
workdir = tempfile.mkdtemp()
os.chdir(workdir)
# aux reads g3_analysis_data.csv from the working directory (3D only, but the
# 2D analytic path ignores it); make it available before import
import shutil  # noqa: E402
shutil.copy(G3, os.path.join(workdir, "g3_analysis_data.csv"))
import cav_n_d_3d_g3_LJ as orig  # noqa: E402

t0 = time.time()
init = seed_config(SEED_N[MODE], orig.side_of_box, DIM[MODE])
orig.main(MC, init.copy())
t_orig = time.time() - t0

logf = sorted(glob.glob("*.log"))
data = np.loadtxt(logf[0], comments="#")
orig_eta = data[:, 1]

# ---- engine ----
sys.path.insert(0, os.path.join(HERE, ".."))
from gcmc import sim_setup as config, MC_engine as engine          # noqa: E402
from gcmc.state import ChainState        # noqa: E402

cfg = config.from_params(PARAMS[MODE])
pot = cfg.build_potential()
w = engine.make_weigher(cfg, pot)
st = ChainState.fresh(cfg.dimension, seed=12345)
st.positions = seed_config(SEED_N[MODE], cfg.box, cfg.dimension)
n_windows = MC // cfg.sampling_window
mine_eta = []
t0 = time.time()
for _ in range(n_windows):
    mine_eta.append(engine.advance(st, cfg, pot, w)["packing_fraction"])
t_mine = time.time() - t0
mine_eta = np.array(mine_eta)


def stats(x):
    eq = x[len(x) // 3:]                       # drop first third as equilibration
    return eq.mean(), eq.std(ddof=1) / np.sqrt(len(eq)), len(eq)


mo, so, no = stats(orig_eta)
mm, sm, nm = stats(mine_eta)
diff = abs(mo - mm)
comb_se = np.sqrt(so ** 2 + sm ** 2)

print(f"mode={MODE} windows orig={len(orig_eta)} mine={len(mine_eta)} "
      f"(orig {t_orig:.1f}s, engine {t_mine:.1f}s)")
print(f"  original eta = {mo:.4f} +/- {so:.4f}  (n={no})")
print(f"  engine   eta = {mm:.4f} +/- {sm:.4f}  (n={nm})")
print(f"  |diff| = {diff:.4f}   combined SE = {comb_se:.4f}   diff/SE = {diff/comb_se:.2f}")

if diff <= max(4 * comb_se, 0.01):
    print(f"PASS {MODE}: means agree within statistical error")
else:
    print(f"FAIL {MODE}: means differ beyond tolerance")
    sys.exit(1)
