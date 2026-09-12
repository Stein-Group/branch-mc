"""Reweighting checks.

Two layers. First, the order-parameter core is compared exactly against the
original: the 6-fold phasor sums and the bridge that gathers PBC neighbour
displacements are exec'd from the original source and must match
gcmc.reweighting term for term. Second, an end-to-end check runs the original
with reweighting_flag flipped on and confirms the 2D hexatic-biased run reaches
the same equilibrium packing fraction as the engine's reweight path; a wrong
selection weight or acceptance term would move that plateau.
"""

import glob
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..", "..")
from _frozen import add_orig_to_path

ORIG = str(add_orig_to_path())  # R1: resolved from $ORIG_ROOT or the tag
sys.path.insert(0, WORK)
from gcmc import config_specs
from gcmc.sampling import reweighting as RW


def ok(m):
    print("  ok ", m)


# ---- exact order-parameter match against the original ----------------------
with open(os.path.join(ORIG, "cav_n_d_3d_g3_LJ.py")) as _f:
    src = _f.read().splitlines()
ns = {"np": np}
# calculate_psi_6_2d (805-852) and calculate_order_parameter (926-981)
exec(compile("\n".join(src[804:852]), "<psi6>", "exec"), ns)
exec(compile("\n".join(src[925:981]), "<orderparam>", "exec"), ns)

rng = np.random.default_rng(0)
side = 75.0
mol = rng.uniform(0, side, size=(400, 2))
tree = cKDTree(mol, boxsize=[side, side])
trials = rng.uniform(0, side, size=(60, 2))
dists, idxs = tree.query(trials, k=40, distance_upper_bound=3.7545)

mc, ms, mc_cnt = RW.calculate_order_parameter(trials, dists, idxs, tree, side, 1.0)
tc, ts, tcnt = ns["calculate_order_parameter"](trials, dists, idxs, tree, side, 1.0)
assert np.allclose(mc, tc) and np.allclose(ms, ts) and np.array_equal(mc_cnt, tcnt)
ok("calculate_order_parameter matches the original (cos, sin, counts)")

# psi6 magnitude with the original's 0.3 liquid floor
psi = RW.psi6_magnitude(mc, ms, mc_cnt)
with np.errstate(divide="ignore", invalid="ignore"):
    ref = np.sqrt((tc / tcnt) ** 2 + (ts / tcnt) ** 2)
ref = np.where((ref == 0) | np.isnan(ref), 0.3, ref)
assert np.allclose(psi, ref)
assert np.all(psi > 0)
ok("psi6 magnitude matches, with the 0.3 isotropic floor")


# ---- end-to-end: same equilibrium as the original with reweighting on ------
def seed_config(n, box, dim, seed=999):
    rg = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        for p in rg.uniform(0, box, size=(n, dim)):
            if not pts:
                pts.append(p)
                continue
            d = np.array(pts) - p
            d -= box * np.round(d / box)
            if np.all(np.sum(d * d, axis=1) >= 1.0):
                pts.append(p)
            if len(pts) >= n:
                break
    return np.array(pts[:n])


MC = 24000
tmp = tempfile.mkdtemp()
patched = os.path.join(tmp, "orig_rw.py")
shutil.copy(os.path.join(ORIG, "cav_n_d_3d_g3_LJ.py"), patched)
with open(patched) as _f:
    txt = _f.read().replace("reweighting_flag = False", "reweighting_flag = True", 1)
with open(patched, "w") as _f:
    _f.write(txt)

old_argv, old_cwd, old_path = sys.argv, os.getcwd(), list(sys.path)
sys.path.insert(0, ORIG)  # for the parallelization import
sys.argv = [
    "orig_rw",
    "--mu",
    "2.0",
    "--k",
    "40",
    "--d",
    "0",
    "--r",
    "2.5",
    "--2d",
    "--T",
    "1.0",
]
os.chdir(tmp)
spec = importlib.util.spec_from_file_location("orig_rw", patched)
orig = importlib.util.module_from_spec(spec)
sys.modules["orig_rw"] = orig
spec.loader.exec_module(orig)
assert orig.reweighting_flag and orig.two_dim_flag
orig.main(MC, seed_config(300, orig.side_of_box, 2).copy())
data = np.loadtxt(sorted(glob.glob("*.log"))[0], comments="#")
oeta = data[:, 1]
mo = float(oeta[len(oeta) // 3 :].mean())

sys.argv, old = old_argv, None
os.chdir(old_cwd)
sys.path[:] = old_path

from gcmc.engine import metropolis_dynamics as engine
from gcmc.run import sim_setup as config
from gcmc.system.state import ChainState

cfg = config.from_params(
    {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 2.0, "dimension": 2},
        "sampler": {
            "type": "reweight",
            "trials": 40,
            "trial_box_side": 5.0,
            "depth": 0,
        },
        "run": {"sampling_window": 1000},
    }
)
assert cfg.box == orig.side_of_box, "box mismatch between engine and original"
pot = cfg.build_potential()
w = engine.make_weigher(config_specs.kernel_spec(cfg), pot)
st = ChainState.fresh(2, seed=12345)
st.positions = seed_config(300, cfg.box, 2)
me = np.array(
    [
        engine.advance(st, cfg, pot, w)["packing_fraction"]
        for _ in range(MC // cfg.sampling_window)
    ]
)
mm = float(me[len(me) // 3 :].mean())
diff = abs(mo - mm)
print(
    f"  orig(reweight) equilib eta = {mo:.4f}   engine equilib eta = {mm:.4f}   diff = {diff:.4f}"
)
assert diff < 0.02, f"reweighting equilibrium disagrees by {diff:.4f}"
ok("engine reweight reaches the same equilibrium as the original (reweight on)")

print("\nALL REWEIGHTING TESTS PASSED")
