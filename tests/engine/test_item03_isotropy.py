"""Item 3: the three translation components must be independent.

The old code set the z displacement to (ranf1 + ranf2) % 1 built from the same
two uniforms that made x and y, so a proposed move lived on a 2D surface and dz
was a deterministic function of dx + dy. Each marginal stays uniform, so per-axis
variance alone cannot see the defect; the coupling between dz and (dx, dy) can.
Drawing a third independent uniform removes it.

Two independent checks:

1. Record the actual uniform(0, 1, 3) draw the translate step makes (its
   signature is unique in the muVT path, so nothing else is captured) and confirm
   the three columns are uncorrelated with equal spread. The old code drew only
   two uniforms, so this also fails loudly if the third draw is ever removed.

2. Drive a dilute system, recover the accepted per-move displacements from
   position differences, and confirm corr(dz, dx), corr(dz, dy), corr(dz, dx+dy)
   are all ~0. This sees the realised z displacement, not just the raw draw.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics
from gcmc.run import sim_setup
from gcmc.system.state import ChainState

# Slow tier: both tests are statistical (correlation tolerances 0.03-0.04 need
# large samples, ~136s total). The default tier still guards that the translate
# step draws three independent uniforms via the stage-A RNG-consumption oracle.
pytestmark = pytest.mark.slow

BOX = 8.0
DIM = 3
MU = -3.0  # <N> ~ 21: dilute, so a move is tiny vs spacing
N_STEPS = 120_000
BURN = 2_000


class _RecordingRNG:
    """Delegates to a real Generator but records every uniform(0, 1, 3) draw,
    which in the muVT engine is exactly the 3D translation displacement draw."""

    def __init__(self, gen, log):
        self._gen = gen
        self._log = log

    def uniform(self, *a, **k):
        v = self._gen.uniform(*a, **k)
        low = a[0] if len(a) > 0 else k.get("low", 0.0)
        high = a[1] if len(a) > 1 else k.get("high", 1.0)
        size = a[2] if len(a) > 2 else k.get("size")
        if low == 0 and high == 1 and size == 3:
            self._log.append(np.asarray(v, float).copy())
        return v

    def __getattr__(self, name):
        return getattr(self._gen, name)


def _make(seed):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": MU, "dimension": DIM, "box": BOX},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
        },
        "engine": {"depth": 0},
        "run": {"sampling_window": 1},
    }
    cfg = sim_setup.from_params(p)
    pot = cfg.build_potential()
    w = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(DIM, seed=seed)
    st.positions = np.empty((0, DIM))
    return cfg, pot, w, st


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_translation_draws_three_independent_uniforms():
    cfg, pot, w, st = _make(seed=11)
    log = []
    st.rng = _RecordingRNG(st.rng, log)
    steps = 0
    while len(log) < 40_000 and steps < 400_000:
        metropolis_dynamics.advance(st, cfg, pot, w)
        steps += 1
    u = np.array(log)
    assert len(u) > 5_000, (
        f"only {len(u)} translation draws of size 3 seen; the third displacement "
        f"component may not be drawn independently"
    )
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    c12, c13, c23 = _corr(u1, u2), _corr(u1, u3), _corr(u2, u3)
    print(
        f"[draw] n={len(u)} corr(u1,u2)={c12:+.3f} corr(u1,u3)={c13:+.3f} "
        f"corr(u2,u3)={c23:+.3f} means={u.mean(axis=0).round(3)}"
    )
    for name, c in (("u1,u2", c12), ("u1,u3", c13), ("u2,u3", c23)):
        assert abs(c) < 0.03, f"drawn components correlated: corr({name})={c:+.3f}"
    for j in range(3):
        assert abs(u[:, j].mean() - 0.5) < 0.02, "draw not uniform on [0,1]"


def test_translation_displacement_is_isotropic():
    cfg, pot, w, st = _make(seed=7)
    box = cfg.box
    disp = []
    for i in range(N_STEPS + BURN):
        before = st.positions.copy()
        metropolis_dynamics.advance(st, cfg, pot, w)
        after = st.positions
        if i < BURN or len(before) == 0 or len(after) != len(before):
            continue
        tree = cKDTree(before, boxsize=box)
        dd, jj = tree.query(after, k=1)
        for a_row, j, dist in zip(after, jj, dd):
            if dist > 1e-9:
                d = a_row - before[j]
                d -= box * np.round(d / box)
                disp.append(d)
    d = np.array(disp)
    assert len(d) > 10_000, f"too few accepted moves to test ({len(d)})"
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    c_zx, c_zy, c_zs = _corr(dz, dx), _corr(dz, dy), _corr(dz, dx + dy)
    vx, vy, vz = dx.var(), dy.var(), dz.var()
    print(
        f"[disp] n={len(d)} corr(dz,dx)={c_zx:+.3f} corr(dz,dy)={c_zy:+.3f} "
        f"corr(dz,dx+dy)={c_zs:+.3f} var=({vx:.4f},{vy:.4f},{vz:.4f})"
    )
    for name, c in (("dz,dx", c_zx), ("dz,dy", c_zy), ("dz,dx+dy", c_zs)):
        assert abs(c) < 0.04, f"corr({name})={c:+.3f} not ~0 (z still coupled)"
    assert abs(vz / vx - 1.0) < 0.06 and abs(vz / vy - 1.0) < 0.06, (
        f"per-axis variance mismatch: {vx:.4f} {vy:.4f} {vz:.4f}"
    )


if __name__ == "__main__":
    test_translation_draws_three_independent_uniforms()
    test_translation_displacement_is_isotropic()
    print("PASS test_item03_isotropy")
