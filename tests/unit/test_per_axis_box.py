"""Phase 2 (A2.1/A2.2/A2.3): the orthorhombic box in the metropolis engine.

The neighbour layer was already per-axis (test_anisotropic_box pins that). These
tests cover the engine path Phase 2 actually threaded:

  reduction    an isotropic list box [L, L, L] is the cube. It must collapse to
               box_sides None and produce the identical run id and the identical
               trajectory as the scalar box L (A2.3): the scalar path is
               untouched, so cubic runs are bit-for-bit what they always were.

  orthorhombic a genuinely anisotropic box runs, keeps every particle inside its
               own per-axis extent, and honours hard-core overlap under the
               per-axis minimum image. Volume is the product of the sides.

  guards       the modes Phase 2 did not convert (sweep samplers, parallel,
               confined pores) refuse an orthorhombic box with a specific
               message rather than silently using the wrong scalar side.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import MC_engine
from gcmc.run.hash_fingerprint import run_id
from gcmc.run.sim_setup import from_params
from gcmc.system.state import ChainState


def _p(box, sampler="metropolis", extra_ens=None, extra_run=None):
    ens = {"dimension": 3, "box": box, "type": "muVT", "mu": 2.5}
    ens.update(extra_ens or {})
    run = {"kind": "single", "radius": 0.5, "sampling_window": 80}
    run.update(extra_run or {})
    return {
        "potential": {"main": "hard_core"},
        "sampler": {"type": sampler},
        "ensemble": ens,
        "run": run,
    }


def _run(params, seed, windows):
    cfg = from_params(params)
    pot = cfg.build_potential()
    wgh = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(cfg.dimension, seed=seed)
    for _ in range(windows):
        MC_engine.advance(st, cfg, pot, wgh)
    return cfg, st


def test_isotropic_list_reduces_to_scalar_box():
    cfg_s = from_params(_p(8.0))
    cfg_l = from_params(_p([8.0, 8.0, 8.0]))
    assert cfg_l.box_sides is None
    assert run_id(cfg_l) == run_id(cfg_s)
    assert cfg_l.total_volume == cfg_s.total_volume

    _, st_s = _run(_p(8.0), seed=7, windows=6)
    _, st_l = _run(_p([8.0, 8.0, 8.0]), seed=7, windows=6)
    np.testing.assert_array_equal(st_s.positions, st_l.positions)


def test_orthorhombic_runs_and_stays_inside_box():
    sides = [10.0, 10.0, 6.0]
    cfg, st = _run(_p(sides), seed=3, windows=15)
    assert cfg.box_sides == (10.0, 10.0, 6.0)
    assert cfg.total_volume == pytest.approx(600.0)
    pos = st.positions
    assert len(pos) > 0
    for a, L in enumerate(sides):
        assert pos[:, a].min() >= 0.0
        assert pos[:, a].max() < L


def test_orthorhombic_respects_hard_core_under_per_axis_mic():
    sides = np.array([12.0, 8.0, 8.0])
    _, st = _run(_p(list(sides)), seed=5, windows=20)
    pos = st.positions
    n = len(pos)
    # per-axis minimum image; hard spheres of diameter 1 (radius 0.5) must not
    # overlap, so every pair distance clears sigma=1 within tolerance
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dr -= sides * np.round(dr / sides)
            assert np.sqrt(dr @ dr) > 1.0 - 1e-9


def test_orthorhombic_run_id_differs_from_nearest_cube():
    ortho = from_params(_p([10.0, 10.0, 6.0]))
    cube = from_params(_p(6.0))  # shares min side but different physics
    assert run_id(ortho) != run_id(cube)


def _branch_params(box):
    # a fully-valid branch config (from the reference modes) so from_params
    # reaches the orthorhombic guard rather than failing earlier
    return {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
        "ensemble": {"mu": 3.0, "dimension": 3, "box": box, "type": "muVT"},
        "run": {"kind": "single", "radius": 0.5, "sampling_window": 60},
    }


def test_orthorhombic_accepted_for_sweep_sampler():
    """A per-axis box is a general feature now, not a metropolis-only one.

    It used to be refused for the sweep samplers, which placed their trial
    bodies against a scalar side. Every sampler and the parallel decomposition
    now take wrap, insertion range and volume through box_dims, so the guard
    that used to fire here is gone. The one surviving exception is a confined
    pore, whose slab/cylinder geometry is defined inside a cubic cell; that is
    the next test.
    """
    cfg = from_params(_branch_params([10.0, 10.0, 6.0]))
    assert tuple(cfg.box_dims) == (10.0, 10.0, 6.0)


def test_orthorhombic_still_rejected_for_a_confined_pore():
    p = _branch_params([10.0, 10.0, 6.0])
    p["ensemble"]["type"] = "confined"
    p["ensemble"]["pore"] = {"geometry": "slit", "H": 4.0}
    with pytest.raises(ValueError, match="cubic"):
        from_params(p)


def test_orthorhombic_rejected_for_parallel():
    base = _p([10.0, 10.0, 6.0])
    base["run"]["kind"] = "parallel"
    with pytest.raises(ValueError, match=r"orthorhombic|parallel"):
        from_params(base)
