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

  general      the orthorhombic box is now general: every sampler and the
               parallel decomposition wrap, insertion range and volume through
               box_dims. Only a confined pore (slab/cylinder in a cubic cell)
               still refuses an orthorhombic box. The parallel grid tiles per
               axis, so a box can be split into n_x x n_y x n_z cells with
               independent counts (a per-axis cell_length does the same in a
               cubic box).
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import MC_engine
from gcmc.hash_fingerprint import run_id
from gcmc.sim_setup import from_params
from gcmc.state import ChainState


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
    wgh = MC_engine.make_weigher(cfg, pot)
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


def test_orthorhombic_supported_for_branch_sampler():
    # orthorhombic is now general: the branch sampler must build and expose the
    # per-axis box (only a confined pore stays cubic-only).
    cfg = from_params(_branch_params([10.0, 10.0, 6.0]))
    assert cfg.box_dims == [10.0, 10.0, 6.0]
    assert cfg.box == 6.0  # scalar guard side is the minimum


def test_orthorhombic_supported_for_parallel():
    base = _p([10.0, 10.0, 6.0])
    base["run"]["kind"] = "parallel"
    cfg = from_params(base)
    assert cfg.mode == "parallel"
    assert cfg.box_dims == [10.0, 10.0, 6.0]


def test_confined_pore_still_refuses_orthorhombic():
    p = {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "metropolis"},
        "ensemble": {
            "type": "confined",
            "mu": -3.0,
            "dimension": 3,
            "box": [10.0, 10.0, 6.0],
            "pore": {"geometry": "slit", "H": 4.0},
        },
        "run": {"seed": 1, "sampling_window": 60},
    }
    with pytest.raises(ValueError, match=r"confined|cubic"):
        from_params(p)


def _partition_ok(box, cell, buffer, dim, seed):
    """Every main atom belongs to exactly one cell and main+rest covers all: the
    structural guarantee that cells never double-count or drop a molecule."""
    from gcmc import decomposition

    box = np.asarray(box, float)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, (1500, dim))
    cells, rest = decomposition.grid_up_nd(pos, box, cell, buffer, rng, dim)
    seen: set[tuple] = set()
    n_main = 0
    for main, _border, _origin in cells:
        for row in main:
            key = tuple(np.round(row, 9))
            assert key not in seen, "a main atom was claimed by two cells"
            seen.add(key)
            n_main += 1
    assert n_main + len(rest) == len(pos), "main + rest must cover every atom"
    return len(cells)


def test_anisotropic_decomposition_tiles_per_axis():
    # per-axis cell_length in a cubic box: 4 x 4 x 5 = 80
    assert _partition_ok([50.0, 50.0, 50.0], [10.5, 10.5, 8.0], 2.0, 3, 0) == 80
    # anisotropic box with a scalar cell: also 4 x 4 x 5 = 80
    assert _partition_ok([45.0, 45.0, 56.25], 9.25, 2.0, 3, 1) == 80
    # fully general counts, and the cubic default still tiles as a cube
    assert _partition_ok([30.0, 50.0, 40.0], [8.0, 10.0, 8.0], 2.0, 3, 2) == 48
    assert _partition_ok([50.0, 50.0, 50.0], 12.5, 2.0, 3, 3) == 27


def test_cell_length_list_matches_scalar_when_isotropic():
    # an isotropic cell_length list collapses to the scalar path (no drift)
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
            "ensemble": {"mu": 3.0, "dimension": 3, "box": 50.0, "type": "muVT"},
            "run": {"kind": "parallel", "cell_length": [12.5, 12.5, 12.5], "n_workers": 27},
        }
    )
    assert cfg.cell_length_sides is None
    assert cfg.cell_dims == [12.5, 12.5, 12.5]


@pytest.mark.parametrize(
    "sampler",
    [
        {"type": "metropolis"},
        {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 2.0},
        {"type": "rosenbluth", "depth": 0, "trials": 30, "trial_box_side": 2.0},
        {"type": "channel", "depth": 1, "channel_length": 5.0,
         "channel_width": 1.2, "trial_density": 2.0},
        {"type": "channel", "depth": 1, "channel_length": "full",
         "channel_width": 1.2, "trial_density": 2.0},
    ],
    ids=["metropolis", "branch", "rosenbluth", "channel_fixed", "channel_full"],
)
@pytest.mark.parametrize("box", [40.0, [30.0, 40.0]], ids=["box_float", "box_list"])
def test_every_sampler_accepts_scalar_or_list_box(sampler, box):
    """box works as a float (cubic) or a per-axis list (orthorhombic) for every
    sampler; the run builds and the per-axis extent is exposed via box_dims."""
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": sampler,
            "ensemble": {"type": "muVT", "mu": 2.0, "dimension": 2, "box": box},
            "run": {"kind": "single", "seed": 1, "sampling_window": 1},
        }
    )
    want = [float(box)] * 2 if not isinstance(box, list) else box
    assert cfg.box_dims == want


@pytest.mark.parametrize("box", [30.0, [24.0, 30.0]], ids=["box_float", "box_list"])
@pytest.mark.parametrize(
    "cell", [7.0, [7.0, 6.0]], ids=["cell_float", "cell_list"]
)
def test_parallel_accepts_scalar_or_list_box_and_cell(box, cell):
    """The parallel decomposition accepts every combination of scalar/list box
    and scalar/list cell_length, and tiles per axis accordingly."""
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 2.0},
            "ensemble": {"type": "muVT", "mu": 2.0, "dimension": 2, "box": box},
            "run": {"kind": "parallel", "n_workers": 8, "cell_length": cell, "buffer": 2.0},
        }
    )
    want_box = [float(box)] * 2 if not isinstance(box, list) else box
    want_cell = [float(cell)] * 2 if not isinstance(cell, list) else cell
    assert cfg.box_dims == want_box and cfg.cell_dims == want_cell
