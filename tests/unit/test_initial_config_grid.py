"""`initial_config: "grid"` bootstraps from a lattice instead of an empty box.

The token is a keyword, not a path, so it must be resolved before the file /
library dispatch. What the lattice has to guarantee: no pair starts overlapping
(the hard core forbids r <= diameter, so exact contact is already an overlap,
and the lattice must be commensurate with the periodic box or the wrap seam
overlaps), the density lands near the requested one, NVT still gets exactly
target_n, and the constrained ensembles never place a particle inside an
obstacle or a wall. It must also draw no random numbers: the starting
configuration is deterministic, so adding it cannot shift the chain's draws.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc import config_specs, geometry, runner
from gcmc.sim_setup import from_params

_RNG = lambda: np.random.default_rng(0)  # noqa: E731


def _cfg(spec="grid", *, box=12.0, dim=3, pot="hard_core", ens=None, run=None):
    ensemble = {"mu": 2.0, "dimension": dim, "box": box}
    ensemble.update(ens or {})
    if ensemble.get("type") == "NVT":
        ensemble.pop("mu")  # NVT fixes N; mu is undefined and rejected
    r = {"seed": 1}
    if spec is not None:
        r["initial_config"] = spec
    r.update(run or {})
    p = {
        "potential": {"main": pot},
        "sampler": {"type": "rosenbluth", "depth": 0},
        "ensemble": ensemble,
        "run": r,
    }
    if pot == "lennard_jones":
        p["potential"]["T"] = 0.85
    return from_params(p)


def _eta(pos, cfg):
    r = cfg.diameter / 2.0
    v = math.pi * r**2 if cfg.dimension == 2 else 4.0 / 3.0 * math.pi * r**3
    return len(pos) * v / float(np.prod(cfg.box_dims))


def _min_separation(pos, cfg):
    """Smallest pair distance under the run's own periodicity (per-axis, so an
    orthorhombic box is checked on its real sides and not on min(sides))."""
    sides = np.asarray(cfg.box_dims, dtype=float)
    tree = cKDTree(pos % sides, boxsize=sides)
    return float(tree.query(pos % sides, k=2)[0][:, 1].min())


@pytest.mark.parametrize(("dim", "box"), [(2, 20.0), (3, 12.0)], ids=["2d", "3d"])
def test_grid_fills_the_box_at_a_fluid_density(dim, box):
    cfg = _cfg(box=box, dim=dim)
    pos = runner._init_config(cfg, _RNG())
    assert pos.shape[1] == dim
    assert len(pos) > 0
    # dense enough to be a real starting fluid, not so dense it is a crystal
    # locked above freezing (hard disks ~0.70, hard spheres ~0.494)
    lo, hi = (0.30, 0.60) if dim == 2 else (0.25, 0.45)
    assert lo < _eta(pos, cfg) < hi


@pytest.mark.parametrize(
    ("dim", "box", "pot"),
    [(2, 20.0, "hard_core"), (3, 12.0, "hard_core"), (3, 10.0, "lennard_jones")],
)
def test_no_pair_starts_overlapping(dim, box, pot):
    cfg = _cfg(box=box, dim=dim, pot=pot)
    pos = runner._init_config(cfg, _RNG())
    # strictly greater: hard_sphere.weight_pair treats r <= diameter as overlap
    assert _min_separation(pos, cfg) > cfg.diameter


def test_orthorhombic_box_is_commensurate_on_every_axis():
    # min(sides) would tile the long axes wrongly and leave an overlapping seam
    cfg = _cfg(box=[10.0, 14.0, 18.0])
    pos = runner._init_config(cfg, _RNG())
    assert _min_separation(pos, cfg) > cfg.diameter


def test_explicit_packing_fraction_is_honoured():
    dense = runner._init_config(_cfg("grid:0.30"), _RNG())
    sparse = runner._init_config(_cfg("grid:0.10"), _RNG())
    assert len(sparse) < len(dense)
    assert _eta(sparse, _cfg("grid:0.10")) == pytest.approx(0.10, rel=0.25)


@pytest.mark.parametrize("spec", ["grid", "GRID", "  Grid  ", "grid=0.30", "grid:0.30"])
def test_token_is_recognised_and_never_treated_as_a_path(spec):
    # a path arm would raise FileNotFoundError: no such file, and no such library
    pos = runner._init_config(_cfg(spec), _RNG())
    assert len(pos) > 0


def test_grid_draws_no_random_numbers():
    """The lattice is deterministic, so it must not consume the generator: the
    chain that follows has to see exactly the draws it would have seen."""
    rng = _RNG()
    before = rng.bit_generator.state
    runner._init_config(_cfg(), rng)
    assert rng.bit_generator.state == before


def test_nvt_honours_target_n_and_ignores_the_density():
    """Repacking conserves N, so the starting count must BE the target -- the
    grid token must not substitute a density that happens to be nearby."""
    ens = {"N": 100, "type": "NVT"}
    with_token = runner._init_config(_cfg("grid", ens=ens), _RNG())
    without = runner._init_config(_cfg(None, ens=ens), _RNG())
    assert len(with_token) == 100
    assert np.array_equal(with_token, without)


def test_muvt_target_density_sets_the_starting_density():
    cfg = _cfg(ens={"target_eta": 0.20})
    pos = runner._init_config(cfg, _RNG())
    assert _eta(pos, cfg) == pytest.approx(0.20, rel=0.25)


def test_qa_grid_clears_the_frozen_matrix():
    box = 12.0
    matrix = np.random.default_rng(3).uniform(0, box, (300, 3))
    cfg = dataclasses.replace(
        _cfg(ens={"type": "QA"}, run={"matrix_config": "unused.xyz"}),
        matrix_positions=matrix,
    )
    pos = runner._init_config(cfg, _RNG())
    assert len(pos) > 0
    nearest = cKDTree(matrix % box, boxsize=[box] * 3).query(pos % box, k=1)[0]
    assert nearest.min() > cfg.diameter


def test_confined_grid_stays_inside_the_pore():
    cfg = _cfg(ens={"type": "confined", "pore": {"geometry": "slit", "H": 6.0}})
    pos = runner._init_config(cfg, _RNG())
    assert len(pos) > 0
    # `geometry.inside_pore` is the ONE pore predicate now (S2.11). This test
    # previously called `runner._inside_pore_seed`, a verbatim duplicate of it --
    # so it was asserting that the seeder agreed with its own copy of the rule,
    # which it could not fail to do. Against the shared predicate it asserts what
    # it is named for.
    ens = config_specs.ensemble_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    assert all(geometry.inside_pore(p, ens, samp) for p in pos)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("grid:9.9", "above the densest lattice"),
        ("grid:abc", "must be a packing fraction"),
        ("grid:-0.2", "must be positive"),
    ],
)
def test_unbuildable_requests_fail_loudly(spec, message):
    with pytest.raises(ValueError, match=message):
        runner._init_config(_cfg(spec), _RNG())


# --------------------------------------------------------------- parallel mode
# A lattice is the worst case for domain decomposition: its spacing can be
# exactly commensurate with the cell pitch, so a whole plane of molecules can
# land on a cell edge at once instead of the one-off a random configuration
# would give. What must hold is that the decomposition still PARTITIONS the
# molecules -- every one owned by exactly one cell's main region or by the gap
# ("rest") -- because advance_parallel rebuilds state.positions as
# vstack(mains + rest), so a molecule counted twice is created and one counted
# never is destroyed.


def _par_cfg(box=12.0, dim=3, n_cell=8, spec="grid"):
    return from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {
                "type": "rosenbluth",
                "depth": 0,
                "trials": 10,
                "trial_box_side": 3.0,
            },
            "ensemble": {"mu": 4.3, "dimension": dim, "box": box},
            "run": {
                "kind": "parallel",
                "n_cell": n_cell,
                "seed": 5,
                "sampling_window": 100,
                "initial_config": spec,
            },
        }
    )


def _assert_partitions(cfg, pos, rng):
    from gcmc import decomposition

    dim = cfg.dimension
    cells, rest, _ = decomposition.grid_up_nd(
        pos, cfg.box_dims, cfg.cell_dims, cfg.buffer, rng, dim, n_ax=cfg.cell_counts
    )
    owned = [c[0] for c in cells if len(c[0])] + ([rest] if len(rest) else [])
    rows = np.vstack(owned)
    assert len(rows) == len(pos), "molecules lost or duplicated by the decomposition"
    assert len(np.unique(np.round(rows, 12), axis=0)) == len(pos)


@pytest.mark.parametrize(
    ("box", "dim", "n_cell"),
    [
        (12.0, 3, 8),  # pitch 6.0 / spacing 1.2: EXACTLY commensurate
        (20.0, 2, 4),
        (20.0, 2, 16),
        ([10.0, 14.0, 18.0], 3, [2, 2, 2]),  # orthorhombic, per-axis counts
    ],
)
def test_grid_partitions_cleanly_over_many_decomposition_origins(box, dim, n_cell):
    cfg = _par_cfg(box=box, dim=dim, n_cell=n_cell)
    pos = runner._init_config(cfg, _RNG())
    rng = np.random.default_rng(12345)
    for _ in range(50):  # a fresh random cell origin every window
        _assert_partitions(cfg, pos, rng)


def test_grid_partitions_even_when_a_plane_sits_on_a_cell_edge():
    """The adversarial origin, constructed rather than hoped for: `d < cl` is a
    strict comparison, so a molecule exactly on a cell edge must fall into the
    frozen gap, not into both neighbours and not out of the system."""

    class _FixedOrigin:
        """rng stand-in returning a chosen decomposition origin."""

        def __init__(self, origin):
            self.origin = np.asarray(origin, float)

        def uniform(self, lo, hi, size=None):
            return self.origin.copy()

    cfg = _par_cfg()
    pos = runner._init_config(cfg, _RNG())
    spacing = cfg.box_dims[0] / 10  # 1000 sites over 12.0 -> 1.2
    pitch = cfg.box_dims[0] / cfg.cell_counts[0]
    # sites sit at (i + 0.5) * spacing; shift them exactly onto d == cell length
    bad = (0.5 * spacing - cfg.cell_dims[0]) % pitch
    _assert_partitions(cfg, pos, _FixedOrigin([bad] * cfg.dimension))
