"""Reading the g3 table without interpolating it.

The table lookup was the auxiliary's largest single cost: the interpolator
located every query with a binary search per axis and blended four corners, and
the first Rosenbluth level issues one query per trial per neighbour. The grid is
REGULAR, so `_NearestG3` locates a point by arithmetic instead and reads the one
closest tabulated value. It is now the only reader; the interpolator survives in
THIS FILE as a reference oracle, which is the right place for it -- the claims
made for the coarse reader are claims about how far it sits from the fine one,
and they need the fine one to be checkable without shipping it.

WHAT HAS TO HOLD is not accuracy. g3 is a selection bias and it is divided back
out of the acceptance ratio, so any strictly positive function of the geometry
leaves the sampled ensemble exactly unchanged. Its resolution is a variance
knob, like `min_g3`. What has to hold is that the function be

  TOTAL over the same domain as the interpolator, so that switching the key
    changes the resolution and nothing else, and out-of-range pairs stay
    unbiased,
  EXACT on the tabulated points, so the coarse reader is the table and not an
    approximation to it,
  DETERMINISTIC and floored, because the adaptive solver reads log g3 and the
    reverse pass must read the same numbers as the forward one.

The accuracy tests below are CHARACTERISATION with loose bounds: they say the
departure is a fraction of a percent in the median, which is the claim made for
the key, not that any particular value is right.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from gcmc.forcefield.auxiliary import G3_READER, HardCoreAux3D, _NearestG3
from gcmc.run import sim_setup

G3_CSV = "data/g3_analysis_data.csv"
FLOOR = 0.005


def _aux(min_g3=FLOOR):
    """The production reader, as `build` assembles it."""
    return HardCoreAux3D.build(
        diameter=1.0,
        upper_cutoff=2.5,
        side=20.0,
        n_trials=200,
        g3_csv=G3_CSV,
        min_g3=min_g3,
    )


def _oracle(min_g3=FLOOR):
    """The interpolator the reader replaced, built here from the same table and
    the same floor. Not a second production path: a yardstick."""
    r_u, cos_u, grid = _table(min_g3)
    return RegularGridInterpolator(
        (r_u, cos_u), grid, method="linear", bounds_error=False, fill_value=1.0
    )


def _table(min_g3=FLOOR):
    d = np.loadtxt(G3_CSV, delimiter=",", skiprows=1)
    r_u, cos_u = np.unique(d[:, 0]), np.unique(d[:, 2])
    grid = d[:, 4][np.lexsort((d[:, 2], d[:, 0]))].reshape(len(r_u), len(cos_u))
    return r_u, cos_u, np.where(grid < min_g3, min_g3, grid)


def _nodes():
    r_u, cos_u, grid = _table()
    rr, cc = np.meshgrid(r_u, cos_u, indexing="ij")
    return np.column_stack([rr.ravel(), cc.ravel()]), grid.ravel()


def test_the_nearest_reader_returns_the_table_exactly_on_its_own_nodes() -> None:
    """The coarse reader IS the table, not an approximation to it. Bit for bit,
    because a node lookup is an index and not an arithmetic result."""
    pts, values = _nodes()
    got = _aux()._interp(pts)
    assert np.array_equal(got, values)


def test_the_two_readers_agree_on_the_nodes() -> None:
    """Which pins the two against each other somewhere they must not differ: a
    query that lands on a tabulated point has nothing to interpolate."""
    pts, _ = _nodes()
    lin = _oracle()(pts)
    near = _aux()._interp(pts)
    assert np.allclose(near, lin, rtol=0.0, atol=1e-12)


def test_the_domain_is_identical_so_the_same_pairs_stay_unbiased() -> None:
    """Out of the tabulated range both readers return exactly 1.0, an unbiased
    pair. If the domains differed, switching the key would change WHICH pairs
    are biased at all, which is a bigger change than the resolution and would be
    invisible in an average."""
    rng = np.random.default_rng(4)
    pts = np.column_stack(
        [rng.uniform(-0.5, 3.5, 20000), rng.uniform(-1.5, 1.5, 20000)]
    )
    lin = _oracle()(pts)
    near = _aux()._interp(pts)
    assert np.array_equal(lin == 1.0, near == 1.0)
    assert (near[near == 1.0] == 1.0).all()


def test_a_missing_reference_direction_lands_on_the_fill_value() -> None:
    """`weight_pair` sends NaN coordinates for a trial with no reference
    neighbour and relies on the reader returning 1.0 for them. A comparison
    against NaN is false, so the out-of-range mask catches it -- asserted here
    because the alternative is an index built from a NaN, which is a crash."""
    pts = np.array([[np.nan, 0.0], [1.2, np.nan], [np.nan, np.nan]])
    assert np.array_equal(_aux()._interp(pts), np.ones(3))


def test_the_departure_from_the_interpolated_value_is_a_fraction_of_a_percent() -> None:
    """Characterisation, with a loose bound. The grid is 52 x 100 and g3 is
    smooth across a cell, so the nearest tabulated point is close; the tail is
    wider near contact, where g3 is steep."""
    rng = np.random.default_rng(5)
    pts = np.column_stack([rng.uniform(0.0, 2.7, 40000), rng.uniform(-1, 1, 40000)])
    lin = _oracle()(pts)
    near = _aux()._interp(pts)
    inside = lin != 1.0
    rel = np.abs(near[inside] - lin[inside]) / lin[inside]
    assert np.median(rel) < 0.02
    assert np.quantile(rel, 0.95) < 0.5


def test_the_floor_reaches_the_table_before_the_lookup() -> None:
    """`min_g3` floors the grid in `build`, so no lookup can return less than
    the floor -- which the adaptive solver depends on, because it takes a log."""
    pts, _ = _nodes()
    for floor in (0.005, 0.25):
        got = HardCoreAux3D.build(
            diameter=1.0,
            upper_cutoff=2.5,
            side=20.0,
            n_trials=200,
            g3_csv=G3_CSV,
            min_g3=floor,
        )._interp(pts)
        assert got.min() >= floor


def test_the_reader_is_deterministic_across_repeated_calls() -> None:
    rng = np.random.default_rng(6)
    pts = np.column_stack([rng.uniform(0.0, 2.7, 500), rng.uniform(-1, 1, 500)])
    aux = _aux()
    first = aux._interp(pts)
    for _ in range(5):
        assert np.array_equal(aux._interp(pts), first)


def test_a_row_order_change_only_reorders_the_answer() -> None:
    """Element-wise by construction, and worth pinning: the engine permutes the
    trial pool, and a lookup that mixed rows would make g3 a function of the
    permutation."""
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.uniform(0.0, 2.7, 300), rng.uniform(-1, 1, 300)])
    aux = _aux()
    ref = aux._interp(pts)
    perm = rng.permutation(len(pts))
    assert np.array_equal(aux._interp(pts[perm]), ref[perm])


def test_an_unevenly_spaced_axis_is_refused_with_the_axis_named() -> None:
    """The arithmetic assumes a constant step. A table that broke that would
    otherwise be read at confidently wrong indices, which is the worst kind of
    failure: silent, deterministic and plausible."""
    good = np.linspace(0.0, 1.0, 5)
    bad = np.array([0.0, 0.1, 0.4, 0.9, 2.0])
    grid = np.ones((5, 5))
    with pytest.raises(ValueError, match="r axis of the g3 table"):
        _NearestG3(bad, good, grid)
    with pytest.raises(ValueError, match="cos theta axis of the g3 table"):
        _NearestG3(good, bad, grid)


def test_one_node_per_axis_is_refused_because_there_is_no_step() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        _NearestG3(np.array([1.0]), np.linspace(0, 1, 4), np.ones((1, 4)))


# --------------------------------------------------------------------------
# The identity marker. There is no config key: one reader, no choice to state.
# What there IS, is the fact that the reader changed, and a run id that does not
# move across that change would let an old pmf directory be resumed by a chain
# proposing differently.
# --------------------------------------------------------------------------


def _params(aux_extra=None, dimension=3, aux=True):
    block = {"type": "pmf", "min_g3": 0.005}
    if dimension == 3:
        block["g3_csv"] = G3_CSV
    else:
        block.pop("min_g3")
    if aux_extra:
        block.update(aux_extra)
    p = {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "rosenbluth", "trials": 30},
        "ensemble": {"mu": 2.0, "dimension": dimension, "box": 10.0},
        "run": {"kind": "single", "seed": 1, "sampling_window": 10},
    }
    if aux:
        p["auxiliary"] = block
    return p


def test_a_three_dimensional_pmf_run_builds_the_nearest_reader() -> None:
    """No key to set: `build_potential` has one reader and this is it."""
    cfg = sim_setup.from_params(_params())
    assert isinstance(cfg.build_potential()._interp, _NearestG3)


def test_the_marker_is_on_exactly_the_runs_that_read_a_table() -> None:
    """A marker that appeared on runs with no table would split run ids for
    chains that are bit-identical, which is the opposite failure and just as
    bad: the baseline arm of a comparison would stop being resumable."""
    assert sim_setup.from_params(_params()).g3_reader == G3_READER
    assert sim_setup.from_params(_params(dimension=2)).g3_reader is None
    assert sim_setup.from_params(_params(aux=False)).g3_reader is None


def test_the_marker_reaches_the_run_identity() -> None:
    """The reader is not settable, so this cannot be shown by varying it. What
    can be shown is that it is hashed: a 3D pmf run's identity dict carries it,
    so the id moved when the reader did, and an old directory will not be
    resumed by a chain that proposes differently."""
    from gcmc.run.hash_fingerprint import identity_dict

    ident = identity_dict(sim_setup.from_params(_params()))
    assert ident.get("g3_reader") == G3_READER
    assert "g3_reader" not in identity_dict(sim_setup.from_params(_params(aux=False)))


def test_the_retired_key_is_refused_rather_than_ignored() -> None:
    """A params file written against the pilot version named the reader. Silently
    ignoring the key would let its author believe a choice is in force when there
    is no choice left to make."""
    with pytest.raises(ValueError, match="unknown keys"):
        sim_setup.from_params(_params({"g3_lookup": "linear"}))
