"""The two geometry-aware weighers survive an empty environment.

WHY THIS EXISTS. `MC_engine.advance` passes `tree = cl if cl.n_active else None`,
so an empty exclusion set -- no fluid left and no frozen matrix -- reaches every
weigher as `kdtree=None` with fully padded neighbour rows. The distance-only
weighers have always handled that: an infinite distance contributes zero energy
and unit weight, which is why the ordinary LJ and hard-core paths can start from
an empty box.

The two GEOMETRY-AWARE weighers could not. `reweighting.calculate_order_parameter`
dereferenced `kdtree.data`; `HardCoreAux3D.weight_bulk` asserted `kdtree is not None`.
Between them that is psi6 reweighting and the pmf auxiliary -- and the feature
matrix reported it as eight failures across four modes.

WHAT IT WAS AND WAS NOT. It was recorded as "the cell-list migration removed the
KD-tree and neither feature was ever ported". Measured: the port is fine. Both
run correctly against a `CellList` the moment the system holds a single particle
-- `CellList.data` exists precisely so `idxs` index it the way they indexed a
kd-tree's. What was missing was the N = 0 branch, which is a much smaller claim
and a much better-understood one: not two dead features, one unhandled state that
every run passes through on its way up from an empty box.

These tests pin the state directly rather than through a whole window, so a
future edit that reintroduces the dereference fails here, at the weigher, instead
of somewhere inside an engine loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.sampling import reweighting


def _padded(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """What `find_neighbors` returns when the exclusion set is empty: every row
    an infinite distance and the pad index."""
    return np.full((n, k), np.inf), np.zeros((n, k), dtype=np.int64)


def test_psi6_on_an_empty_environment_returns_the_zero_phasor() -> None:
    """Zero sums with zero counts, which `psi6_magnitude` reads as the liquid
    floor -- the same answer it already gives an isolated particle in a sparse
    system. The empty box must not get a second rule."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dists, idxs = _padded(3, 4)

    cos_sum, sin_sum, counts = reweighting.calculate_order_parameter(
        coords, dists, idxs, None, 10.0, 1.0
    )

    assert np.array_equal(cos_sum, np.zeros(3))
    assert np.array_equal(sin_sum, np.zeros(3))
    assert np.array_equal(counts, np.zeros(3))

    # and the magnitude that follows is the documented floor, not a NaN
    psi6 = reweighting.psi6_magnitude(cos_sum, sin_sum, counts)
    assert np.allclose(psi6, 0.3)
    assert np.isfinite(psi6).all()


def test_psi6_agrees_with_the_sparse_case_it_generalises() -> None:
    """An empty system and a system whose particles have no neighbours in range
    must give the same psi6. That equality is the whole argument for the values
    chosen above, so it is asserted rather than described."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    empty = reweighting.psi6_magnitude(
        *reweighting.calculate_order_parameter(coords, *_padded(2, 4), None, 10.0, 1.0)
    )

    # non-empty container, but every neighbour row still beyond the cutoff
    class _Far:
        data = np.array([[100.0, 100.0]])

    sparse = reweighting.psi6_magnitude(
        *reweighting.calculate_order_parameter(
            coords, *_padded(2, 4), _Far(), 10.0, 1.0
        )
    )
    assert np.allclose(empty, sparse)


def test_the_pmf_auxiliary_on_an_empty_environment_applies_no_bias() -> None:
    """An empty product is 1: the g3 factor is three-body, and with no
    neighbours there is no triplet to weight."""
    aux = pytest.importorskip("gcmc.forcefield.aux")
    HardCoreAux3D = aux.HardCoreAux3D

    obj = HardCoreAux3D.__new__(
        HardCoreAux3D
    )  # no g3 table needed: the empty path reads none
    obj.side = 10.0
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dists, idxs = _padded(2, 5)

    w, ref = obj.weight_bulk(dists, coords=coords, idxs=idxs, kdtree=None)

    assert np.array_equal(w, np.ones(2)), "an empty triplet set must not bias"
    assert ref.shape == coords.shape
    assert np.isnan(ref).all(), (
        "the reference neighbour is UNDEFINED here and must say so. Zero would "
        "not be neutral: weight_pair builds vec_BA = ref - trial, so a zero "
        "reference is a real direction pointing at the origin and the per-pair "
        "recompute would apply a definite bias derived from nothing."
    )


def test_the_pmf_per_pair_recompute_honours_the_undefined_reference() -> None:
    """`weight_pair` must take the same no-bias branch, or the bulk weight and
    the recompute would disagree on the same move."""
    aux = pytest.importorskip("gcmc.forcefield.aux")
    obj = aux.HardCoreAux3D.__new__(aux.HardCoreAux3D)
    obj.side = 10.0

    trial = np.array([[1.0, 1.0, 1.0]])
    out = obj.weight_pair(
        np.array([4.0]),
        trial_coords=trial,
        ref_coords=np.full((1, 3), np.nan),
        chosen_coord=np.array([[2.0, 2.0, 2.0]]),
    )
    assert np.array_equal(out, np.ones(1))
