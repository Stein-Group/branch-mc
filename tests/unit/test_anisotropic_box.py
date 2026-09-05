"""A2.2/A2.4: does the neighbour layer already work in a non-cubic box?

Phase 2 has to turn the scalar cubic side into per-axis lengths. The grep
inventory says the two lowest layers already take a vector: `CellList.__init__`
broadcasts `box` to (dim,) and `_pair_dists` does its minimum image per axis off
`self.box[a]`, and the engine hands `cKDTree` a `boxsize` array built from
`cfg.box_dims`. That is a reading of the code, and a reading is not a
measurement. These tests are the measurement, and they stay afterwards as the
permanent A2.4 parity suite.

Brute force is the oracle: O(N^2) distances with an explicit per-axis minimum
image, no cell list, no tree. Boxes are deliberately anisotropic (30 x 30 x 24
in reduced units, and a nastier 31 x 17 x 23) with points placed hard against
each boundary, because the failure mode this guards is an axis whose length got
read off the wrong index.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc.space.neighbors import CellList

BOXES = [
    np.array([30.0, 30.0, 24.0]),  # the plan's example
    np.array([31.0, 17.0, 23.0]),  # no two axes equal, none a multiple of another
    np.array([24.0, 30.0, 30.0]),  # the short axis first, in case axis 0 is special
]
RC = 2.5


def brute_pairs(points: np.ndarray, box: np.ndarray, rc: float) -> list[list[float]]:
    """Every neighbour within rc of every point, per-axis minimum image, O(N^2)."""
    out = []
    for p in points:
        dr = points - p
        dr -= box * np.round(dr / box)
        d = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        out.append(sorted(float(x) for x in d[(d < rc) & (d > 0.0)]))
    return out


def cl_pairs(cl: CellList, points: np.ndarray, rc: float) -> list[list[float]]:
    """The same lists as `brute_pairs`, read off the cell list.

    The self-hit at distance 0 is dropped so the two agree on content; what is
    under test is which neighbours are in range and how far, not the padding.
    """
    dists, _ = cl.query(points, rc)
    return [
        sorted(float(x) for x in row[np.isfinite(row) & (row > 0.0)]) for row in dists
    ]


def _corners_and_seams(box: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Random points, plus pairs straddling each boundary.

    For each axis a pair sits at 0.1 and L - 0.1 along that axis: 0.2 apart
    through the wall, and L - 0.2 apart if the minimum image is wrong. With
    rc = 2.5 that is the difference between "neighbour" and "invisible".
    """
    pts = [rng.uniform(0, box, size=(n, 3))]
    for a in range(3):
        lo = np.array([1.0, 2.0, 3.0])
        hi = lo.copy()
        lo[a], hi[a] = 0.1, box[a] - 0.1
        pts.append(np.array([lo, hi]))
    pts.append(np.array([[0.05, 0.05, 0.05], box - 0.05]))  # the corner pair
    return np.vstack(pts)


@pytest.mark.parametrize("box", BOXES, ids=lambda b: "x".join(str(int(v)) for v in b))
def test_cell_list_matches_brute_force_in_an_anisotropic_box(box):
    rng = np.random.default_rng(7)
    pts = _corners_and_seams(box, rng, 400)
    cl = CellList.from_positions(pts, box, 3, RC)
    got = cl_pairs(cl, pts, RC)
    want = brute_pairs(pts, box, RC)
    for i, (g, w) in enumerate(zip(got, want)):
        assert len(g) == len(w), (
            f"point {i} at {pts[i]}: {len(g)} neighbours, brute force says {len(w)}"
        )
        assert np.allclose(g, w, atol=1e-12), f"point {i} at {pts[i]}"


@pytest.mark.parametrize("box", BOXES, ids=lambda b: "x".join(str(int(v)) for v in b))
def test_pairs_across_every_boundary_are_seen(box):
    """The seam pairs specifically: each is 0.2 apart through its own wall. A
    cell list that used the wrong axis length would report L - 0.2 and drop them."""
    rng = np.random.default_rng(8)
    pts = _corners_and_seams(box, rng, 0)
    cl = CellList.from_positions(pts, box, 3, RC)
    dists, _ = cl.query(pts, RC)
    for i in range(0, 6, 2):
        row = dists[i][np.isfinite(dists[i]) & (dists[i] > 0.0)]
        assert len(row) >= 1, f"seam pair {i // 2} lost"
        assert row.min() == pytest.approx(0.2, abs=1e-12)


@pytest.mark.parametrize("box", BOXES, ids=lambda b: "x".join(str(int(v)) for v in b))
def test_cell_list_agrees_with_ckdtree_in_an_anisotropic_box(box):
    """The engine's other neighbour route is `cKDTree(..., boxsize=cfg.box_dims)`.
    Both take a vector today, so both must give the same answer on the same box;
    if Phase 2 only fixes one of them, this catches it."""
    rng = np.random.default_rng(9)
    pts = _corners_and_seams(box, rng, 300)
    cl = CellList.from_positions(pts, box, 3, RC)
    tree = cKDTree(np.mod(pts, box), boxsize=box)
    for i, p in enumerate(pts):
        d, _ = tree.query(np.mod(p, box), k=len(pts), distance_upper_bound=RC)
        want = sorted(float(x) for x in d[np.isfinite(d) & (d > 0.0)])
        got = cl_pairs(cl, pts[i : i + 1], RC)[0]
        assert np.allclose(got, want, atol=1e-9), f"point {i} at {p}"


def test_cubic_box_is_unchanged_by_the_vector_route():
    """The reduction A2.3 will need: a scalar side and the same side as a vector
    must give bit-identical distances, so the per-axis path can replace the cubic
    one without moving a single trajectory."""
    rng = np.random.default_rng(10)
    L = 20.0
    pts = rng.uniform(0, L, size=(500, 3))
    scalar = CellList.from_positions(pts, L, 3, RC)
    vector = CellList.from_positions(pts, np.array([L, L, L]), 3, RC)
    ds, _ = scalar.query(pts, RC)
    dv, _ = vector.query(pts, RC)
    assert np.array_equal(ds, dv)


def test_anisotropic_cell_grid_is_per_axis():
    """The grid itself: nc = floor(L_a / rc) per axis, so a 30 x 30 x 24 box at
    rc = 2.5 is 12 x 12 x 9 cells of width 2.5 x 2.5 x 2.667, not a cube.

    This is the structural claim behind the parity above. If nc were built from a
    scalar the counts would come out equal and the distances would still look
    right for points far from a wall, which is exactly the bug that hides.
    """
    cl = CellList.from_positions(np.zeros((1, 3)), np.array([30.0, 30.0, 24.0]), 3, RC)
    assert list(cl.nc) == [12, 12, 9]
    assert cl.cw == pytest.approx([2.5, 2.5, 24.0 / 9.0])
