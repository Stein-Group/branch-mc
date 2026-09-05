"""Item 4: neighbour queries must not silently truncate at a fixed k.

The old energy loops asked a kd-tree for a fixed number of neighbours (k = no_neigh)
and summed only those. In a dense fluid a particle can have more interacting
neighbours than that fixed k, so the tail of the pair sum is dropped and the
energy is quietly wrong. geometry.query_saturating doubles k until the k-th slot
falls outside the cutoff (or every point is returned), so the returned set is
exactly the set of neighbours within the cutoff.

This checks query_saturating against query_ball_point (the exact within-cutoff
set) on a dense cloud, and confirms that a fixed small k really would have missed
neighbours, so the fix addresses a real truncation rather than a hypothetical one.
"""

import numpy as np
from scipy.spatial import cKDTree

from gcmc.space import geometry


def _within_cutoff_sorted(tree, pts, cutoff, box):
    """Ground-truth sorted within-cutoff distances for each query point."""
    out = []
    for p in pts:
        idx = tree.query_ball_point(p, cutoff)
        d = []
        for j in idx:
            v = tree.data[j] - p
            v -= box * np.round(v / box)
            dist = np.sqrt(np.dot(v, v))
            if dist > 1e-12:  # drop self
                d.append(dist)
        out.append(np.sort(np.array(d)))
    return out


def test_saturating_matches_ball_point_and_fixed_k_truncates():
    rng = np.random.default_rng(0)
    box = 10.0
    n = 900  # dense: ~0.9 number density
    cutoff = 2.5
    fixed_k = 6
    pts = rng.uniform(0, box, size=(n, 3))
    tree = cKDTree(pts, boxsize=box)
    query = pts[:200]

    truth = _within_cutoff_sorted(tree, query, cutoff, box)

    dists, _idxs = geometry.query_saturating(tree, query, fixed_k, cutoff)
    assert dists.shape[0] == len(query)

    truncated_rows = 0
    for row_i in range(len(query)):
        finite = np.sort(dists[row_i][np.isfinite(dists[row_i])])
        finite = finite[finite > 1e-12]
        exact = truth[row_i]
        assert finite.size == exact.size, (
            f"row {row_i}: saturating returned {finite.size} within-cutoff "
            f"neighbours, ball_point says {exact.size}"
        )
        if finite.size:
            assert np.allclose(finite, exact, atol=1e-9)
        # would a fixed k of fixed_k have captured everything within cutoff here?
        d_fixed, _ = tree.query([query[row_i]], k=fixed_k, distance_upper_bound=cutoff)
        got = np.isfinite(d_fixed[0]).sum()
        if got < exact.size:
            truncated_rows += 1

    print(
        f"checked {len(query)} query points; fixed k={fixed_k} would have "
        f"truncated {truncated_rows} of them"
    )
    assert truncated_rows > 0, (
        "test cloud not dense enough to exercise truncation; fixed k never "
        "missed a neighbour so the saturating path is untested"
    )


def test_saturating_equivalent_when_unsaturated():
    """When k already exceeds the neighbour count, the saturating result is the
    same as a single fixed-k query, so sparse-system behaviour is unchanged."""
    rng = np.random.default_rng(1)
    box = 30.0
    pts = rng.uniform(0, box, size=(40, 3))  # very sparse
    tree = cKDTree(pts, boxsize=box)
    query = pts[:10]
    cutoff = 2.5
    k = 8
    d_sat, _i_sat = geometry.query_saturating(tree, query, k, cutoff)
    d_fix, _i_fix = tree.query(query, k=k, distance_upper_bound=cutoff)
    assert np.array_equal(np.isfinite(d_sat), np.isfinite(d_fix))
    assert np.allclose(d_sat[np.isfinite(d_sat)], d_fix[np.isfinite(d_fix)])


if __name__ == "__main__":
    test_saturating_matches_ball_point_and_fixed_k_truncates()
    test_saturating_equivalent_when_unsaturated()
    print("PASS test_item04_saturation")
