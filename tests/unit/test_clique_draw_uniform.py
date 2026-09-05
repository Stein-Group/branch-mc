"""The branch acceptance charges log Z for the group it selects, which is only
correct if the group is drawn UNIFORMLY over the Z non-overlapping k-cliques of
the pool. This checks that against brute-force enumeration: every clique must be
reachable and the draw frequencies must be flat within counting noise.

The predecessor (a deterministic lowest-in-order witness) failed this badly --
at k = 3 more than a third of the cliques had probability exactly zero -- and
that bias showed up as a systematic packing-fraction excess in the equilibrium
density (+0.0069 in eta at 2D mu = 1 with a 2.0 cavity).
"""

from __future__ import annotations

import itertools

import numpy as np

from gcmc.sampling.branch import _build_adj, count_and_sample

CASES = [(2, 13, 2, 4.0), (2, 15, 3, 4.5), (3, 15, 3, 3.2), (3, 16, 4, 3.4)]
DRAWS = 6000


def _brute(points, k, box):
    adj, _ = _build_adj(points, 1.0, box)
    n = len(points)
    return [
        c
        for c in itertools.combinations(range(n), k)
        if all(adj[i][j] for i, j in itertools.combinations(c, 2))
    ]


def test_draw_is_uniform_over_cliques():
    setup = np.random.default_rng(0)
    draw_rng = np.random.default_rng(11)
    for dim, n, k, side in CASES:
        box = np.full(dim, side)
        points = setup.uniform(0, side, size=(n, dim))
        cliques = _brute(points, k, box)
        z = len(cliques)
        assert z > 20, f"degenerate case: only {z} cliques"

        seen = dict.fromkeys(cliques, 0)
        for _ in range(DRAWS):
            count, sample = count_and_sample(
                points, k, 1.0, box_lengths=box, rng=draw_rng
            )
            assert count == z, f"count {count} != brute force {z}"
            assert sample is not None
            key = tuple(sample)
            assert key in seen, f"drew a non-clique {key}"
            seen[key] += 1

        freq = np.array(list(seen.values()), dtype=float)
        expected = DRAWS / z
        # chi-square against a flat draw; z - 1 dof, generous two-sided band so
        # the test is not flaky, but tight enough that zero-probability cliques
        # or an order-of-magnitude skew cannot pass
        chi2 = float(((freq - expected) ** 2 / expected).sum())
        assert 0.4 < chi2 / (z - 1) < 2.2, f"k={k}: chi2/dof={chi2 / (z - 1):.2f}"


def test_rng_free_path_is_still_the_original_witness():
    """The deterministic path survives only for the orig/ count-parity check."""
    points = np.random.default_rng(3).uniform(0, 4.0, size=(14, 2))
    a = count_and_sample(points, 3, 1.0, box_lengths=4.0)
    b = count_and_sample(points, 3, 1.0, box_lengths=4.0)
    assert a == b
