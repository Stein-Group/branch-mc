"""`analysis.energy.configuration_energy` against an O(N^2) brute force."""

from __future__ import annotations

import numpy as np

from gcmc.analysis.energy import configuration_energy
from gcmc.forcefield.lennard_jones import lj_pair_energy


def _brute(pos, box, rc, matrix=None, tail=0.0):
    u = 0.0
    pts = [(p, False) for p in pos] + [
        (m, True) for m in (matrix if matrix is not None else [])
    ]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i][1] and pts[j][1]:
                continue  # matrix-matrix is not an energy the chain sees
            dr = pts[i][0] - pts[j][0]
            dr -= box * np.round(dr / box)
            r = float(np.sqrt(dr @ dr))
            if r < rc:
                u += float(lj_pair_energy(np.array([r]))[0])
    return u + tail * len(pos) ** 2


def test_matches_brute_force_with_matrix_and_tail():
    rng = np.random.default_rng(3)
    box = np.array([7.0, 7.0, 7.0])
    # a loose lattice with jitter keeps every pair above the LJ floor
    g = np.stack(np.meshgrid(*[np.arange(0.5, 7.0, 1.4)] * 3), -1).reshape(-1, 3)
    g = g + rng.uniform(-0.2, 0.2, g.shape)
    pos, mat = g[:100], g[100:110]
    got = configuration_energy(pos, box, 3, 2.5, matrix=mat, tail_pref=-0.01)
    assert np.isclose(got, _brute(pos, box, 2.5, mat, -0.01), rtol=1e-12)


def test_empty_box_is_tail_only():
    assert configuration_energy(np.empty((0, 3)), 5.0, 3, 2.5, tail_pref=0.3) == 0.0
