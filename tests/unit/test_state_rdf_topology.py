"""Small pure-object guarantees: a fresh chain is empty and reproducibly
seeded; g(r) of an ideal gas is flat at 1; an empty matrix leaves all space
free; and a single-atom Species reports its trivial geometry.
"""

from __future__ import annotations

import numpy as np

from gcmc.analysis.rdf import RDFAccumulator
from gcmc.analysis.widom import free_volume_fraction
from gcmc.forcefield.topology import single_atom
from gcmc.system.state import ChainState


def test_chainstate_fresh_is_empty_and_seed_deterministic():
    a = ChainState.fresh(3, seed=7)
    b = ChainState.fresh(3, seed=7)
    assert a.n == 0 and a.positions.shape == (0, 3)
    assert np.array_equal(a.rng.uniform(size=5), b.rng.uniform(size=5))


def test_rdf_default_r_max_is_half_box():
    acc = RDFAccumulator(box=20.0, dimension=3)
    assert acc.r_max == 10.0


def test_rdf_of_ideal_gas_is_flat_near_one():
    box = 20.0
    rng = np.random.default_rng(0)
    acc = RDFAccumulator(box=box, dimension=3, r_max=6.0, n_bins=30)
    for _ in range(40):
        acc.accumulate(rng.uniform(0, box, (400, 3)))
    r, g, _ = acc.finalize()
    mid = (r > 1.0) & (r < 6.0)  # skip the small-r bins with few pairs
    assert abs(float(np.mean(g[mid])) - 1.0) < 0.1


def test_free_volume_fraction_empty_matrix_is_one():
    empty = np.empty((0, 3))
    assert free_volume_fraction(empty, box=10.0, diameter=1.0) == 1.0


def test_single_atom_species_roundtrip():
    sp = single_atom(dim=3)
    assert sp.n_sites == 1 and sp.dim == 3 and sp.is_single_atom
