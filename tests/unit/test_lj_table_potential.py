"""A1.5: the table-aware LJ potential against the frozen one it must reduce to.

The load-bearing assertions compare `LennardJonesTable` to a real
`LennardJones` object, not to a formula: with one guest type at sigma* = eps* = 1
every answer must match bit for bit, because a single-species Xe run is supposed
to reuse the frozen "lj" trajectories unchanged. The multi-type tests then check
that types actually change the answer, and that the pore no-tail convention is
enforced rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.forcefield import make_potential
from gcmc.forcefield.lennard_jones import LennardJones
from gcmc.forcefield.lj_table import LennardJonesTable
from gcmc.forcefield.pair_table import PairTable


def _unit_table() -> PairTable:
    return PairTable.from_reduced(("X",), 1, np.array([[1.0]]), np.array([[1.0]]), 1.0)


def _both(beta: float = 1.7, rc: float = 2.5) -> tuple[LennardJones, LennardJonesTable]:
    frozen = LennardJones(beta=beta, upper_cutoff=rc, diameter=1.0, volume=None)
    table = LennardJonesTable(
        beta=beta, table=_unit_table(), trial_type=0, upper_cutoff=rc
    )
    return frozen, table


def test_registered_and_built_through_make_potential():
    t = make_potential("lj_table", beta=1.0, table=_unit_table())
    assert isinstance(t, LennardJonesTable)


def test_weight_pair_reduces_bit_for_bit():
    """1-D distances: the per-pair Boltzmann factor must equal the frozen one to
    the last bit across overlap, well, and beyond-cutoff."""
    frozen, table = _both()
    r = np.array([0.9, 1.0, 1.122, 1.5, 2.0, 2.4999, 2.5, 3.0])
    assert np.array_equal(table.weight_pair(r), frozen.weight_pair(r))


def test_weight_pair_row_product_reduces_bit_for_bit():
    """2-D input (a trial's padded neighbour row) reduces by row product the same
    way, inf pads staying neutral."""
    frozen, table = _both()
    rows = np.array(
        [
            [0.95, 1.3, np.inf, np.inf],
            [1.1, 1.1, 2.2, np.inf],
            [3.0, 3.0, 3.0, 3.0],  # all beyond rc: weight exactly 1
        ]
    )
    assert np.array_equal(table.weight_pair(rows), frozen.weight_pair(rows))
    assert table.weight_pair(rows)[2] == 1.0


def test_weight_bulk_reduces_bit_for_bit():
    frozen, table = _both()
    d = np.array([[0.98, 1.4, np.inf], [1.05, 2.6, 2.6]])
    assert np.array_equal(table.weight_bulk(d), frozen.weight_bulk(d))


def test_total_energy_matches_the_engine_lj_energy():
    """`total_energy` is the table analogue of the engine's `_lj_energy`; at one
    type it returns the same float. Compared here against the same operations the
    engine does: sum of lj_pair_energy over 0 < r < rc."""
    from gcmc.forcefield.lennard_jones import lj_pair_energy

    _, table = _both()
    d = np.array([0.0, 0.97, 1.3, 2.0, 2.5, 3.1])  # includes a self-pair and beyond-rc
    keep = d[(d < 2.5) & (d > 0)]
    expected = float(np.sum(lj_pair_energy(keep)))
    assert table.total_energy(d) == pytest.approx(expected, rel=0, abs=0)


def test_none_neighbour_types_is_the_single_species_path():
    """Passing neighbour_types=None must be identical to the all-same-type array,
    so a caller with no type information (the frozen path) gets the frozen
    answer."""
    _, table = _both()
    r = np.array([0.99, 1.2, 1.8])
    same = np.zeros(len(r), int)
    assert np.array_equal(
        table.weight_pair(r), table.weight_pair(r, neighbour_types=same)
    )
    assert table.total_energy(r) == table.total_energy(r, neighbour_types=same)


def test_types_actually_change_the_answer():
    """A two-type table where the cross pair is deeper: a neighbour of the other
    type must give a different weight than one of the same type, or the type
    plumbing is dead."""
    tab = PairTable.from_reduced(
        ("A", "B"),
        2,
        sigma_star=np.array([[1.0, 1.3], [1.3, 1.0]]),
        eps_star=np.array([[1.0, 3.0], [3.0, 1.0]]),
        core_floor_c=0.8,
    )
    pot = LennardJonesTable(beta=1.0, table=tab, trial_type=0, upper_cutoff=3.0)
    r = np.array([1.5, 1.5])
    same = pot.weight_pair(r, neighbour_types=np.array([0, 0]))
    cross = pot.weight_pair(r, neighbour_types=np.array([1, 1]))
    assert not np.allclose(same, cross)
    # the deeper cross well means a larger Boltzmann factor at this separation
    assert cross[0] > same[0]


def test_mixed_neighbour_types_in_one_row():
    """A trial seeing one A and one B neighbour: the row weight is the product of
    the two correctly-typed pair factors, not one type applied to both."""
    tab = PairTable.from_reduced(
        ("A", "B"),
        2,
        sigma_star=np.array([[1.0, 1.3], [1.3, 1.0]]),
        eps_star=np.array([[1.0, 3.0], [3.0, 1.0]]),
        core_floor_c=0.8,
    )
    pot = LennardJonesTable(beta=1.0, table=tab, trial_type=0, upper_cutoff=3.0)
    r = np.array([1.5, 1.7])
    types = np.array([0, 1])
    got = pot.weight_pair(r, neighbour_types=types)
    wa = pot.weight_pair(np.array([1.5]), neighbour_types=np.array([0]))
    wb = pot.weight_pair(np.array([1.7]), neighbour_types=np.array([1]))
    assert got == pytest.approx(np.array([wa[0], wb[0]]))


def test_no_tail_arguments_are_refused():
    """The pore convention is no-tail. A caller that passes the tail arguments is
    asking for a term this mode does not have, and must be told, not silently
    given a tail-free answer it thinks includes one."""
    _, table = _both()
    d = np.array([[1.1, 1.4]])
    with pytest.raises(ValueError, match="no long-range tail"):
        table.weight_bulk(d, n_particles=10, choice="add")


def test_diameter_is_the_trial_types_self_floor():
    """The overlap radius the engine reads off `.diameter` is c*sigma for the
    trial type against itself, so the frozen overlap test keeps working with a
    single number even though the table is per pair."""
    tab = PairTable.from_reduced(
        ("A", "B"),
        2,
        sigma_star=np.array([[1.0, 2.0], [2.0, 1.5]]),
        eps_star=np.ones((2, 2)),
        core_floor_c=0.75,
    )
    a = LennardJonesTable(beta=1.0, table=tab, trial_type=0, upper_cutoff=3.0)
    b = LennardJonesTable(beta=1.0, table=tab, trial_type=1, upper_cutoff=3.0)
    assert a.diameter == pytest.approx(0.75)  # 0.75 * sigma_AA = 0.75
    assert b.diameter == pytest.approx(0.75 * 1.5)  # 0.75 * sigma_BB
