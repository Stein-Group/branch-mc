"""Gate P3 of multispecies_plan.md: the species column through the hot path.

Two tests, and the plan calls them the most valuable in the whole document.
Neither runs a chain; both are pure bookkeeping or pure arithmetic, both are
sub-second, and both catch failures that would otherwise surface as a wrong
isotherm several phases later with nothing pointing at the cause.

  1. SLOT ALIGNMENT. `CellList` holds particle state in slots, with a tombstone
     protocol and a free list that RECLAIMS slots at rebuild. So a species column
     can drift from the coordinates it is supposed to describe in a way no test
     of coordinates would notice: the slot is right, the position is right, and
     the type is the previous tenant's. This drives a scripted sequence of
     inserts, removes, moves, tombstones and reactivations against an
     independently maintained dict and compares every live slot.

  2. BRUTE FORCE. Cell-list total energy and overlap verdict against an O(N^2)
     minimum-image reference, to the last bit, on a three-species configuration.
     Three species rather than two because three is the smallest number that
     catches "the second species is special" (M13): an asymmetric k_ij, and a
     cross term that is weak for one pair and strong for another.

The reference energy goes through `table_pair_energy_sq`, the centralised
force-field layer, not through an inline 12-6 formula. An independent
implementation of the LOOP is what makes it a reference; an independent
implementation of the POTENTIAL would just be a second thing that can be wrong,
and the second AST guard forbids it anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.neighbors import CellList
from gcmc.potentials.pair_table import PairTable, table_pair_energy_sq
from gcmc.species_file import load_forcefield, reduce_tables

THREE = "ff/species_three_reduced.yaml"


@pytest.fixture(scope="module")
def table() -> PairTable:
    ff = load_forcefield(THREE)
    red = reduce_tables(ff, 1.2)
    assert red.core_floor_c is not None
    return PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, red.core_floor_c
    )


# ------------------------------------------------------------- slot alignment


def test_every_live_slot_keeps_its_own_species_through_500_mutations():
    """Pure bookkeeping, no physics. The reference is a plain dict, maintained by
    the test in the obvious way, so a disagreement is unambiguously the cell
    list's.

    The mutation mix is deliberate. Removes leave tombstones; a later insert may
    reclaim one, which is the only path by which a slot's species can be stale.
    Reactivations undo a remove and must restore the ORIGINAL species, not a
    default. Moves must leave species alone, because a translation does not change
    what a particle is. Rebuilds reorder the CSR and free the tombstones, and are
    triggered here both explicitly and implicitly by crossing the pending
    threshold.
    """
    rng = np.random.default_rng(4)
    box, dim, n_species = 8.0, 3, 3
    cl = CellList(box, dim, 2.5, capacity=32, pending_threshold=17)
    ref: dict[int, int] = {}  # slot -> species, live slots only
    tombstoned: dict[int, int] = {}  # slot -> species, removable again

    for step in range(500):
        act = rng.random()
        if act < 0.35 or not ref:
            sp = int(rng.integers(n_species))
            s = int(cl.insert(rng.uniform(0.0, box, dim), species=sp))
            # A reclaimed slot must no longer be counted as tombstoned, and the
            # new species overwrites whatever the previous tenant had.
            tombstoned.pop(s, None)
            ref[s] = sp
        elif act < 0.55:
            s = int(rng.choice(list(ref)))
            tombstoned[s] = ref.pop(s)
            cl.remove(s)
        elif act < 0.70 and tombstoned:
            s = int(rng.choice(list(tombstoned)))
            ref[s] = tombstoned.pop(s)
            cl.reactivate(s)
        elif act < 0.95:
            s = int(rng.choice(list(ref)))
            cl.move(s, rng.uniform(0.0, box, dim))
        else:
            cl.rebuild()
            # rebuild frees every tombstone; those slots are no longer the
            # caller's and may be handed to the next insert.
            tombstoned.clear()

        live = cl.active_slots()
        assert set(live.tolist()) == set(ref), f"step {step}: live set diverged"
        got = cl.species[live]
        want = np.array([ref[int(s)] for s in live], dtype=np.int64)
        assert got.tolist() == want.tolist(), f"step {step}: species misaligned"

    assert len(ref) > 20, "the mix degenerated; the test stopped exercising anything"


def test_growth_carries_the_species_column():
    """`_grow` copies each slot-indexed array by name. A column missing from that
    list is zeroed on the first reallocation, which reads as "everything is
    species 0" -- a plausible-looking single-species run rather than a crash."""
    cl = CellList(6.0, 3, 2.5, capacity=16)
    rng = np.random.default_rng(0)
    want = []
    for _ in range(200):  # forces several _grow calls off capacity 16
        sp = int(rng.integers(3))
        cl.insert(rng.uniform(0.0, 6.0, 3), species=sp)
        want.append(sp)
    assert cl.capacity > 16, "capacity never grew; the test proves nothing"
    live = cl.active_slots()
    assert cl.species[live].tolist() == want


def test_a_species_column_of_the_wrong_length_raises():
    cl = CellList(6.0, 3, 2.5)
    pts = np.zeros((4, 3))
    with pytest.raises(ValueError, match="3 entries for 4 points"):
        cl.bulk_insert(pts, species=np.array([0, 1, 2]))


def test_the_default_is_type_zero_everywhere():
    """A single-species run must read exactly what it read before this column
    existed, which is what makes the P4 identity snapshot still hold."""
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.0, 8.0, size=(50, 3))
    cl = CellList.from_positions(pos, 8.0, 3, 2.5)
    assert not cl.species[cl.active_slots()].any()


def test_species_at_clamps_the_pad_index():
    cl = CellList.from_positions(np.zeros((3, 3)), 6.0, 3, 2.5, species=[2, 1, 0])
    pad = cl.capacity  # the out-of-range index the queries hand out
    got = cl.species_at(np.array([0, 1, 2, pad]))
    assert got[:3].tolist() == [2, 1, 0]
    assert len(got) == 4  # the pad read something rather than raising


# ---------------------------------------------------------------- brute force


def _reference_energy(
    pos: np.ndarray, sp: np.ndarray, box: float, rc: float, table: PairTable
) -> float:
    """O(N^2) minimum-image total energy, truncated at rc.

    Written as a plain double loop over pairs on purpose. It is slow and it does
    not care: at 200 particles it is milliseconds, and the point of a reference
    is that its loop shares no structure with the thing under test.
    """
    n = len(pos)
    total = 0.0
    for i in range(n):
        d = pos[i + 1 :] - pos[i]
        d -= box * np.round(d / box)
        d2 = np.einsum("ij,ij->i", d, d)
        for off, dd2 in enumerate(d2):
            if dd2 < rc * rc:
                total += float(table_pair_energy_sq(dd2, sp[i], sp[i + 1 + off], table))
    return total


def _cell_list_energy(cl: CellList, rc: float, table: PairTable) -> float:
    """Total energy from the cell list, each pair counted once.

    Uses the same per-slot query the engine uses, then halves the double count,
    so the quantity being compared is the one the hot path actually computes.
    """
    total = 0.0
    for s in cl.active_slots():
        nb, d = cl.neighbours_within(cl.coords[s], rc)
        si = int(cl.species[s])
        for j, dj in zip(nb, d):
            if int(j) == int(s):
                continue
            total += float(table_pair_energy_sq(dj * dj, si, int(cl.species[j]), table))
    return 0.5 * total


def test_cell_list_energy_matches_the_brute_force_reference(table):
    """The single most valuable test in the plan. 200 particles, three species,
    box 8, one fixed seed.

    Positions are drawn with a rejection floor at the largest pair core floor, so
    the configuration is admissible for every pair and the comparison is not
    dominated by one near-singular contact whose last bits are meaningless.
    """
    rng = np.random.default_rng(7)
    box, dim, rc, n = 8.0, 3, 2.5, 200
    floor = float(table.core_floor.max())

    pos = np.empty((0, dim))
    while len(pos) < n:
        cand = rng.uniform(0.0, box, size=(4 * n, dim))
        for p in cand:
            if len(pos):
                d = pos - p
                d -= box * np.round(d / box)
                if np.sqrt(np.einsum("ij,ij->i", d, d)).min() <= floor:
                    continue
            pos = np.vstack([pos, p])
            if len(pos) == n:
                break
    sp = rng.integers(len(table.types), size=n)

    cl = CellList.from_positions(pos, box, dim, rc, species=sp)
    assert cl.species[cl.active_slots()].tolist() == sp.tolist()

    got = _cell_list_energy(cl, rc, table)
    want = _reference_energy(pos, sp, box, rc, table)
    # Both sums run over the same pairs with the same per-pair function, but in a
    # different ORDER, so they agree to floating-point associativity and not to
    # the bit. The tolerance is that of the summation, not of the physics.
    assert got == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_the_overlap_verdict_matches_pair_for_pair(table):
    """The overlap floor is per pair now, `core_floor[si, sj]`. A configuration
    admissible for the small-small pair can be an overlap for the large-large
    one at the same distance, so a scalar floor gets a whole species wrong in one
    direction. Verdicts must agree exactly -- this is a boolean, so there is no
    tolerance to hide behind."""
    rng = np.random.default_rng(11)
    box, dim, rc, n = 8.0, 3, 2.5, 200
    pos = rng.uniform(0.0, box, size=(n, dim))
    sp = rng.integers(len(table.types), size=n)
    cl = CellList.from_positions(pos, box, dim, rc, species=sp)

    # reference: every pair, minimum image, its own floor
    ref = set()
    for i in range(n):
        d = pos[i + 1 :] - pos[i]
        d -= box * np.round(d / box)
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        for off, dd in enumerate(dist):
            j = i + 1 + off
            if dd <= table.core_floor[sp[i], sp[j]] - 1e-12:
                ref.add((i, j))

    got = set()
    for s in cl.active_slots():
        nb, d = cl.neighbours_within(cl.coords[s], rc)
        for j, dj in zip(nb, d):
            if int(j) == int(s):
                continue
            a, b = sorted((int(s), int(j)))
            if dj <= table.core_floor[sp[a], sp[b]] - 1e-12:
                got.add((a, b))

    assert got == ref
    assert ref, "no overlapping pairs at all; the test verified nothing"
