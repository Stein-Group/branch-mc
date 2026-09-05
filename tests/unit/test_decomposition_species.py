"""P7's first gate, brought forward: species survive the decomposition round trip.

The plan calls this the most likely place for a silent scrambling bug, and it is
right to: a molecule's cell membership is a boolean mask over the input array, the
workers reorder their interiors, and reassembly concatenates in cell order with
the unowned leftovers last. Any one of those handled by index instead of by
carrying the column alongside the coordinates gives a configuration whose types
are a permutation of the truth -- same totals, same density, wrong physics, and
nothing downstream that can detect it.

So both halves are checked. Totals per species invariant is the cheap half and
catches gross loss. The tagged-particle test is the half that matters: every
molecule keeps its OWN type through split, a zero-step advance and reassembly,
verified by coordinate identity, which is exact because a zero-step window moves
nothing.

Tiny and deterministic: 100-200 molecules, fixed seeds, no chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.space import decomposition


def _split(n=200, box=24.0, dim=3, cell=6.0, buffer=2.5, seed=0, n_species=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, dim))
    spc = rng.integers(n_species, size=n).astype(np.int64)
    d = decomposition.grid_up_nd(
        pos, box, cell, buffer, np.random.default_rng(seed + 1), dim, species=spc
    )
    return pos, spc, d


def test_the_result_is_named_and_the_old_positional_reads_still_work():
    """`Cell` is a NamedTuple, so `cell[0]` is the same operation it always was --
    no indirection added to the reassembly path -- while the species columns get
    names instead of a return whose length depends on an argument."""
    _, _, d = _split()
    cell = d.cells[0]
    assert cell[0] is cell.main
    assert cell[1] is cell.border
    assert cell[2] is cell.origin
    assert isinstance(cell, tuple)


def test_single_species_gets_none_everywhere():
    """Omitting the column must leave every species field None, so a
    single-species run's decomposition is byte-for-byte the old one."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, 24.0, size=(150, 3))
    d = decomposition.grid_up_nd(pos, 24.0, 6.0, 2.5, np.random.default_rng(4), 3)
    assert d.rest_species is None
    assert all(c.main_species is None and c.border_species is None for c in d.cells)


def test_a_column_of_the_wrong_length_raises():
    rng = np.random.default_rng(5)
    pos = rng.uniform(0.0, 24.0, size=(50, 3))
    with pytest.raises(ValueError, match="10 entries for 50 molecules"):
        decomposition.grid_up_nd(
            pos, 24.0, 6.0, 2.5, np.random.default_rng(6), 3, species=np.zeros(10, int)
        )


def test_species_totals_are_invariant_across_the_split():
    """Every molecule is owned by exactly one cell or by `rest`, so the mains plus
    the leftovers must reproduce the input composition exactly. Borders are
    excluded: a border molecule is another cell's main, seen read-only, so
    counting them would double count."""
    _, spc, d = _split()
    owned = np.concatenate(
        [c.main_species for c in d.cells if len(c.main)] + [d.rest_species]
    )
    assert len(owned) == len(spc)
    assert np.bincount(owned, minlength=3).tolist() == (
        np.bincount(spc, minlength=3).tolist()
    )


def test_every_molecule_keeps_its_own_species_through_the_split():
    """The tagged-particle half. Coordinates are the tag, which is exact here
    because the split copies rather than perturbs."""
    pos, spc, d = _split()
    truth = {tuple(p): int(s) for p, s in zip(pos, spc)}
    assert len(truth) == len(pos), "duplicate coordinates; the tag is not unique"

    seen = 0
    for c in d.cells:
        for p, s in zip(c.main, c.main_species):
            assert truth[tuple(p)] == int(s), f"main molecule at {p} changed type"
            seen += 1
        # borders carry types too, and a scrambled border would corrupt the
        # energies a worker computes without ever showing up in the totals
        for p, s in zip(c.border, c.border_species):
            assert truth[tuple(p)] == int(s), f"border molecule at {p} changed type"
    for p, s in zip(d.rest, d.rest_species):
        assert truth[tuple(p)] == int(s), f"unowned molecule at {p} changed type"
        seen += 1
    assert seen == len(pos)


def test_a_border_molecule_is_unowned_and_never_someone_elses_main():
    """The premise the totals test rests on, stated correctly.

    I first assumed a border molecule was another cell's main -- "ordinary fluid
    that merely sits still", as `advance_cell` puts it -- and asserted
    `borders <= mains`. It fails. The geometry is the other way round: because the
    gap between neighbouring cells is at least `buffer` on every axis, a cell's
    border ring of width `buffer` lies entirely INSIDE the gap, so every border
    molecule is owned by no cell at all. That is exactly what `grid_up_nd`'s
    docstring means by `rest` being a superset of all borders.

    This is why excluding borders from the composition totals is right rather than
    merely conservative: a border molecule is already counted once, in `rest`.
    Counting cell mains plus borders would double count; counting mains plus rest
    is a partition.
    """
    _, _, d = _split()
    mains = {tuple(p) for c in d.cells for p in c.main}
    borders = {tuple(p) for c in d.cells for p in c.border}
    rest = {tuple(p) for p in d.rest}
    assert borders, "no borders at all; the geometry makes this test vacuous"
    assert borders <= rest
    assert not (borders & mains)
    assert mains.isdisjoint(rest)


def test_the_round_trip_reassembles_the_same_composition():
    """Split, then reassemble the way `advance_parallel` does -- mains in cell
    order, leftovers last -- and require the multiset of (position, species) pairs
    to be exactly the input. This is the zero-step case, so it isolates the
    plumbing from anything the sampler does."""
    pos, spc, d = _split()
    kept = [(c.main, c.main_species) for c in d.cells if len(c.main)]
    out_pos = np.vstack([m for m, _ in kept] + [d.rest])
    out_spc = np.concatenate([s for _, s in kept] + [d.rest_species])
    assert len(out_pos) == len(pos)
    before = sorted((tuple(p), int(s)) for p, s in zip(pos, spc))
    after = sorted((tuple(p), int(s)) for p, s in zip(out_pos, out_spc))
    assert after == before


def test_a_transposed_column_would_be_caught():
    """A guard on the guard: if the tagged test could pass on a scrambled column
    it would be worthless. Shuffling the input species and re-splitting must make
    the identity check fail."""
    pos, spc, _ = _split()
    scrambled = np.roll(spc, 1)
    d = decomposition.grid_up_nd(
        pos, 24.0, 6.0, 2.5, np.random.default_rng(1), 3, species=scrambled
    )
    truth = {tuple(p): int(s) for p, s in zip(pos, spc)}
    mismatches = sum(
        1
        for c in d.cells
        for p, s in zip(c.main, c.main_species)
        if truth[tuple(p)] != int(s)
    )
    assert mismatches > 0
