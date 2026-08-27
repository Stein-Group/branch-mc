"""M19's prerequisite: `TrialSet` keeps species row-aligned through everything.

`TrialSet`'s own docstring says the row alignment IS the data structure and that
any new field must be added to every method. This checks that claim for the
species column, method by method, because the exchange pool is assembled by

    pool = trial_set.mask(km).merge(mol_in_cav)
    pool = pool[rng.permutation(len(pool))]

-- a mask, a merge and a fancy index, in that order. A species column held
anywhere other than inside the object would survive the first two by luck and be
destroyed by the permutation, and the symptom would be a composition that drifts
for no visible reason.

The permutation test is the one that matters. The rest are the cheap checks that
make a failure localisable to one method instead of to the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.trials import TrialSet


def _ts(n: int = 8, dim: int = 3, species=None) -> TrialSet:
    """A set whose every field is distinguishable per row, so a misalignment
    shows up as a mismatch rather than as a coincidence."""
    coords = np.arange(n * dim, dtype=float).reshape(n, dim)
    return TrialSet(
        coords,
        np.arange(n, dtype=float) + 0.5,
        species=np.arange(n, dtype=np.int64) % 3 if species is None else species,
    )


def _row_key(ts: TrialSet, i: int) -> tuple:
    """Everything about row i that should travel together."""
    return (tuple(ts.coords[i]), float(ts.energy_weight[i]), int(ts.species[i]))


def test_the_default_is_type_zero_and_the_field_exists():
    """Omitting species must give the single-species case, so every existing
    positional construction keeps meaning what it meant."""
    ts = TrialSet(np.zeros((5, 3)), np.ones(5))
    assert ts.species.shape == (5,)
    assert ts.species.dtype == np.int64
    assert not ts.species.any()


def test_a_wrong_length_column_raises():
    with pytest.raises(ValueError, match="3 entries for 5 trials"):
        TrialSet(np.zeros((5, 3)), np.ones(5), species=np.zeros(3, dtype=np.int64))


def test_species_is_last_so_positional_construction_is_unchanged():
    """Every caller builds a TrialSet positionally with up to six arguments. If
    species were inserted anywhere earlier, one of those arguments would silently
    land in the wrong field."""
    n, dim = 4, 3
    ts = TrialSet(
        np.zeros((n, dim)),
        np.ones(n),
        np.full(n, 2.0),
        np.full(n, 3.0),
        np.full(n, 4.0),
        np.full((n, dim), 5.0),
    )
    assert ts.cossum.tolist() == [2.0] * n
    assert ts.sinsum.tolist() == [3.0] * n
    assert ts.counts.tolist() == [4.0] * n
    assert not ts.species.any()


def test_mask_keeps_alignment():
    ts = _ts(8)
    keep = np.array([True, False, True, True, False, False, True, False])
    want = [_row_key(ts, i) for i in np.flatnonzero(keep)]
    out = ts.mask(keep)
    assert [_row_key(out, i) for i in range(len(out))] == want


def test_fancy_index_keeps_alignment():
    ts = _ts(8)
    idx = np.array([5, 0, 7, 2])
    want = [_row_key(ts, i) for i in idx]
    out = ts[idx]
    assert [_row_key(out, i) for i in range(len(out))] == want


def test_an_integer_index_still_returns_the_historical_five_tuple():
    """Callers in the Rosenbluth loop unpack exactly five values. Species must NOT
    join that tuple, or every one of them breaks."""
    ts = _ts(4)
    row = ts[2]
    assert isinstance(row, tuple) and len(row) == 5
    assert len(ts.first()) == 5


def test_delete_keeps_alignment():
    ts = _ts(8)
    want = [_row_key(ts, i) for i in (0, 1, 2, 4, 6, 7)]
    ts.delete(np.array([3, 5]))
    assert [_row_key(ts, i) for i in range(len(ts))] == want


def test_copy_is_deep_in_the_species_column_too():
    ts = _ts(5)
    dup = ts.copy()
    dup.species[0] = 99
    assert int(ts.species[0]) != 99


def test_merge_concatenates_species_in_the_same_order_as_coords():
    a = _ts(4, species=np.zeros(4, dtype=np.int64))
    b = _ts(3, species=np.full(3, 2, dtype=np.int64))
    out = a.merge(b)
    assert len(out) == 7
    assert out.species.tolist() == [0, 0, 0, 0, 2, 2, 2]
    assert out.coords[:4].tolist() == a.coords.tolist()
    assert out.coords[4:].tolist() == b.coords.tolist()


def test_the_exchange_pool_pipeline_preserves_every_row():
    """The real sequence: mask the shadow trials, merge the tombstoned members,
    permute. Species must ride with coordinates through all three.

    A member's species is exactly what must NOT be redrawn (M19), so the two sides
    of the merge are given disjoint species sets: any leakage across the merge or
    the permutation shows up as a coordinate carrying the wrong one.
    """
    rng = np.random.default_rng(0)
    trials = TrialSet(
        rng.uniform(0.0, 5.0, size=(12, 3)),
        rng.uniform(0.5, 1.5, size=12),
        species=rng.integers(0, 2, size=12).astype(np.int64),  # types 0, 1
    )
    members = TrialSet(
        rng.uniform(0.0, 5.0, size=(5, 3)),
        rng.uniform(0.5, 1.5, size=5),
        species=np.full(5, 2, dtype=np.int64),  # type 2 only
    )
    truth = {
        tuple(p): int(s)
        for p, s in list(zip(trials.coords, trials.species))
        + list(zip(members.coords, members.species))
    }

    km = rng.random(12) < 0.7
    pool = trials.mask(km).merge(members)
    pool = pool[rng.permutation(len(pool))]

    assert len(pool) == int(km.sum()) + 5
    for p, s in zip(pool.coords, pool.species):
        assert truth[tuple(p)] == int(s), f"row at {p} carries species {int(s)}"
    # the members are all still there and still type 2
    assert int(np.count_nonzero(pool.species == 2)) == 5


def test_a_column_carried_outside_the_object_would_have_been_scrambled():
    """A guard on the guard. If the permutation left row order alone, the test
    above would pass with a species column held anywhere at all, so confirm the
    permutation really does reorder."""
    rng = np.random.default_rng(1)
    ts = _ts(20)
    perm = rng.permutation(len(ts))
    assert perm.tolist() != list(range(len(ts)))
    out = ts[perm]
    assert out.species.tolist() != ts.species.tolist()
