"""The trial set: the currency passed between geometry and the samplers.

This is the same object the original called TrialMoleculeSet, with the property
layer removed. Fields are plain public arrays, so reads in the Rosenbluth loop
are direct and the whole thing is a struct-of-arrays that a C kernel can take
without unwrapping accessors.

Field semantics, unchanged from the original:
  coords              (N, dim)  trial coordinates
  energy_weight       (N,)      per-trial weight (g_r, LJ Boltzmann, or g3)
  cossum, sinsum      (N,)      reweighting phasor accumulators (default 1)
  counts              (N,)      reweighting neighbour counts        (default 1)
  ref_neighbor_coord  (N, dim)  3D-aux reference neighbour, carried for the
                                pairwise g3 recompute (default zeros)
  species             (N,)      per-trial type id (default 0, single species)
  pool_row            (N,)      this atom's row in the pool it was drawn from
                                (default -1, "not from a pool")
  log_bias            (N,)      log of the level-1 selection bias, UNEXPONENTIATED
                                (default 0, i.e. bias 1 / no bias)

Method behaviour is kept identical, including the two that callers depend on:
  delete(i)  mutates in place and returns self
  set[i]     with an int returns the 5-tuple (coord, ew, cos, sin, count),
             without ref_neighbor_coord, exactly as before.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from .._types import BoolArray, FloatArray, IntArray


class TrialSet:
    """Struct-of-arrays container for one move's trial cloud.

    Physics: a multi-trial GCMC move proposes many candidate positions at once
    (the "trial molecules" of configurational-bias / Rosenbluth sampling); each
    carries the per-trial statistical weight the acceptance rule needs. This
    object is that cloud: `coords[i]` and `energy_weight[i]` describe trial i,
    and the optional phasor / neighbour fields ride along for the reweighting
    and 3D-auxiliary modes that need more than a scalar weight per trial.

    Engineering: plain public numpy arrays, no property layer, so the samplers
    index rows directly and the layout is ready for a C kernel. All methods
    preserve row alignment across every field; that alignment IS the data
    structure, so any new field must be added to every method below.
    """

    __slots__ = (
        "coords",
        "cossum",
        "counts",
        "energy_weight",
        "log_bias",
        "pool_row",
        "ref_neighbor_coord",
        "sinsum",
        "species",
    )

    def __init__(
        self,
        coords: FloatArray,
        energy_weight: FloatArray,
        cossum: FloatArray | None = None,
        sinsum: FloatArray | None = None,
        counts: FloatArray | None = None,
        ref_neighbor_coord: FloatArray | None = None,
        species: IntArray | None = None,
        pool_row: IntArray | None = None,
        log_bias: FloatArray | None = None,
    ) -> None:
        """Wrap the given arrays; omitted optional fields default to neutral
        values (unit phasors/counts, zero reference coordinates, species 0) so
        modes that do not use them can ignore them safely.

        New fields go LAST in the signature, because existing callers build a
        TrialSet positionally and a field inserted in the middle would silently
        shift their arguments. Each new field's default is its neutral value, so
        those callers keep producing exactly what they produced before the field
        existed: species 0 is the one guest of a single-species run, pool_row -1
        is "not from a pool", and log_bias 0 is a bias of 1.
        """
        n = len(coords)
        self.coords = coords
        self.energy_weight = energy_weight
        self.cossum = cossum if cossum is not None else np.ones(n, dtype=float)
        self.sinsum = sinsum if sinsum is not None else np.ones(n, dtype=float)
        self.counts = counts if counts is not None else np.ones(n, dtype=float)
        if ref_neighbor_coord is not None:
            self.ref_neighbor_coord = ref_neighbor_coord
        else:
            dim = 3 if (n > 0 and coords.shape[1] == 3) else 2
            self.ref_neighbor_coord = np.zeros((n, dim), dtype=float)
        # Per-trial type id, row-aligned with `coords` like every other field.
        # The exchange pool is a merge of the surviving trial cloud with the
        # tombstoned cavity members and is then PERMUTED before the sweep reads
        # it, so a species column held anywhere but here would be scrambled by
        # that permutation.
        self.species = (
            np.zeros(n, dtype=np.int64)
            if species is None
            else np.asarray(species, dtype=np.int64)
        )
        if len(self.species) != n:
            raise ValueError(
                f"species column has {len(self.species)} entries for {n} trials"
            )
        # PROVENANCE. Which row of the exchange pool this atom is, numbered
        # AFTER the pool was permuted. -1 means "not from a pool".
        #
        # This column exists because M21 decides provenance BY POOL ROW and never
        # by comparing coordinates -- a reselected member and a trial atom that
        # happens to land on it are indistinguishable by position, and the row
        # index is exact. The engine permutes the pool
        # (`pool = pool[rng.permutation(len(pool))]`) and the Rosenbluth selection
        # then deletes and masks it between levels, so a row number held anywhere
        # outside the struct would be scrambled by the first of those and stale
        # after the second. Held here it rides along, because every method below
        # preserves row alignment across every field.
        self.pool_row = (
            np.full(n, -1, dtype=np.int64)
            if pool_row is None
            else np.asarray(pool_row, dtype=np.int64)
        )
        if len(self.pool_row) != n:
            raise ValueError(
                f"pool_row column has {len(self.pool_row)} entries for {n} trials"
            )
        # THE LEVEL-1 BIAS, NOT YET RAISED TO ANYTHING. Zero everywhere means
        # bias 1, which is the no-bias case and what every mode but adaptive
        # gamma leaves it as.
        #
        # This is a column rather than a local array because of WHERE it is
        # produced and WHO reads it. `_weighted_trials` computes it at pool-build
        # time, before the forward and reverse passes diverge; the engine then
        # merges the surviving trials with the tombstoned members and PERMUTES
        # the result (`pool = pool[perm]`), and the Rosenbluth selection deletes
        # and masks rows between levels. An array held outside the struct would
        # be scrambled by the permutation and stale after the first mask, which
        # is the same argument the `pool_row` comment above makes for
        # provenance. Held here it rides along, because every method below
        # preserves row alignment across every field.
        #
        # Deeper levels do NOT need this: their increment is produced and
        # consumed inside one iteration of the selection loop, so it stays a
        # local. Level 1 is the only value that has to cross the permutation.
        #
        # Named for the bias in general and not for g3: the slot is "the part of
        # the weight the exponent applies to", and the tabulated three-body g3
        # is merely the only bias that currently fills it.
        self.log_bias = np.zeros(n, dtype=float) if log_bias is None else log_bias
        if len(self.log_bias) != n:
            raise ValueError(
                f"log_bias column has {len(self.log_bias)} entries for {n} trials"
            )

    def __len__(self) -> int:
        """Number of trials in the set."""
        return len(self.coords)

    @overload
    def __getitem__(
        self, index: int
    ) -> tuple[FloatArray, np.float64, np.float64, np.float64, np.float64]: ...
    @overload
    def __getitem__(self, index: slice | list[int] | IntArray) -> TrialSet: ...

    def __getitem__(
        self, index: int | slice | list[int] | IntArray
    ) -> TrialSet | tuple[FloatArray, np.float64, np.float64, np.float64, np.float64]:
        """Array-like indexing with two deliberate behaviours: a fancy index
        (array / list / slice) returns a new TrialSet over those rows, while an
        integer returns the historical 5-tuple (coord, energy_weight, cossum,
        sinsum, count) WITHOUT ref_neighbor_coord -- callers in the Rosenbluth
        loop unpack exactly that tuple."""
        if isinstance(index, (np.ndarray, list, slice)):
            return TrialSet(
                self.coords[index],
                self.energy_weight[index],
                self.cossum[index],
                self.sinsum[index],
                self.counts[index],
                self.ref_neighbor_coord[index],
                self.species[index],
                self.pool_row[index],
                self.log_bias[index],
            )
        return (
            self.coords[index],
            self.energy_weight[index],
            self.cossum[index],
            self.sinsum[index],
            self.counts[index],
        )

    def first(
        self,
    ) -> tuple[FloatArray, np.float64, np.float64, np.float64, np.float64]:
        """The 5-tuple of trial 0 (see __getitem__); raises on an empty set."""
        if len(self) == 0:
            raise IndexError("TrialSet is empty.")
        return (
            self.coords[0],
            self.energy_weight[0],
            self.cossum[0],
            self.sinsum[0],
            self.counts[0],
        )

    def mask(self, boolean_mask: BoolArray) -> TrialSet:
        """New TrialSet keeping the rows where `boolean_mask` is True. Used to
        drop the shadow trials (keep_mask) before assembling the reverse pool."""
        return TrialSet(
            self.coords[boolean_mask],
            self.energy_weight[boolean_mask],
            self.cossum[boolean_mask],
            self.sinsum[boolean_mask],
            self.counts[boolean_mask],
            self.ref_neighbor_coord[boolean_mask],
            self.species[boolean_mask],
            self.pool_row[boolean_mask],
            self.log_bias[boolean_mask],
        )

    def delete(self, index: int | IntArray) -> TrialSet:
        """Remove row(s) IN PLACE and return self; the Rosenbluth selection
        deletes the chosen trial between levels and relies on the mutation."""
        self.coords = np.delete(self.coords, index, axis=0)
        self.energy_weight = np.delete(self.energy_weight, index, axis=0)
        self.cossum = np.delete(self.cossum, index, axis=0)
        self.sinsum = np.delete(self.sinsum, index, axis=0)
        self.counts = np.delete(self.counts, index, axis=0)
        self.ref_neighbor_coord = np.delete(self.ref_neighbor_coord, index, axis=0)
        self.species = np.delete(self.species, index, axis=0)
        self.pool_row = np.delete(self.pool_row, index, axis=0)
        self.log_bias = np.delete(self.log_bias, index, axis=0)
        return self

    def copy(self) -> TrialSet:
        """Deep copy of every field; the forward pool is copied so selection
        can mutate it without touching the trial set the reverse pass reads."""
        return TrialSet(
            self.coords.copy(),
            self.energy_weight.copy(),
            self.cossum.copy(),
            self.sinsum.copy(),
            self.counts.copy(),
            self.ref_neighbor_coord.copy(),
            self.species.copy(),
            self.pool_row.copy(),
            self.log_bias.copy(),
        )

    def merge(self, other: TrialSet) -> TrialSet:
        """Concatenate two sets row-wise (self first). The exchange move merges
        the surviving trials with the removed members to form the shared pool
        that both the forward and the reverse weights are computed over."""
        return TrialSet(
            np.vstack([self.coords, other.coords]),
            np.hstack([self.energy_weight, other.energy_weight]),
            np.hstack([self.cossum, other.cossum]),
            np.hstack([self.sinsum, other.sinsum]),
            np.hstack([self.counts, other.counts]),
            np.vstack([self.ref_neighbor_coord, other.ref_neighbor_coord]),
            np.hstack([self.species, other.species]),
            np.hstack([self.pool_row, other.pool_row]),
            np.hstack([self.log_bias, other.log_bias]),
        )
