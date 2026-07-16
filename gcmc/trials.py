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

Method behaviour is kept identical, including the two that callers depend on:
  delete(i)  mutates in place and returns self
  set[i]     with an int returns the 5-tuple (coord, ew, cos, sin, count),
             without ref_neighbor_coord, exactly as before.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from ._types import BoolArray, FloatArray, IntArray


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
        "ref_neighbor_coord",
        "sinsum",
    )

    def __init__(
        self,
        coords: FloatArray,
        energy_weight: FloatArray,
        cossum: FloatArray | None = None,
        sinsum: FloatArray | None = None,
        counts: FloatArray | None = None,
        ref_neighbor_coord: FloatArray | None = None,
    ) -> None:
        """Wrap the given arrays; omitted optional fields default to neutral
        values (unit phasors/counts, zero reference coordinates) so modes that
        do not use them can ignore them safely."""
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
        )
