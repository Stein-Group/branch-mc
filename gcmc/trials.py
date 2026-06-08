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
import numpy as np


class TrialSet:
    __slots__ = ("coords", "energy_weight", "cossum", "sinsum", "counts",
                 "ref_neighbor_coord")

    def __init__(self, coords, energy_weight, cossum=None, sinsum=None,
                 counts=None, ref_neighbor_coord=None):
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

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        if isinstance(index, (np.ndarray, list, slice)):
            return TrialSet(self.coords[index], self.energy_weight[index],
                            self.cossum[index], self.sinsum[index],
                            self.counts[index], self.ref_neighbor_coord[index])
        return (self.coords[index], self.energy_weight[index],
                self.cossum[index], self.sinsum[index], self.counts[index])

    def first(self):
        if len(self) == 0:
            raise IndexError("TrialSet is empty.")
        return (self.coords[0], self.energy_weight[0], self.cossum[0],
                self.sinsum[0], self.counts[0])

    def mask(self, boolean_mask):
        return TrialSet(self.coords[boolean_mask], self.energy_weight[boolean_mask],
                        self.cossum[boolean_mask], self.sinsum[boolean_mask],
                        self.counts[boolean_mask], self.ref_neighbor_coord[boolean_mask])

    def delete(self, index):
        self.coords = np.delete(self.coords, index, axis=0)
        self.energy_weight = np.delete(self.energy_weight, index, axis=0)
        self.cossum = np.delete(self.cossum, index, axis=0)
        self.sinsum = np.delete(self.sinsum, index, axis=0)
        self.counts = np.delete(self.counts, index, axis=0)
        self.ref_neighbor_coord = np.delete(self.ref_neighbor_coord, index, axis=0)
        return self

    def copy(self):
        return TrialSet(self.coords.copy(), self.energy_weight.copy(),
                        self.cossum.copy(), self.sinsum.copy(),
                        self.counts.copy(), self.ref_neighbor_coord.copy())

    def merge(self, other):
        return TrialSet(np.vstack([self.coords, other.coords]),
                        np.hstack([self.energy_weight, other.energy_weight]),
                        np.hstack([self.cossum, other.cossum]),
                        np.hstack([self.sinsum, other.sinsum]),
                        np.hstack([self.counts, other.counts]),
                        np.vstack([self.ref_neighbor_coord, other.ref_neighbor_coord]))
