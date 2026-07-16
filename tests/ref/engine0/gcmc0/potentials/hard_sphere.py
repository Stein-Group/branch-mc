"""Hard sphere / hard disk.

There is no soft interaction: a pair either overlaps (r < diameter, forbidden)
or does not (weight 1). In the sampler the overlap is handled by the neighbour
mask, so every surviving trial carries weight 1 and the Rosenbluth normalisation
at each level is just the count of survivors.

`weight_pair` reproduces the original `calculate_interaction_energy` hard branch
exactly so it can be characterised against it:
  1-D distances  -> 0 where r <= diameter else 1   (overlap indicator)
  2-D distances  -> per row, the number of neighbours with r >= diameter (count)
That branch is not on the hot path for plain hard spheres (the sampler uses unit
weights and counts directly), but matching it pins the transcription.
"""

from __future__ import annotations
import numpy as np
from .base import Potential, register


@register("hard_sphere")
class HardSphere(Potential):
    def __init__(self, diameter=1.0, upper_cutoff=None):
        self.diameter = float(diameter)
        # the hard core has no range; a finite cutoff is kept only so neighbour
        # queries have an upper bound. Plain-HS runs set this per dimension.
        self.upper_cutoff = float(upper_cutoff) if upper_cutoff is not None else self.diameter

    @classmethod
    def build(cls, diameter=1.0, upper_cutoff=None, **_ignored):
        return cls(diameter=diameter, upper_cutoff=upper_cutoff)

    def weight_bulk(self, dists, *, coords=None, idxs=None, kdtree=None,
                    n_particles=None, choice=None, gamma=1.0):
        # one row per surviving trial; survivors are non-overlapping by
        # construction, so each carries unit weight.
        n = dists.shape[0] if getattr(dists, "ndim", 1) == 2 else len(dists)
        return np.ones(n, dtype=float)

    def weight_pair(self, r, gamma=1.0, **_ignored):
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            return np.where(r <= self.diameter, 0, 1)
        return np.sum(r >= self.diameter, axis=1)
