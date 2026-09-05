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

from typing import Any

import numpy as np

from .._types import FloatArray, IntArray
from .base import Potential, register


@register("hard_sphere")
class HardSphere(Potential):
    """Pure hard core (see module docstring): admissibility is the whole
    interaction, every surviving trial weighs 1."""

    def __init__(
        self, diameter: float = 1.0, upper_cutoff: float | None = None
    ) -> None:
        """`diameter` is the hard-core contact distance sigma (reduced units:
        1.0 everywhere in this codebase); `upper_cutoff` only bounds neighbour
        queries, since the interaction itself has zero range beyond contact."""
        self.diameter = float(diameter)
        # the hard core has no range; a finite cutoff is kept only so neighbour
        # queries have an upper bound. Plain-HS runs set this per dimension.
        self.upper_cutoff = (
            float(upper_cutoff) if upper_cutoff is not None else self.diameter
        )

    @classmethod
    def build(
        cls, diameter: float = 1.0, upper_cutoff: float | None = None, **_ignored: Any
    ) -> HardSphere:
        """Registry constructor (see potentials.base.register); extra config
        keys aimed at other potentials are ignored on purpose."""
        return cls(diameter=diameter, upper_cutoff=upper_cutoff)

    def weight_bulk(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        n_particles: int | None = None,
        choice: str | None = None,
        gamma: float = 1.0,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Unit weight per surviving trial. For a hard core, admissibility is
        the whole interaction: geometry has already removed every overlapping
        trial, so each survivor contributes Boltzmann factor exp(0) = 1 and the
        Rosenbluth normalisation degenerates to a count of survivors."""
        # one row per surviving trial; survivors are non-overlapping by
        # construction, so each carries unit weight.
        n = dists.shape[0] if getattr(dists, "ndim", 1) == 2 else len(dists)
        return np.ones(n, dtype=float)

    def weight_pair(  # kwargs-absorbing variant of the base signature
        self, r: FloatArray | float, gamma: float = 1.0, **_ignored: Any
    ) -> FloatArray:
        """Overlap indicator, matching the original code's hard branch exactly:
        a 1-D input returns 0 where r <= diameter else 1 (per-pair indicator);
        a 2-D input returns the per-row count of non-overlapping neighbours.
        Kept for characterisation tests; the hot path never calls it."""
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            return np.where(r <= self.diameter, 0, 1)
        return np.sum(r >= self.diameter, axis=1)
