"""Potential interface.

A Potential answers two questions the sampler asks during an insertion or
deletion:

  weight_bulk  : given each trial's distances to the system (and, for the
                 angular 3D-aux case, the trial coords / neighbour indices /
                 kd-tree), return one weight per trial. This is what the
                 neighbour search computes for the first nesting level.

  weight_pair  : given the distances between a just-chosen trial and the
                 surviving trials, return the incremental weight factor for
                 each survivor. This is what the Rosenbluth loop multiplies in
                 at every deeper nesting level.

Overlap is a hard core at `diameter`; anything closer is rejected upstream by
the neighbour mask, so the weight functions never see sub-diameter pairs except
where the original code explicitly clamps them.

Each concrete potential reads only the arguments it needs and ignores the rest.
The signatures are deliberately flat (numpy arrays and scalars) so the hot
inner kernels can later be reimplemented in C behind the same calls.
"""

from __future__ import annotations
import numpy as np


class Potential:
    #: hard-core diameter (lower cutoff for the overlap test)
    diameter: float = 1.0
    #: interaction cutoff (upper bound for neighbour queries / pair weights)
    upper_cutoff: float = 1.0
    #: True only for the 3D angular aux potential, which needs geometry, not
    #: just scalar distances, to evaluate its weight.
    needs_geometry: bool = False

    def weight_bulk(self, dists, *, coords=None, idxs=None, kdtree=None,
                    n_particles=None, choice=None, gamma=1.0):
        raise NotImplementedError

    def weight_pair(self, r, gamma=1.0, *, trial_coords=None,
                    ref_coords=None, chosen_coord=None, box=None):
        raise NotImplementedError

    def pairwise(self, sq, gamma=1.0, **_ignored):
        """Incremental weight against a just-chosen trial, given squared
        distances. Scalar-distance potentials take the square root; the angular
        3D-aux potential overrides this to use the squared distances directly
        together with the carried reference neighbour."""
        return self.weight_pair(np.sqrt(sq), gamma)


_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def available():
    return sorted(_REGISTRY)


def make_potential(kind, **kwargs):
    """Build a potential by name. `kwargs` are the constants that potential
    needs (diameter, cutoff, beta, derived parameters); config supplies them."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown potential {kind!r}; have {available()}")
    return cls.build(**kwargs)
