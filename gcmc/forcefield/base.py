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

from typing import Any

import numpy as np

from .._types import FloatArray, IntArray


class Potential:
    """Abstract interface every interaction implements; see the module
    docstring for the two-question contract (`weight_bulk`, `weight_pair`)
    and the argument conventions."""

    #: hard-core diameter (lower cutoff for the overlap test)
    diameter: float = 1.0
    #: interaction cutoff (upper bound for neighbour queries / pair weights)
    upper_cutoff: float = 1.0
    #: True only for the 3D angular aux potential, which needs geometry, not
    #: just scalar distances, to evaluate its weight.
    needs_geometry: bool = False

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
        """One weight per trial from its neighbour distances to the system
        (row-padded with inf; padding must be neutral). Geometry-aware
        potentials additionally read `coords`/`idxs`/`kdtree`."""
        raise NotImplementedError

    def weight_pair(
        self,
        r: FloatArray | float,
        gamma: float = 1.0,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        box: FloatArray | None = None,
    ) -> FloatArray:
        """Incremental weight factor per surviving trial given its distance(s)
        `r` to a just-chosen trial; multiplied into the survivors' weights at
        every deeper Rosenbluth level."""
        raise NotImplementedError

    @classmethod
    def build(cls, **kwargs: Any) -> Potential:
        """Registry constructor hook: every registered potential implements a
        `build(**config_constants)` classmethod; `make_potential` calls it."""
        raise NotImplementedError

    def pairwise(
        self, sq: FloatArray, gamma: float = 1.0, **_ignored: Any
    ) -> FloatArray:
        """Incremental weight against a just-chosen trial, given squared
        distances. Scalar-distance potentials take the square root; the angular
        3D-aux potential overrides this to use the squared distances directly
        together with the carried reference neighbour."""
        return self.weight_pair(np.sqrt(sq), gamma)


_REGISTRY: dict[str, type[Potential]] = {}


def register(name: str) -> Any:
    """Class decorator: file the potential under `name` in the registry that
    `make_potential` (and therefore the config's potential_type key) uses."""

    def deco(cls: type[Potential]) -> type[Potential]:
        _REGISTRY[name] = cls
        return cls

    return deco


def available() -> list[str]:
    """Sorted names of every registered potential (for error messages/CLI)."""
    return sorted(_REGISTRY)


def make_potential(kind: str, **kwargs: Any) -> Potential:
    """Build a potential by name. `kwargs` are the constants that potential
    needs (diameter, cutoff, beta, derived parameters); config supplies them."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown potential {kind!r}; have {available()}") from None
    return cls.build(**kwargs)
