"""Molecule foundation: the seam that lets a particle become a rigid group of
sites later without disturbing the single-point pipeline that exists today.

A molecule is a Species: a fixed list of site types and their body-frame offsets
from the molecule centre. Today every molecule is a single_atom, one site of
type 0 sitting exactly on the centre, so placement is the identity and the
numbers are bit-identical to treating the molecule as a bare point. A water-like
three-site species is provided to exercise the rigid-body path, but nothing in
the engine consumes multi-site species yet; this module only lays the geometry
and typing seam.

Three things generalise when real molecules arrive:
  * the system gains per-site type and molecule-id arrays (Topology.site_types,
    Topology.mol_ids), so a potential can ask which pair of types it is scoring;
  * potentials gain a type-pair weight; single-type systems ignore the indices;
  * place() turns a centre (and orientation) into world-frame site coordinates,
    returning exactly [centre] for a single atom.

The guarantee for single_atom is mechanical, not statistical: place and expand
return the input centres unchanged, so the verified modes are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._types import FloatArray


@dataclass(frozen=True)
class Species:
    """A rigid molecule template. site_offsets are body-frame coordinates with
    the molecule centre at the origin; site_types label each site for the
    potential."""

    name: str
    site_types: np.ndarray  # (n_sites,) int
    site_offsets: np.ndarray  # (n_sites, dim) float

    @property
    def n_sites(self) -> int:
        """Number of interaction sites in this species template."""
        return len(self.site_types)

    @property
    def dim(self) -> int:
        """Spatial dimension of the site coordinates (2 or 3)."""
        return self.site_offsets.shape[1]

    @property
    def is_single_atom(self) -> bool:
        """True for a one-site species at the origin, where orientation is
        meaningless and the molecule machinery collapses to the atomic path."""
        return self.n_sites == 1 and not np.any(self.site_offsets)


def single_atom(dim: int = 3, site_type: int = 0) -> Species:
    """The degenerate species used everywhere today: one site of one type, on
    the centre. Placement is the identity."""
    return Species(
        "single_atom", np.array([site_type], dtype=int), np.zeros((1, dim), dtype=float)
    )


def water(dim: int = 3, bond: float = 0.9572, angle_deg: float = 104.52) -> Species:
    """A rigid three-site water template (O at the centre, two H), here only to
    exercise the multi-site path. Types: O=0, H=1. No charges or electrostatics.
    In 2D the molecule lies in the plane."""
    half = np.deg2rad(angle_deg) / 2.0
    h1 = np.array([bond * np.sin(half), bond * np.cos(half), 0.0])
    h2 = np.array([-bond * np.sin(half), bond * np.cos(half), 0.0])
    off3 = np.vstack(([0.0, 0.0, 0.0], h1, h2))
    off = off3[:, :dim]
    return Species("water", np.array([0, 1, 1], dtype=int), off)


def place(
    species: Species, center: FloatArray, orientation: FloatArray | None = None
) -> np.ndarray:
    """World-frame site coordinates for one molecule at `center`.

    For a single atom this returns exactly [center] with no arithmetic, so the
    point pipeline is bit-identical. `orientation`, when given, is a (dim, dim)
    rotation applied to the body-frame offsets; it is ignored for a single atom.
    """
    center = np.asarray(center, float)
    if species.is_single_atom:
        return center.reshape(1, -1)
    off = species.site_offsets
    if orientation is not None:
        off = off @ np.asarray(orientation, float).T
    return center.reshape(1, -1) + off


@dataclass
class Topology:
    """A homogeneous system of one species. Holds the mapping from molecule
    centres to flat site arrays the future site-aware engine will use."""

    species: Species

    @property
    def n_sites_per_mol(self) -> int:
        """Sites per molecule (constant: one species per system for now)."""
        return self.species.n_sites

    def site_types(self, n_mol: int) -> np.ndarray:
        """Flat per-site type array for `n_mol` molecules (template tiled)."""
        return np.tile(self.species.site_types, n_mol)

    def mol_ids(self, n_mol: int) -> np.ndarray:
        """Owning molecule index of each flat site row (0..n_mol-1 repeated
        n_sites times each) -- the site -> molecule back-map."""
        return np.repeat(np.arange(n_mol), self.species.n_sites)

    def expand(
        self, centers: FloatArray, orientations: FloatArray | None = None
    ) -> np.ndarray:
        """Turn (n_mol, dim) centres into (n_mol * n_sites, dim) site coords.
        For single_atom this returns the centres unchanged (same array shape and
        values), which is the degeneracy the current engine relies on."""
        centers = np.asarray(centers, float)
        if self.species.is_single_atom:
            return centers
        n_mol = len(centers)
        out = np.empty((n_mol * self.species.n_sites, self.species.dim))
        for i in range(n_mol):
            ori = None if orientations is None else orientations[i]
            out[i * self.species.n_sites : (i + 1) * self.species.n_sites] = place(
                self.species, centers[i], ori
            )
        return out
