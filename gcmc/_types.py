"""Shared type aliases for the package.

Numpy's gradual typing cannot express shapes, so the aliases here encode the
next best thing: the dtype and the ROLE. `FloatArray` is any float64 ndarray
(positions, distances, weights), `IntArray` any int64 ndarray (slot ids,
indices, counts), `BoolArray` a mask. Annotations in this codebase state the
contract a reader needs (what kind of array, what it means) and do not fight
numpy over dimensionality, which mypy cannot check anyway.

`RNG` is the modern `numpy.random.Generator`; the legacy global stream is
banned outside the frozen transcripts (see NPY002 policy in pyproject.toml).
"""

from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
RNG: TypeAlias = np.random.Generator


class NeighborQueryable(Protocol):
    """The query surface `geometry.find_neighbors` needs: satisfied by both
    scipy's cKDTree (fixed-k `query` with inf/N padding) and by `CellList`
    (radius `query` with the same padding convention). Structural typing here
    keeps geometry container-agnostic, which is exactly the design intent."""

    def query(self, *args: object, **kwargs: object) -> object:
        """Neighbours of one or more points. Signature is intentionally loose:
        a `cKDTree` and a `CellList` both satisfy this and take different
        keywords, and the point of the protocol is that callers can hold either.
        """
        ...
