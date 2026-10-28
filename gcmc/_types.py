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

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
RNG: TypeAlias = np.random.Generator
