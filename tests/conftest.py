"""Shared pytest configuration for the branch_mc suite.

Puts the repo root and ``tests/scripts`` on ``sys.path`` so ``from gcmc import
...`` resolves without an editable install and the engine tests can ``import
_harness`` after the restructure moved the harness under ``tests/scripts``.

Tiering is marker-based (``slow``, ``characterisation`` in pyproject); there is
deliberately no ``--runslow`` flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # tests/
_ROOT = _HERE.parent  # repo root (holds gcmc/, orig/, examples/)
_SCRIPTS = _HERE / "scripts"  # helpers + check_*.py live here after M7

for _p in (_ROOT, _SCRIPTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# --------------------------------------------------------------------------
# Fixtures moved up from tests/matrix/conftest.py at S2.
#
# A conftest is only visible to its own directory and below, so `matrix_dir` was
# reachable from tests/matrix but not from tests/unit -- and the sub-config
# round-trip test needs exactly the same generated QA matrices the feature
# matrix uses. Duplicating the generator would give two libraries of "the same"
# obstacles that could silently diverge; moving it up gives one.

from pathlib import Path

import numpy as np
import pytest

MATRIX_SEED = 4242
MATRIX_N = {2: 12, 3: 20}  # obstacles; sparse enough that a fluid still mixes


def _poisson_disk(
    n: int, dim: int, box: float, min_sep: float, seed: int
) -> np.ndarray:
    """Draw `n` non-overlapping points in a periodic box by dart-throwing.

    A frozen matrix whose own obstacles overlap is not a physical quench, and it
    would make every insertion into that region fail for a reason that has
    nothing to do with the sampler under test. Rejecting overlaps here keeps a
    matrix failure attributable to the code rather than the fixture.

    Parameters
    ----------
    n
        Number of obstacles to place.
    dim
        2 or 3.
    box
        Cubic box side, reduced units.
    min_sep
        Minimum centre-to-centre separation, reduced units.
    seed
        Fixed so the fixture is byte-reproducible across machines.

    Returns
    -------
    ndarray, shape (n, dim)
        In-box coordinates. Never wrapped by the caller: the loader treats the
        file as authoritative.

    Raises
    ------
    RuntimeError
        If `n` points could not be placed. Better a loud fixture failure than a
        silently sparser matrix that quietly changes what the test covers.
    """
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    for _ in range(200 * n):
        if len(pts) == n:
            break
        cand = rng.random(dim) * box
        if pts:
            delta = np.abs(np.asarray(pts) - cand)
            delta = np.minimum(delta, box - delta)  # minimum image
            if np.min(np.hypot.reduce(delta, axis=1)) < min_sep:
                continue
        pts.append(cand)
    if len(pts) != n:
        raise RuntimeError(
            f"placed {len(pts)}/{n} obstacles at min_sep={min_sep} in a "
            f"{dim}D box of {box}; loosen the fixture rather than the physics"
        )
    return np.asarray(pts, dtype=np.float64)


@pytest.fixture(scope="session")
def matrix_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of frozen QA matrices, one per (dimension, box).

    Named `matrix_<dim>d_<box>.txt`; `_axes.build_params` looks them up by the
    same convention. Plain coordinate tables rather than .npz so a failure can
    be diagnosed with `cat`.
    """
    from tests.matrix._axes import BOX, PARALLEL_BOX

    out = tmp_path_factory.mktemp("qa_matrices")
    seen: set[tuple[int, float]] = set()
    for table in (BOX, PARALLEL_BOX):
        for dim, box in table.items():
            if (dim, box) in seen:
                continue
            seen.add((dim, box))
            pos = _poisson_disk(
                MATRIX_N[dim], dim, box, min_sep=1.6, seed=MATRIX_SEED + dim
            )
            # The loader takes the first `dimension` columns, so a 2D file is
            # written with two columns and no padding z.
            np.savetxt(out / f"matrix_{dim}d_{box:g}.txt", pos, fmt="%.12f")
    return out


@pytest.fixture(scope="session")
def equilibrated_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic equilibrated library with the deployment layout and no data.

    Layout mirrored from the real one:

        <root>/<system>/box_<L>/<eta or rho>.extxyz

    The resolver's contract is about the TREE -- how it picks a system, a box
    and the nearest density -- so the frames only need to be well-formed, not
    equilibrated. A real library is validated separately at deployment time,
    where the files actually exist.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    root = tmp_path_factory.mktemp("equilibrated")
    rng = np.random.default_rng(MATRIX_SEED)
    for system, dim, box in (("hard_disk_2d", 2, 10.0), ("hard_sphere_3d", 3, 8.0)):
        for eta in (0.30, 0.45):
            n = 24
            pos = np.zeros((n, 3))
            pos[:, :dim] = _poisson_disk(n, dim, box, 1.05, rng.integers(1 << 30))
            cell = [box, box, box if dim == 3 else 1.0]
            d = root / system / f"box_{box:g}"
            d.mkdir(parents=True, exist_ok=True)
            ase_write(
                d / f"eta_{eta:.2f}.extxyz",
                Atoms("Ar" * n, positions=pos, cell=cell, pbc=True),
                format="extxyz",
            )
    return root
