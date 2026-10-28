"""Domain decomposition for the parallel path.

Ported verbatim from the original parallelization.grid_up_the_system and the
PARALLEL-mode geometry helpers, so the 2D decomposition is byte-identical to
the verified original.

Two structures answering different questions, both staying: `grid_up_nd` says
which cell OWNS a molecule (domain decomposition, 2D and 3D, and the only
symbol the parallel engine calls); `neighbors.CellList` answers neighbour
search inside a cell. The verbatim 2D helpers below `grid_up_the_system` are
kept as the reference the characterisation tier pins against orig/; the engine
does not call them (see cell_unify_plan.md U2).

3D parallel is supported for metropolis, rosenbluth and branch. sim_setup
rejects reweight, QA, channel and confined up front.

`cell_length_global` is the per-cell side the grid uses; the parallel engine
sets it via configure() before applying the cell PBC, matching the original
module global the helpers assert against.
"""

from __future__ import annotations

import itertools as _itertools
from typing import NamedTuple

import numpy as np

from .._types import FloatArray, IntArray

cell_length_global = 30


# --------------------------------------------------------------------------
# dimension-generic gridding (2D and 3D), seedable. The 2D helpers above are
# the characterisation reference; the parallel engine does not call them.
# --------------------------------------------------------------------------


class Cell(NamedTuple):
    """One cell of a decomposition.

    A NamedTuple rather than a bare tuple so the species columns can be added
    without a conditional arity, and without costing anything: it IS a tuple, so
    `cell[0]` is the same operation it always was and the hot reassembly path is
    unaffected. The species fields default to None, which is the single-species
    case, so a caller that never asks for species sees exactly the old three
    values in the old order.

    `main_species` and `border_species` are positionally aligned with `main` and
    `border`. That alignment is the whole point: species travels WITH the
    coordinates, never by index into the original array, because a cell's
    molecules are a mask-selected subset and the parallel workers reorder them.
    """

    main: FloatArray
    border: FloatArray
    origin: FloatArray
    main_species: IntArray | None = None
    border_species: IntArray | None = None


class Decomposition(NamedTuple):
    """The result of `grid_up_nd`: the cells, plus the molecules no cell owns.

    Three fields rather than the old two, so the leftover species have somewhere
    to live. Callers unpack all three (`cells, rest, rest_species = ...`) or read
    the attributes; there are only two call sites, and one unconditional shape is
    easier to reason about than a return whose length depends on an argument.
    """

    cells: list[Cell]
    rest: FloatArray
    rest_species: IntArray | None = None


def grid_up_nd(positions, side, cell_length, buffer, rng, dim, n_ax=None, species=None):
    """Random-origin box decomposition, per-axis (anisotropic) capable.

    Each axis a is tiled with n_ax[a] cells at pitch[a] = side[a] / n_ax[a], so
    the gap between neighbouring cells is always >= buffer on every axis.
    `side` and `cell_length` may each be a scalar (replicated over all axes --
    the old cubic behaviour, bit-identical) or a length-`dim` array, so a box
    can be tiled n_x x n_y x n_z with independent counts (e.g. 4 x 4 x 5 = 80).
    Each cell owns the molecules inside its box (main) and sees a frozen
    read-only ring of width `buffer` around it (border); because gap >= buffer
    per axis, no cell's border overlaps another cell's main, so cells run
    concurrently without touching each other's live molecules.

    `n_ax` (optional): the per-axis cell counts, supplied by the caller. When
    omitted they are derived here as floor(side / (cell_length + buffer)),
    which is what this function has always done and what
    tests/scripts/check_decomposition.py pins against orig/. Pass them
    explicitly when the counts are the authoritative input: the derivation is
    NOT exact in float64 and silently loses a cell in ordinary cases (e.g.
    side 31.322, 4 cells, buffer 0.7155 -> cell 7.115 -> floor gives 3). See
    parallel_input_plan.md, decision D7.

    `species` (optional): a per-molecule type column, sliced by the SAME boolean
    masks as the coordinates so every molecule keeps its own type through the
    split. Omitted means single species, and every species field of the result
    is None.

    Returns a `Decomposition`: `cells` is a list of `Cell`, `rest` is the
    coordinates owned by no cell (gap regions, a superset of all borders), and
    `rest_species` their types.
    """
    positions = np.asarray(positions, float).reshape(-1, dim)
    if species is None:
        spc = None
    else:
        spc = np.asarray(species, dtype=np.int64).ravel()
        if len(spc) != len(positions):
            raise ValueError(
                f"species column has {len(spc)} entries for {len(positions)} molecules"
            )
    side = np.broadcast_to(np.asarray(side, float), (dim,))
    cl = np.broadcast_to(np.asarray(cell_length, float), (dim,))
    if n_ax is None:
        n_ax = np.floor(side / (cl + buffer)).astype(int)
    else:
        n_ax = np.asarray(n_ax, dtype=int).reshape(dim)
    if np.any(n_ax < 1):
        raise ValueError(
            f"cell_length + buffer = {(cl + buffer).tolist()} does not fit in "
            f"box side {side.tolist()} on every axis; shrink the cell or buffer"
        )
    pitch = side / n_ax
    origin0 = rng.uniform(0.0, side, size=dim)

    u = (positions - origin0) % side
    in_main = np.all(u - np.floor(u / pitch) * pitch < cl, axis=1)

    cells = []
    for idx in _itertools.product(*(range(int(n)) for n in n_ax)):
        lo = np.asarray(idx, float) * pitch
        d = (u - lo) % side
        main_mask = np.all(d < cl, axis=1)
        ext_mask = np.all((d < cl + buffer) | (d > side - buffer), axis=1)
        border_mask = ext_mask & ~main_mask
        origin = (origin0 + lo) % side
        cells.append(
            Cell(
                positions[main_mask].copy(),
                positions[border_mask].copy(),
                origin,
                None if spc is None else spc[main_mask].copy(),
                None if spc is None else spc[border_mask].copy(),
            )
        )
    return Decomposition(
        cells,
        positions[~in_main].copy(),
        None if spc is None else spc[~in_main].copy(),
    )
