"""Cell planning for the parallel path: how a box is cut into cells.

One cell is one core, so the number of cells IS the number of cores and is
never stated twice. The user submits either

  * `n_cell` -- a total count (auto-factored per axis) or an explicit per-axis
    list -- and everything else is derived here (AUTO mode); or
  * the full geometry (`cell_length`, `buffer`) and the count falls out of it
    (MANUAL mode).

Geometry, one definition shared by both modes. With `n_ax[a]` cells on axis a,

    pitch[a]       = L[a] / n_ax[a]
    cell_length[a] = pitch[a] - buffer

so the gap between neighbouring cells is exactly `buffer` on every axis. That
gap is what makes concurrent cells safe: a cell's border ring of width `buffer`
holds every molecule that can interact with one of its live molecules, and
because the gap is at least as wide as the ring, no cell's border overlaps
another cell's main region.

WHY THE COUNTS ARE AUTHORITATIVE (and are handed to the decomposition rather
than recovered from a length): the inverse map is not exact in float64. Taking
`cell_length = L/n - buffer` and recovering `n` by
`floor(L / (cell_length + buffer))` loses a cell in many ordinary cases --
measured, 28 failures over sides {30, 31.322, 24.994, 12, 200, 9.41, 15.6611,
33.333} x n in 1..32 x buffer in {2.0, 2.5, 1.0, 0.7155, 3.9241}, among them

    L = 31.322, n = 4, buffer = 0.7155  ->  cell = 7.115  ->  floor gives 3

which is the MFM-300(In) supercell axis with the FF-A core floor as the buffer.
Silently getting 3 cells where 4 were asked for wastes a quarter of the node
and moves the cell/cavity ratio, which is exactly what the depth>=1 bias
depends on. See parallel_input_plan.md, decision D7.

This module is deliberately free of any dependency on `sim_setup`, `parallel`
and `decomposition`: it is a pure function of numbers, so it is tested as one
and cannot grow an engine dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CellPlan:
    """A resolved decomposition. `n_ax` is authoritative; `cell_length` is
    derived from it, never the other way round."""

    n_ax: tuple[int, ...]
    cell_length: tuple[float, ...]
    buffer: float
    n_cells: int

    @property
    def min_cell(self) -> float:
        """Shortest cell edge: the quantity both surviving guards key off."""
        return min(self.cell_length)


def _anisotropy(n_ax: Sequence[int], sides: Sequence[float]) -> float:
    """Aspect ratio of the resulting cell: longest edge over shortest. 1.0 is a
    cube. This is the objective decision D2 minimises, because both guards
    (cell must clear the cavity, cell must be positive after the buffer) bite
    on the SHORTEST edge, so the most-cubic split maximises the usable cell."""
    edges = [s / n for s, n in zip(sides, n_ax)]
    return max(edges) / min(edges)


def _factorisations(n_cell: int, dim: int) -> list[tuple[int, ...]]:
    """Every ordered dim-tuple of positive integers whose product is n_cell.

    Enumerated, not sampled: for the counts a node can have (n_cell <= a few
    hundred) the divisor tree is tiny, and enumerating makes "the best split"
    a fact rather than a heuristic.
    """
    if dim == 1:
        return [(n_cell,)]
    out: list[tuple[int, ...]] = []
    for first in range(1, n_cell + 1):
        if n_cell % first:
            continue
        for rest in _factorisations(n_cell // first, dim - 1):
            out.append((first, *rest))
    return out


def factor_cells(n_cell: int, sides: Sequence[float]) -> tuple[int, ...]:
    """Split `n_cell` cells across the axes of a box with lengths `sides` so the
    cells come out as close to cubic as possible (decision D2).

    Tie-break is deterministic: among equally-anisotropic splits, take the one
    whose count tuple is smallest lexicographically. A submission must plan the
    same way every time it is read, or a resumed run would decompose
    differently from the run it resumes.
    """
    if n_cell < 1:
        raise ValueError(f"n_cell must be >= 1, got {n_cell}")
    dim = len(sides)
    best = min(
        _factorisations(n_cell, dim),
        key=lambda n_ax: (_anisotropy(n_ax, sides), n_ax),
    )
    return best


def _plan_from_counts(
    n_ax: Sequence[int], sides: Sequence[float], buffer: float
) -> CellPlan:
    """Build the plan for given per-axis counts, with no guards applied."""
    cell = tuple(s / n - buffer for s, n in zip(sides, n_ax))
    n_ax_t = tuple(int(n) for n in n_ax)
    n_cells = 1
    for n in n_ax_t:
        n_cells *= n
    return CellPlan(
        n_ax=n_ax_t, cell_length=cell, buffer=float(buffer), n_cells=n_cells
    )


def _check(
    plan: CellPlan,
    sides: Sequence[float],
    min_cell: float,
    min_cell_desc: str = "the required minimum cell edge",
) -> None:
    """Apply the geometry guards, naming both the offending number and the
    number it had to satisfy (rule 5 of the plan). `min_cell_desc` names WHICH
    constraint set the floor, because the caller knows and this module must not
    guess: the floor may come from the cavity, from the interaction reach, or
    from whichever is larger."""
    for a, (cl, s, n) in enumerate(zip(plan.cell_length, sides, plan.n_ax)):
        if n < 1:
            raise ValueError(f"axis {a} was given {n} cells; every axis needs >= 1")
        if cl <= 0.0:
            raise ValueError(
                f"axis {a}: {n} cells across a side of {s:g} leaves a pitch of "
                f"{s / n:.6g}, which the buffer of {plan.buffer:g} consumes "
                f"entirely (cell length would be {cl:.6g}). Use fewer cells on "
                f"this axis, a shorter buffer, or a larger box."
            )
    if min_cell > 0.0 and plan.min_cell <= min_cell:
        raise ValueError(
            f"the shortest cell edge is {plan.min_cell:.6g}, which does not "
            f"exceed {min_cell:.6g} ({min_cell_desc}). Use fewer cells, a "
            f"shorter buffer, or a smaller trial_box_side."
        )


def plan_manual(
    cell_length: float | Sequence[float],
    buffer: float,
    sides: Sequence[float],
    *,
    min_cell: float = 0.0,
    min_cell_desc: str = "the required minimum cell edge",
) -> CellPlan:
    """MANUAL mode: the user stated the geometry, so the count falls out of it
    exactly as `decomposition.grid_up_nd` has always computed it,
    `n_ax[a] = floor(L[a] / (cell_length[a] + buffer))`.

    The returned `cell_length` is the SUBMITTED one, not `pitch - buffer`: in
    manual mode the user's number is authoritative and the gap is whatever is
    left over (>= buffer), which is the historical behaviour.
    """
    dim = len(sides)
    if isinstance(cell_length, (int, float)):
        cl = tuple(float(cell_length) for _ in range(dim))
    else:
        cl = tuple(float(v) for v in cell_length)
        if len(cl) != dim:
            raise ValueError(
                f"cell_length as a list needs one length per axis "
                f"(dimension={dim}, got {len(cl)})"
            )
    n_ax = []
    for a, (s, c) in enumerate(zip(sides, cl)):
        n = int((s) // (c + buffer))
        if n < 1:
            raise ValueError(
                f"axis {a}: cell_length + buffer = {c + buffer:g} does not fit "
                f"in the box side {s:g}; shrink the cell or the buffer"
            )
        n_ax.append(n)
    n_cells = 1
    for n in n_ax:
        n_cells *= n
    plan = CellPlan(
        n_ax=tuple(n_ax), cell_length=cl, buffer=float(buffer), n_cells=n_cells
    )
    _check(plan, sides, min_cell, min_cell_desc)
    return plan


def plan_auto(
    n_cell: int | Sequence[int],
    sides: Sequence[float],
    buffer: float,
    *,
    min_cell: float = 0.0,
    min_cell_desc: str = "the required minimum cell edge",
) -> CellPlan:
    """AUTO mode: `n_cell` cells, per-axis counts chosen here, cell lengths and
    the gap derived. `n_cell` is either a total (auto-factored per D2) or an
    explicit per-axis list, in which case it is honoured verbatim.

    Raises with a list of nearby usable counts when the requested count cannot
    satisfy the guards, rather than quietly planning fewer cells than the user
    has cores (decision D1).
    """
    dim = len(sides)
    if isinstance(n_cell, (list, tuple)):
        n_ax = tuple(int(v) for v in n_cell)
        if len(n_ax) != dim:
            raise ValueError(
                f"n_cell as a list needs one count per axis "
                f"(dimension={dim}, got {len(n_ax)})"
            )
        if any(n < 1 for n in n_ax):
            raise ValueError(f"every per-axis cell count must be >= 1, got {n_ax}")
        plan = _plan_from_counts(n_ax, sides, buffer)
        _check(plan, sides, min_cell, min_cell_desc)
        return plan

    if not isinstance(n_cell, int):
        raise TypeError(
            f"n_cell must be an int or a per-axis sequence of ints, got "
            f"{type(n_cell).__name__}"
        )
    total = int(n_cell)
    if total < 1:
        raise ValueError(f"run.n_cell must be >= 1, got {total}")
    plan = _plan_from_counts(factor_cells(total, sides), sides, buffer)
    try:
        _check(plan, sides, min_cell, min_cell_desc)
    except ValueError as exc:
        hint = usable_cell_counts(sides, buffer, min_cell=min_cell, near=total)
        near = ", ".join(str(c) for c in hint) if hint else "none below this count"
        raise ValueError(
            f"n_cell={total} cannot be decomposed in a box of "
            f"{[float(f'{s:g}') for s in sides]}: {exc} "
            f"Nearest usable cell counts: {near}."
        ) from None
    return plan


def usable_cell_counts(
    sides: Sequence[float],
    buffer: float,
    *,
    min_cell: float = 0.0,
    near: int,
    k: int = 5,
) -> list[int]:
    """Cell counts closest to `near` that this box can actually take, for the
    D1 error hint. Searched outward from `near` so the suggestion is the
    smallest change to the submission that works."""
    good: list[int] = []
    for delta in range(0, 4 * max(near, 1)):
        for cand in {near - delta, near + delta} if delta else {near}:
            if cand < 1:
                continue
            try:
                _check(
                    _plan_from_counts(factor_cells(cand, sides), sides, buffer),
                    sides,
                    min_cell,
                )
            except ValueError:
                continue
            if cand not in good:
                good.append(cand)
            if len(good) >= k:
                return sorted(good)
    return sorted(good)
