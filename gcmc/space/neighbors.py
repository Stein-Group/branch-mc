"""The one neighbour structure in the hot path: a persistent periodic cell list.

Everything the engine asks of a neighbour index -- "which particles are within rc
of these trial points", "which slots could lie inside this body" -- is answered
here, from a structure that is built once per window and updated in O(1) per
accepted move. Nothing else in the hot path may build a kd-tree, a grid, or a
distance loop; if a capability is missing, it belongs in this module.

Why a cell list at all. The engine used to rebuild `scipy.spatial.cKDTree` from
scratch on every move, twice per exchange step: 4.3 ms at N = 15,000, about 90%
of the wall time. A cell list answers the same radius query in tens of
microseconds and survives an accepted move for the cost of moving one slot
between two buckets.

Design, and the two measurements that forced it:

  * A per-query-point Python loop over buckets is 35x SLOWER than scipy. So a
    query is ONE batched gather over the union of the query points' cell
    neighbourhoods, then ONE minimum-image distance matrix. No loops over
    points, none over cells.
  * A gather that loops over buckets in Python costs 2 ms at N = 15,000. So the
    CSR segment gather is the repeat-trick (`np.repeat(start[cells], lens) +
    arange - offsets`), fully vectorised.

Storage and lifetime.

  `coords` is a slot table with free-slot allocation, not a compact array: a
  particle keeps its slot id for its whole life, so the engine can hold slot ids
  across a move. `active` marks live slots; a removal is a tombstone, reclaimed
  at the next CSR rebuild. The CSR index (`_sorted` + `_start`) covers the slots
  that were live at the last rebuild; slots inserted since then live in a small
  `pending` list that every gather concatenates, and a rebuild is triggered once
  `pending` exceeds a threshold. `_in_csr[slot]` says whether a slot's CSR entry
  still describes its current cell, so a slot is found through exactly one of the
  two paths and a gather can never return it twice.

  Frozen quenched-annealed obstacles occupy the lowest slot ids, are marked in
  `is_matrix`, and are never removed. `matrix_at(idxs)` answers the "is this
  neighbour an obstacle" question the LJ weights ask, which is what the fluid /
  matrix split of a compact array used to answer by index comparison.

Query contract. `query` reproduces what `cKDTree.query(..., k, distance_upper_
bound=rc)` hands `geometry.find_neighbors` today: rows sorted ascending, padded
with `inf` distances and pad index `= data.shape[0]`, and (like scipy) a
neighbour at exactly `rc` is EXCLUDED. Unlike a fixed-k query it returns every
neighbour inside the cutoff, so `query_saturating`'s doubling loop is unnecessary
on any converted path: a radius query is complete by construction.

Caller contract for slot lifetime, which the engine relies on and a rebuild would
otherwise break: a slot that has been `remove`d and might still be `reactivate`d
must not be held across an `insert` OR a `move` of a DIFFERENT slot, because
both may rebuild once the pending list crosses its threshold, and a rebuild
hands every tombstoned slot back to the free list. (`move` checks the threshold
only after it has re-activated its own slot, so moving a tombstoned slot is
always safe for that slot itself.) Every engine path obeys this -- a proposal
removes, queries, and then either reactivates (reject) or inserts / moves
(accept, at which point the removed slots are dead or revived, never in limbo).

No scipy here, by rule and by test.
"""

from __future__ import annotations

import math

import numpy as np

from .._types import BoolArray, FloatArray, IntArray

# Number of not-yet-indexed slots tolerated before the CSR index is rebuilt.
# The trade is asymmetric. A rebuild is O(N) (2.2 ms at N = 15,000) but rare;
# a pending slot is cheap to hold but taxes EVERY query, because every gather
# pairs the surviving pending slots with every query point. At 512 the pending
# block doubled the pair count of a 40-trial query and cost more than the O(N)
# rebuild it was deferring; 32 puts the amortised rebuild at roughly 20 us/step
# at N = 15,000 and keeps the pending block negligible. Both `insert` and
# `move` check the threshold, so the pending list is bounded on every path,
# including exchange-free (pure-translation) runs where `insert` never fires.
PENDING_THRESHOLD = 32


class CellList:
    """Periodic cell list over a cubic or orthorhombic box, 2D or 3D.

    Parameters
    ----------
    box : float or (dim,)   periodic box lengths.
    dim : int               2 or 3.
    cutoff : float          the largest radius any query will use. Cell width is
                            keyed to it: nc = max(1, floor(box / cutoff)) per
                            axis, so a query never reaches past its own cell's
                            immediate neighbours.
    capacity : int          initial slot table size; grows x1.5 (a growth forces
                            a CSR rebuild).
    """

    __slots__ = (
        "_capacity",
        "_cell",
        "_cellsel",
        "_coordsT",
        "_count",
        "_dirty",
        "_free",
        "_in_csr",
        "_indexed",
        "_inv_cw",
        "_lut_cache",
        "_n_slots",
        "_ncell",
        "_offs_cache",
        "_pending",
        "_pending_arr",
        "_pending_flag",
        "_pending_threshold",
        "_sorted",
        "_start",
        "_strides",
        "_visible",
        "active",
        "box",
        "cutoff",
        "cw",
        "dim",
        "is_matrix",
        "n_active",
        "n_matrix",
        "n_rebuilds",
        "nc",
        "species",
    )

    # ------------------------------------------------------------------ build
    def __init__(
        self,
        box: float | FloatArray,
        dim: int,
        cutoff: float,
        capacity: int = 1024,
        pending_threshold: int = PENDING_THRESHOLD,
    ) -> None:
        """Empty cell list. Grid geometry (cell counts nc, widths cw, row-major
        strides) is fixed here from box and cutoff; slot storage starts at
        `capacity` and grows on demand. Insert particles with `bulk_insert` /
        `insert` or use `from_positions`."""
        self.dim = int(dim)
        self.box = np.broadcast_to(np.asarray(box, float), (self.dim,)).copy()
        self.cutoff = float(cutoff)
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.nc = np.maximum(1, np.floor(self.box / self.cutoff)).astype(np.int64)
        self.cw = self.box / self.nc
        self._inv_cw = 1.0 / self.cw
        self._offs_cache: dict[int | tuple[str, int], IntArray] = {}
        self._lut_cache: dict[tuple[int, ...], tuple[IntArray, IntArray, IntArray]] = {}
        self._pending_arr: IntArray | None = None
        st = np.ones(self.dim, dtype=np.int64)
        for a in range(self.dim - 2, -1, -1):
            st[a] = st[a + 1] * self.nc[a + 1]
        self._strides = st
        self._ncell = int(np.prod(self.nc))

        cap = max(int(capacity), 16)
        self._capacity = cap
        # coordinates are held TRANSPOSED, (dim, capacity). Every hot-path gather
        # is per-axis (`_coordsT[a][slots]`), and a fancy gather from a
        # contiguous row is markedly faster than the same gather from a stride-dim
        # column of an (n, dim) table. `.data` and `.coords` expose the usual
        # (n, dim) view for the weighers.
        self._coordsT = np.zeros((self.dim, cap), dtype=np.float64)
        self.active = np.zeros(cap, dtype=bool)
        self.is_matrix = np.zeros(cap, dtype=bool)
        # Slot-aligned species column. int64 so it indexes the (T, T)
        # core-floor and pair tables without a cast on the hot path. All-zero is
        # the single-species case -- type 0 is the one guest -- so a single-species
        # run reads the same values it would have read before this column existed.
        #
        # Deliberately NOT folded into `is_matrix`: that flag means "enters the
        # overlap test but never the energy sum, and is never inserted, removed,
        # moved or counted", which is orthogonal to which species something is.
        # A framework atom has a species too.
        self.species = np.zeros(cap, dtype=np.int64)
        self._cell = np.zeros(cap, dtype=np.int64)
        self._in_csr = np.zeros(cap, dtype=bool)
        # `_indexed`: the slot has a row in `_sorted`. `_visible`: that row still
        # describes it (active and its CSR cell is current). `_dirty` counts the
        # indexed-but-not-visible slots, so a gather can skip the filter entirely
        # while nothing is stale, which is the common case right after a rebuild.
        self._indexed = np.zeros(cap, dtype=bool)
        self._visible = np.zeros(cap, dtype=bool)
        self._dirty = 0
        self._pending_flag = np.zeros(cap, dtype=bool)
        self._pending: list[int] = []
        self._free: list[int] = list(range(cap - 1, -1, -1))  # pop() yields ascending
        self._n_slots = 0
        self.n_active = 0
        self.n_matrix = 0
        self.n_rebuilds = 0
        self._pending_threshold = int(pending_threshold)
        self._sorted = np.empty(0, dtype=np.int64)
        self._start = np.zeros(self._ncell + 1, dtype=np.int64)
        self._count = np.zeros(self._ncell, dtype=np.int64)
        self._cellsel = np.zeros(self._ncell, dtype=bool)

    @classmethod
    def from_positions(
        cls,
        positions: FloatArray,
        box: float | FloatArray,
        dim: int,
        cutoff: float,
        matrix: FloatArray | None = None,
        pending_threshold: int = PENDING_THRESHOLD,
        species: IntArray | None = None,
        matrix_species: IntArray | int | None = None,
    ) -> CellList:
        """Build from a fluid configuration plus optional frozen obstacles.

        Obstacles take the lowest slots so `is_matrix` is a prefix, then the
        fluid, so slot i of the fluid is `n_matrix + i` at build time. The engine
        never relies on that beyond the build; it tracks slots explicitly.

        `species` is the fluid's type column and `matrix_species` the framework's;
        both default to type 0, which reproduces the single-species build exactly.
        They are separate arguments because the two blocks are inserted
        separately and a single concatenated column would have to encode the
        obstacles-first ordering, making a caller's correctness depend on knowing
        that ordering.
        """
        positions = np.asarray(positions, float).reshape(-1, dim)
        n_mat = 0 if matrix is None else len(matrix)
        n = len(positions) + n_mat
        cap = max(2 * n, n + 1024, 16)
        cl = cls(box, dim, cutoff, capacity=cap, pending_threshold=pending_threshold)
        if n_mat:
            cl.bulk_insert(
                np.asarray(matrix, float).reshape(-1, dim),
                matrix=True,
                species=matrix_species,
            )
        cl.bulk_insert(positions, species=species)
        cl.rebuild()
        return cl

    # ------------------------------------------------------------- cell index
    def _cell_coords(self, points: FloatArray) -> IntArray:
        """Integer cell coordinates (m, dim) of already-wrapped points.

        The clamp handles one real floating-point edge: np.mod(x, L) can return
        exactly L for a tiny negative x (L - eps rounds to L), which would index
        cell nc; minimum() folds it into the last cell."""
        # points are already wrapped into [0, box), so truncation is the floor and
        # the only out-of-range value a rounding edge can produce is nc itself
        p = np.asarray(points, float).reshape(-1, self.dim)
        c = (p * self._inv_cw).astype(np.int64)
        return np.minimum(c, self.nc - 1)

    def _cell_ids(self, points: FloatArray) -> IntArray:
        """Flat row-major cell id of each point (cell coords dotted with the
        precomputed strides)."""
        return self._cell_coords(points) @ self._strides

    # ---------------------------------------------------------- slot lifetime
    def _grow(self) -> None:
        """Enlarge every slot-indexed array by ~1.5x and rebuild. The rebuild
        is not optional: growth changes `capacity`, which is the pad index the
        queries hand out, and rebuilding also re-derives the free list over
        the new tail in one place instead of two."""
        new = max(self._capacity + 16, int(self._capacity * 1.5))
        ct = np.zeros((self.dim, new), dtype=np.float64)
        ct[:, : self._capacity] = self._coordsT
        self._coordsT = ct
        for name in (
            "active",
            "is_matrix",
            "species",
            "_cell",
            "_in_csr",
            "_indexed",
            "_visible",
            "_pending_flag",
        ):
            old = getattr(self, name)
            arr = np.zeros(new, dtype=old.dtype)
            arr[: self._capacity] = old
            setattr(self, name, arr)
        self._free.extend(range(new - 1, self._capacity - 1, -1))
        self._capacity = new
        self.rebuild()

    def _alloc(self) -> int:
        """Pop one free slot id (growing if exhausted) and extend the
        in-use high-water mark `_n_slots` that rebuild scans."""
        if not self._free:
            self._grow()
        s = self._free.pop()
        if s >= self._n_slots:
            self._n_slots = s + 1
        return s

    def bulk_insert(
        self,
        points: FloatArray,
        matrix: bool = False,
        species: IntArray | int | None = None,
    ) -> IntArray:
        """Insert many points at once (build path). Returns the slot ids.

        Vectorised on purpose: the per-window rebuild inserts every particle, and
        a Python loop over `insert` costs 0.3 s at N = 15,000, which would swamp
        everything the refactor saves. Slots are taken as one contiguous block off
        the free list, so `from_positions` hands out slots in input order.

        `species` is a per-point type column, one scalar for all of them, or None
        for the single-species default of type 0. A column of the wrong length is
        an error rather than a broadcast: silently recycling species across
        particles is exactly the bug that would surface as a wrong isotherm three
        phases later.
        """
        points = np.mod(np.asarray(points, float).reshape(-1, self.dim), self.box)
        n = len(points)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        while len(self._free) < n:
            self._grow()
        take = len(self._free) - n
        slots = np.array(self._free[take:][::-1], dtype=np.int64)
        del self._free[take:]
        self._n_slots = max(self._n_slots, int(slots.max()) + 1)

        self._coordsT[:, slots] = points.T
        self._cell[slots] = self._cell_ids(points)
        self.active[slots] = True
        self.is_matrix[slots] = bool(matrix)
        if species is None:
            self.species[slots] = 0
        elif np.isscalar(species):
            self.species[slots] = int(species)  # type: ignore[arg-type]
        else:
            col = np.asarray(species, dtype=np.int64).ravel()
            if len(col) != n:
                raise ValueError(
                    f"species column has {len(col)} entries for {n} points"
                )
            self.species[slots] = col
        if matrix:
            self.n_matrix += n
        self._in_csr[slots] = False
        self._indexed[slots] = False
        self._visible[slots] = False
        self._pending_flag[slots] = True
        self._pending.extend(slots.tolist())
        self._pending_arr = None
        self.n_active += n
        return slots

    def insert(
        self,
        point: FloatArray,
        matrix: bool = False,
        species: int = 0,
        _defer: bool = False,
    ) -> int:
        """Insert one point; returns its slot id. May trigger a CSR rebuild, which
        reclaims tombstoned slots -- see the module note on slot lifetime.

        The species is written unconditionally, not left at whatever the slot's
        previous tenant was: a reclaimed slot carries a stale species otherwise,
        and that is the one way this column can go wrong without any test of the
        cell list itself noticing.
        """
        s = self._alloc()
        p = np.mod(np.asarray(point, float), self.box)
        self._coordsT[:, s] = p
        self._cell[s] = int(self._cell_ids(p)[0])
        self.active[s] = True
        self.is_matrix[s] = bool(matrix)
        self.species[s] = int(species)
        if matrix:
            self.n_matrix += 1
        self._in_csr[s] = False
        self._indexed[s] = False
        self._visible[s] = False
        if not self._pending_flag[s]:
            self._pending_flag[s] = True
            self._pending.append(s)
            self._pending_arr = None
        self.n_active += 1
        if not _defer and len(self._pending) > self._pending_threshold:
            self.rebuild()
        return s

    def remove(self, slot: int) -> None:
        """Tombstone a slot: invisible to queries, slot id still owned by the
        caller until the next rebuild."""
        if self.is_matrix[slot]:
            raise ValueError("frozen matrix slots are never removed")
        if self.active[slot]:
            self.active[slot] = False
            self.n_active -= 1
            if self._visible[slot]:
                self._visible[slot] = False
                if self._indexed[slot]:
                    self._dirty += 1

    def reactivate(self, slot: int) -> None:
        """Undo a `remove`: the rejection path of a proposal. Restores the
        slot's query visibility through whichever path (CSR row or pending
        list) still describes it; a no-op on an already-active slot. Only
        valid while the tombstone still owns the slot -- see the module
        lifetime contract."""
        if not self.active[slot]:
            self.active[slot] = True
            self.n_active += 1
            if self._in_csr[slot]:
                self._visible[slot] = True
                if self._indexed[slot]:
                    self._dirty -= 1

    def move(self, slot: int, new_point: FloatArray) -> int:
        """Set a slot's position (activating it if it was tombstoned). Stays in
        its CSR bucket when the cell does not change, which is the common case
        for a translation, so most moves add no rebuild pressure at all.

        A cell-crossing move parks the slot on the pending list, exactly like an
        insert, and the same threshold applies: without it, an exchange-free run
        (pure translation MC, where `insert` never fires) would accumulate
        pending slots without bound and every query would degrade back toward
        the O(N) scan the cell list exists to avoid. The threshold check runs
        LAST, after this slot is active again, so a rebuild here can never
        reclaim the very slot being moved; it can reclaim other tombstones,
        which is why the module-level lifetime contract covers `move` too."""
        p = np.mod(np.asarray(new_point, float), self.box)
        cid = int(self._cell_ids(p)[0])
        self._coordsT[:, slot] = p
        if self._in_csr[slot] and cid != self._cell[slot]:
            self._in_csr[slot] = False
            if self._visible[slot]:
                self._visible[slot] = False
                if self._indexed[slot]:
                    self._dirty += 1
            if not self._pending_flag[slot]:
                self._pending_flag[slot] = True
                self._pending.append(slot)
                self._pending_arr = None
        self._cell[slot] = cid
        if not self.active[slot]:
            self.active[slot] = True
            self.n_active += 1
            if self._in_csr[slot]:
                self._visible[slot] = True
                if self._indexed[slot]:
                    self._dirty -= 1
        # Rebuild check AFTER the slot is active: a rebuild frees tombstones,
        # and this slot must not be one of them when its coordinates were just
        # written. See the docstring and the module lifetime contract.
        if len(self._pending) > self._pending_threshold:
            self.rebuild()
        return slot

    def rebuild(self) -> None:
        """Reindex every active slot into the CSR buckets; free the tombstones."""
        live = np.nonzero(self.active[: self._n_slots])[0]
        cells = self._cell[live]
        order = np.argsort(cells, kind="stable")
        self._sorted = live[order].astype(np.int64)
        counts = np.bincount(cells, minlength=self._ncell).astype(np.int64)
        self._count = counts
        self._start = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._in_csr[:] = False
        self._in_csr[live] = True
        self._indexed[:] = False
        self._indexed[live] = True
        self._visible[:] = False
        self._visible[live] = True
        self._dirty = 0
        self._pending_flag[:] = False
        self._pending = []
        self._pending_arr = None
        dead = np.nonzero(~self.active[: self._n_slots])[0]
        self._free = list(range(self._capacity - 1, self._n_slots - 1, -1))
        self._free.extend(dead[::-1].tolist())
        self.n_rebuilds += 1

    # ---------------------------------------------------------------- gathers
    def _gather_cells(self, cells: IntArray) -> IntArray:
        """Active slots in `cells` (a UNIQUE int64 array of cell ids).

        Not-yet-indexed slots are appended wholesale rather than filtered by
        cell: the only callers are the membership gathers, which run an exact
        body test on whatever comes back, so a superset costs a few extra body
        tests and saves a `np.unique` over the whole cell block. There are never
        many of them (see PENDING_THRESHOLD).
        """
        out = []
        if self._sorted.size:
            s = self._start[cells]
            lens = self._count[cells]  # one gather, not start[c+1]-start[c]
            tot = int(lens.sum())
            if tot:
                base = s - (np.cumsum(lens) - lens)
                gidx = np.repeat(base, lens)
                gidx += np.arange(tot)
                slots = self._sorted[gidx]
                if self._dirty:
                    slots = slots[self._visible[slots]]
                out.append(slots)
        if self._pending:
            pend = self._pending_array()
            pend = pend[self.active[pend]]
            if pend.size:
                out.append(pend)
        if not out:
            return np.empty(0, dtype=np.int64)
        return out[0] if len(out) == 1 else np.concatenate(out)

    def _reach(self, margin: float) -> IntArray:
        """Per-axis cell reach `e`: how many neighbouring cells a query with
        this margin must include on each side (at least 1, the classic
        27-cell stencil, since cw >= cutoff >= any query radius)."""
        return np.maximum(1, np.ceil(float(margin) / self.cw).astype(np.int64))

    def _offsets(self, e: int) -> IntArray:
        """Cached [-e..e] int64 offset vector (one allocation per distinct
        reach, ever)."""
        o = self._offs_cache.get(e)
        if o is None:
            o = np.arange(-e, e + 1, dtype=np.int64)
            self._offs_cache[e] = o
        return o

    def _ghost_lut(self, reach: IntArray) -> tuple[IntArray, IntArray, IntArray]:
        """(pad_strides, block_delta, lut) for a uniform reach `e`.

        A point's cell block is the same fixed set of index offsets in a grid
        padded by `e` cells on every side; the padding removes the per-axis `mod`
        and the per-axis multiply-by-stride, which between them were a dozen numpy
        calls per query and dominated a single-point query outright. `lut` maps a
        padded index straight to the real cell id.

        Only valid where every axis has more cells than the block is wide;
        otherwise a padded block would name the same real cell twice, and a
        duplicate candidate would be counted twice. `_point_cells` checks.
        """
        key = tuple(int(e) for e in reach)
        got = self._lut_cache.get(key)
        if got is not None:
            return got
        e = np.asarray(key, dtype=np.int64)
        pad = self.nc + 2 * e
        grids = np.meshgrid(
            *[np.arange(int(p), dtype=np.int64) for p in pad], indexing="ij"
        )
        real = [
            np.mod(g - e[a], self.nc[a]) * self._strides[a] for a, g in enumerate(grids)
        ]
        lut = sum(real).ravel()
        pstr = np.ones(self.dim, dtype=np.int64)
        for a in range(self.dim - 2, -1, -1):
            pstr[a] = pstr[a + 1] * pad[a + 1]
        offs = [self._offsets(int(e[a])) * pstr[a] for a in range(self.dim)]
        delta = offs[0][:, None] + offs[1][None, :]
        if self.dim == 3:
            delta = delta[:, :, None] + offs[2][None, None, :]
        got = (pstr, delta.ravel(), lut)
        self._lut_cache[key] = got
        return got

    def gather_points(self, points: FloatArray, margin: float) -> IntArray:
        """Active slots in every cell within `margin` of any of `points`.

        A superset of the true neighbourhood, which is all the callers need: the
        exact test (a distance, or a body membership predicate) runs afterwards
        on the gathered candidates. Cost is the union of the points' cell blocks,
        not the AABB of the points, so a body that straddles the periodic
        boundary or runs diagonally across the box costs what it covers rather
        than the whole box.
        """
        blocks, _ = self._point_cells(points, margin)
        return np.sort(self._gather_cells(np.unique(blocks)))

    def gather_aabb(
        self, lo: FloatArray, hi: FloatArray, margin: float = 0.0, sort: bool = True
    ) -> IntArray:
        """Active slots in cells overlapping the periodic AABB [lo - margin,
        hi + margin]. `lo`/`hi` may lie outside the box.

        A superset of the body, deliberately: the caller decides membership with
        the exact test. Sorted ascending by default, so `rng.choice` over the
        result is reproducible; a caller that filters first can pass sort=False
        and sort the (much shorter) surviving list itself.
        """
        lo = np.asarray(lo, float).reshape(self.dim) - float(margin)
        hi = np.asarray(hi, float).reshape(self.dim) + float(margin)
        cols = []
        for a in range(self.dim):
            # scalar arithmetic in Python, not numpy: at ~150 candidates this
            # function is dispatch-bound, and `np.floor` on a scalar is a ufunc
            # call. The `mod` is skipped when the span cannot leave the grid,
            # which is the usual case for a body away from a face.
            nca = int(self.nc[a])
            i0 = math.floor(float(lo[a]) * float(self._inv_cw[a]))
            i1 = math.floor(float(hi[a]) * float(self._inv_cw[a]))
            if i1 - i0 + 1 >= nca:
                col = self._axis_cache(a)
            elif i0 >= 0 and i1 < nca:
                col = np.arange(i0, i1 + 1, dtype=np.int64) * self._strides[a]
            else:
                col = (
                    np.mod(np.arange(i0, i1 + 1, dtype=np.int64), nca)
                    * self._strides[a]
                )
            cols.append(col)
        cells = cols[0][:, None] + cols[1][None, :]
        if self.dim == 3:
            cells = cells[:, :, None] + cols[2][None, None, :]
        cand = self._gather_cells(cells.ravel())
        return np.sort(cand) if sort else cand

    def _axis_cache(self, a: int) -> IntArray:
        """Cached stride-multiplied index column for a FULL axis, used when a
        span covers every cell on axis `a`."""
        key = ("axis", a)
        o = self._offs_cache.get(key)
        if o is None:
            o = np.arange(int(self.nc[a]), dtype=np.int64) * self._strides[a]
            self._offs_cache[key] = o
        return o

    # ---------------------------------------------------------------- queries
    @property
    def capacity(self) -> int:
        """Slot-table size; also the pad index `query` returns."""
        return self._capacity

    @property
    def coords(self) -> FloatArray:
        """Slot-indexed (capacity, dim) view of the coordinate table."""
        return self._coordsT.T

    @property
    def data(self) -> FloatArray:
        """Slot-indexed coordinate table. `idxs` from `query` index into it; the
        pad index is `data.shape[0]`, so the `min(idx, n - 1)` clamp the weighers
        apply lands on a real (if arbitrary) row, exactly as with a kd-tree."""
        return self._coordsT.T

    def _point_cells(
        self, points: FloatArray, margin: float
    ) -> tuple[IntArray, BoolArray]:
        """Cell ids of each point's own neighbourhood block: (m, O), no repeats.

        An axis whose block would wrap onto itself (2e + 1 >= nc, i.e. a box only
        a couple of cells wide) takes every cell on that axis instead, which is
        the same set without the duplicates. Duplicates would double-count a
        neighbour, so this is correctness, not tidiness.
        """
        ci = self._cell_coords(points)
        reach = self._reach(margin)
        need_image = np.zeros(self.dim, dtype=bool)
        wide = False
        lo, hi = ci.min(axis=0), ci.max(axis=0)
        for a in range(self.dim):
            nca, e = int(self.nc[a]), int(reach[a])
            if 2 * e + 1 >= nca:
                wide = True
                need_image[a] = True
            else:
                # A pair can need the periodic image only if some point's block
                # crosses the seam, or if the block itself is wider than half the
                # box. Otherwise every displacement is under L/2 and np.round of
                # it is exactly zero, so the correction is a no-op worth skipping:
                # `np.round` is the single most expensive step in the distance.
                wraps = bool(lo[a] - e < 0 or hi[a] + e >= nca)
                need_image[a] = wraps or (e + 1) * self.cw[a] > 0.5 * self.box[a]

        if wide:
            # a box only a couple of cells across: take every cell on the narrow
            # axes, which is the same set without the duplicates a wrapped block
            # would produce
            cols = []
            for a in range(self.dim):
                nca, e = int(self.nc[a]), int(reach[a])
                if 2 * e + 1 >= nca:
                    col = np.broadcast_to(
                        np.arange(nca, dtype=np.int64), (len(ci), nca)
                    )
                else:
                    col = np.mod(ci[:, a : a + 1] + self._offsets(e), nca)
                cols.append(col * self._strides[a])
            out = cols[0][:, :, None] + cols[1][:, None, :]
            if self.dim == 3:
                out = out[:, :, :, None] + cols[2][:, None, None, :]
            return out.reshape(len(ci), -1), need_image

        pstr, delta, lut = self._ghost_lut(reach)
        pi = (ci + reach) @ pstr
        return lut[pi[:, None] + delta[None, :]], need_image

    def _neighbor_pairs(
        self, points: FloatArray, margin: float
    ) -> tuple[IntArray, IntArray, BoolArray | None]:
        """(row, slot) for every (point, candidate-in-its-cell-block) pair.

        This is the shape that matters. Gathering the UNION of the points' cell
        blocks and forming a dense (m, union) matrix costs 6x more distances than
        the points actually need, because each point only ever looks at its own
        block. Here the CSR segments of one point's block are gathered by the
        repeat trick, all points at once, so the pair list is the sparse thing it
        should be and there is still no Python loop over points or cells.
        """
        m = len(points)
        blocks, need_image = self._point_cells(points, margin)  # (m, O)
        rows_out, slots_out = [], []

        if self._sorted.size:
            flat = blocks.ravel()
            s = self._start[flat]
            lens = self._count[flat]  # one gather, not start[c+1]-start[c]
            tot = int(lens.sum())
            if tot:
                # one repeat, not three: fold the segment start and the running
                # offset into a single base before expanding
                base = s - (np.cumsum(lens) - lens)
                gidx = np.repeat(base, lens)
                gidx += np.arange(tot)
                slots = self._sorted[gidx]
                rows = np.repeat(np.arange(m), lens.reshape(m, -1).sum(axis=1))
                if self._dirty:
                    keep = self._visible[slots]
                    rows, slots = rows[keep], slots[keep]
                rows_out.append(rows)
                slots_out.append(slots)

        if self._pending:
            # The pending slots are few (see PENDING_THRESHOLD) and disjoint from
            # the CSR set, so they are paired with every point and left to the
            # distance test. Filtering them by cell first cost a `np.unique` over
            # the whole block array, which was more than the handful of extra
            # distances it saved. Their images are unrestricted, so the skip
            # optimisation in `_pair_dists` is dropped for this call.
            pend = self._pending_array()
            pend = pend[self.active[pend]]
            if pend.size:
                rows_out.append(np.repeat(np.arange(m), len(pend)))
                slots_out.append(np.tile(pend, m))
                need_image = np.ones(self.dim, dtype=bool)

        if not rows_out:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                need_image,
            )
        if len(rows_out) == 1:
            return rows_out[0], slots_out[0], need_image
        # two sources; `query` scatters on the assumption that rows ascend
        rows = np.concatenate(rows_out)
        slots = np.concatenate(slots_out)
        o = np.argsort(rows, kind="stable")
        return rows[o], slots[o], need_image

    def _pending_array(self) -> IntArray:
        """The pending list as a cached int64 array; the cache is invalidated
        by every append so gathers never re-materialise an unchanged list."""
        if self._pending_arr is None:
            self._pending_arr = np.fromiter(
                self._pending, dtype=np.int64, count=len(self._pending)
            )
        return self._pending_arr

    def _pair_dists(
        self,
        points: FloatArray,
        rows: IntArray,
        slots: IntArray,
        need_image: BoolArray | None = None,
    ) -> FloatArray:
        """Minimum-image distance of each (point, slot) pair.

        Three things here are load-bearing, all measured:

        * Each axis is gathered on its own (`coords[slots, a]`), which lands in a
          fresh contiguous array. Slicing a column out of `coords[slots]` leaves
          a stride-3 view, and every ufunc after it runs several times slower.
        * The image correction is skipped on axes that provably cannot wrap (see
          `_point_cells`). `np.round` is the most expensive step in the whole
          distance, and skipping it is exact rather than approximate: for
          |x| <= 0.5, `np.round(x)` is zero.
        * The accumulate order (dx^2, then + dy^2, then + dz^2) followed by one
          sqrt reproduces scipy's accumulation bit for bit. `einsum` does not: it
          contracts through an FMA kernel and lands one ulp away, which is enough
          to flip an acceptance and break trajectory identity.
        """
        acc = None
        ptsT = np.ascontiguousarray(points.T)
        for a in range(self.dim):
            L = self.box[a]
            dd = self._coordsT[a][slots] - ptsT[a][rows]
            if need_image is None or need_image[a]:
                dd -= np.round(dd / L) * L
            dd *= dd
            acc = dd if acc is None else acc + dd
        assert acc is not None  # dim >= 1, the loop always runs
        return np.sqrt(acc)

    def query(
        self,
        points: FloatArray,
        upper_cutoff: float,
        inclusive: bool = False,
        min_width: int = 2,
        min_only: bool = False,
    ) -> tuple[FloatArray, IntArray]:
        """Neighbours of each point within `upper_cutoff`.

        `min_only=True` returns a single column: the nearest neighbour's distance
        (inf if there is none) and a pad index. That is everything a hard-core
        weight needs -- it reads the overlap test and nothing else -- and it skips
        the per-row scatter and sort, which is most of what a query costs once the
        arithmetic is this small. The caller must not read `idxs`.

        Returns (dists, idxs), one row per point, sorted ascending, padded with
        `inf` / `data.shape[0]`. Row width is the largest in-range count over the
        points, floored at `min_width` (aux3d reads columns 0 and 1:). Matching
        `cKDTree.query(distance_upper_bound=rc)`, a neighbour at exactly
        `upper_cutoff` is excluded unless `inclusive`.
        """
        pts = np.mod(np.asarray(points, float).reshape(-1, self.dim), self.box)
        m = len(pts)
        pad = self._capacity

        def empty(k: int) -> tuple[FloatArray, IntArray]:
            k = max(1, int(k))
            return (np.full((m, k), np.inf), np.full((m, k), pad, dtype=np.int64))

        if self.n_active == 0 or m == 0:
            return empty(1 if min_only else min_width)
        rows, slots, img = self._neighbor_pairs(pts, upper_cutoff)
        if rows.size == 0:
            return empty(1 if min_only else min_width)
        d = self._pair_dists(pts, rows, slots, img)
        good = (d <= upper_cutoff) if inclusive else (d < upper_cutoff)
        rows, slots, d = rows[good], slots[good], d[good]
        counts = np.bincount(rows, minlength=m)

        if min_only:
            dists = np.full((m, 1), np.inf)
            if rows.size:
                # `rows` ascends, so each row is one contiguous segment
                nz = counts > 0
                starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
                dists[nz, 0] = np.minimum.reduceat(d, starts[nz])
            return dists, np.full((m, 1), pad, dtype=np.int64)

        k = max(int(min_width), int(counts.max()), 1)

        dists = np.full((m, k), np.inf)
        idxs = np.full((m, k), pad, dtype=np.int64)
        if rows.size:
            # `rows` is already ascending, so the scatter needs no sort: the
            # within-row column is just the running index. Sorting each short row
            # afterwards beats a lexsort over every pair.
            col = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
            dists[rows, col] = d
            idxs[rows, col] = slots
            order = np.argsort(dists, axis=1, kind="stable")
            dists = np.take_along_axis(dists, order, axis=1)
            idxs = np.take_along_axis(idxs, order, axis=1)
        return dists, idxs

    def count_within(self, point: FloatArray, radius: float) -> int:
        """Number of active slots within `radius` of `point`, inclusive -- the
        `len(query_ball_point(p, r))` the hard-core overlap test asks for."""
        d = self.query_point(point, radius, sort=False)[0]
        if d.size == 0:
            return 0
        return int((d <= radius).sum())

    def query_point(
        self,
        point: FloatArray,
        radius: float,
        sort: bool = True,
        inclusive: bool = True,
    ) -> tuple[FloatArray, IntArray]:
        """(dists, slots) within `radius` of ONE point.

        A single point does not need the row bookkeeping `query` carries -- the
        bincount, the scatter, the per-row sort. Translations are the hottest
        single-point callers and this path is what keeps their neighbour cost
        below the rest of the move.
        """
        p = np.mod(np.asarray(point, float).reshape(1, self.dim), self.box)
        if self.n_active == 0:
            return np.empty(0), np.empty(0, dtype=np.int64)
        rows, slots, img = self._neighbor_pairs(p, radius)
        if rows.size == 0:
            return np.empty(0), np.empty(0, dtype=np.int64)
        d = self._pair_dists(p, rows, slots, img)
        keep = (d <= radius) if inclusive else (d < radius)
        d, slots = d[keep], slots[keep]
        if sort and d.size:
            o = np.argsort(d, kind="stable")
            d, slots = d[o], slots[o]
        return d, slots

    def neighbours_within(
        self, point: FloatArray, radius: float, inclusive: bool = True
    ) -> tuple[IntArray, FloatArray]:
        """(slots, dists) within `radius` of a single point, unsorted."""
        p = np.mod(np.asarray(point, float).reshape(1, self.dim), self.box)
        if self.n_active == 0:
            return np.empty(0, dtype=np.int64), np.empty(0)
        rows, slots, img = self._neighbor_pairs(p, radius)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0)
        d = self._pair_dists(p, rows, slots, img)
        keep = (d <= radius) if inclusive else (d < radius)
        return slots[keep], d[keep]

    def matrix_at(self, idxs: IntArray) -> BoolArray:
        """Boolean "is a frozen obstacle" for each returned neighbour index, with
        the pad index clamped. Pads always carry an infinite distance, so their
        value never changes a weight."""
        i = np.minimum(np.asarray(idxs), self._capacity - 1)
        return self.is_matrix[i]

    def species_at(self, idxs: IntArray) -> IntArray:
        """Species id for each returned neighbour index, with the pad index
        clamped, exactly as `matrix_at` does.

        The clamp is safe for the same reason: pads carry an infinite distance, so
        whichever species they read can never reach a weight or an overlap
        verdict. Reading through this accessor rather than `self.species[idxs]`
        directly is what keeps that true at call sites that have not thought
        about pads.
        """
        i = np.minimum(np.asarray(idxs), self._capacity - 1)
        return self.species[i]

    def active_slots(self) -> IntArray:
        """Ascending slot ids of every active particle (fluid AND matrix);
        diagnostics and tests use this, the hot path never does."""
        return np.nonzero(self.active[: self._n_slots])[0]
