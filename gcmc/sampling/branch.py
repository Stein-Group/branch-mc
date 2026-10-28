"""BRANCH selection: k-clique enumeration over non-overlapping centres.

Adjacency means "non-overlapping" (centres at least `overlap_distance` apart), so a
k-clique is a set of k mutually non-overlapping disks/spheres. This counts them
and returns one example clique, which the engine uses as an alternative to the
sequential Rosenbluth selection. Counting and sampling are deterministic (no
random draw): small cases use closed-form / matrix expressions, k>=4 uses kClist
with a degeneracy (or degree, for k==4) ordering.

This is the performance-critical path you may replace with C. Inputs and outputs
are flat: a centres array and k in, a count and a sorted index tuple out, with
the cutoff and box passed explicitly.

Faithfully transcribed from the original; it is not part of the verified-mode
set, so it is pinned to the original's own outputs rather than to physics.

References: Danisch, Blin, Sagot (WWW 2018); Chiba, Nishizeki (SIAM J. Comput. 1985).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from scipy.spatial import KDTree

from .._types import RNG as _RNG
from .._types import BoolArray, FloatArray, IntArray
from ..forcefield.lennard_jones import lj_pair_energy
from ..forcefield.overlap import overlap_radius
from ..forcefield.pair_table import PairTable, table_pair_energy_vec


def _build_adj(
    centers: FloatArray,
    overlap_distance: float,
    box: FloatArray,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
) -> tuple[BoolArray, FloatArray]:
    """Compatibility ("non-conflict") adjacency of the trial cloud: adj[i, j]
    is True iff trials i and j do NOT overlap under the periodic minimum image,
    so a non-overlapping k-subset of trials is exactly a k-clique of this
    graph. Built once per pool from a kd-tree pair query (this is pool-local,
    O(pool size), and deliberately outside the engine's cell-list rule).
    Returns (adj, wrapped centers).

    TYPED (`species` and `floor` given, A1.4): two trials conflict below the
    PER-PAIR floor `c*sigma_ij` between their species, not below one scalar. The
    kd-tree is queried at the LARGEST floor present -- a superset of the real
    conflicts -- and each returned pair is then tested against its own floor. The
    same two-step the multi-species sweep uses for its eviction window, and for
    the same reason: a single query radius must not be allowed to drop a
    conflicting pair before it is tested.
    """
    n = len(centers)
    shifted = np.mod(centers, box)
    tree = KDTree(shifted, boxsize=box)
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    if species is None or floor is None:
        pairs = tree.query_pairs(
            r=overlap_radius(overlap_distance), output_type="ndarray"
        )
    else:
        spc = np.asarray(species, dtype=np.int64)
        fl = np.asarray(floor, dtype=np.float64)
        present = np.unique(spc[:n]) if n else np.zeros(0, dtype=np.int64)
        if not len(present):
            return adj, shifted
        r_max = overlap_radius(float(fl[np.ix_(present, present)].max()))
        pairs = tree.query_pairs(r=r_max, output_type="ndarray")
        if len(pairs) > 0:
            d = shifted[pairs[:, 0]] - shifted[pairs[:, 1]]
            d -= np.round(d / box) * box
            r = np.sqrt((d**2).sum(axis=1))
            # `overlap_radius` takes an array of floors as readily as one
            # diameter, which is exactly what a per-pair test needs
            keep = r <= overlap_radius(fl[spc[pairs[:, 0]], spc[pairs[:, 1]]])
            pairs = pairs[keep]
    if len(pairs) > 0:
        adj[pairs[:, 0], pairs[:, 1]] = False
        adj[pairs[:, 1], pairs[:, 0]] = False
    return adj, shifted


def _degeneracy_rank(adj: BoolArray, n: int) -> IntArray:
    """Bucket-queue degeneracy ordering."""
    nbrs = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = [len(nb) for nb in nbrs]
    max_d = max(deg) if n > 0 else 0
    buckets: list[set[int]] = [set() for _ in range(max_d + 1)]
    for v in range(n):
        buckets[deg[v]].add(v)
    rank = np.empty(n, dtype=np.int64)
    removed = bytearray(n)
    min_d = 0
    for i in range(n):
        while not buckets[min_d]:
            min_d += 1
        v = buckets[min_d].pop()
        rank[v] = i
        removed[v] = 1
        for w in nbrs[v]:
            if not removed[w]:
                old = deg[w]
                buckets[old].discard(w)
                deg[w] = old - 1
                buckets[old - 1].add(w)
                if old - 1 < min_d:
                    min_d = old - 1
    return rank


def _degree_rank(adj: BoolArray, n: int) -> IntArray:
    """Simple degree sort. One vectorized call."""
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(adj.sum(axis=1))] = np.arange(n)
    return rank


def _count_sub(A: BoolArray, k: int) -> int:
    """Count k-cliques in a small bool adjacency matrix."""
    d = A.shape[0]
    if d < k or k <= 0:
        return 0
    if k == 1:
        return d
    if k == 2:
        return int(np.count_nonzero(A)) // 2
    if k == 3:
        Af = A.astype(np.float64) if A.dtype != np.float64 else A
        return int(np.sum(Af * (Af @ Af))) // 6
    total = 0
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1 :])[0] + (i + 1)
        if fwd.size >= k - 1:
            total += _count_sub(A[np.ix_(fwd, fwd)], k - 1)
    return total


def _find_sub(A: BoolArray, k: int) -> Sequence[int] | None:
    """Find one k-clique in a small adjacency. Returns index list or None."""
    d = A.shape[0]
    if d < k or k <= 0:
        return None
    if k == 1:
        return [0]
    if k == 2:
        for i in range(d - 1):
            c = int(np.argmax(A[i, i + 1 :]))
            if A[i, i + 1 + c]:
                return [i, i + 1 + c]
        return None
    if k == 3:
        Af = A.astype(np.float64) if A.dtype != np.float64 else A
        M = Af * (Af @ Af)
        for i in range(d - 1):
            row = M[i, i + 1 :]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(A[i, :j] & A[j, :j])[0]
            if len(cn) == 0:
                cn = np.where(A[i, j + 1 :] & A[j, j + 1 :])[0]
                if len(cn) > 0:
                    return [i, j, int(cn[0]) + j + 1]
            else:
                return [int(cn[0]), i, j]
        return None
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1 :])[0] + (i + 1)
        if fwd.size >= k - 1:
            inner = _find_sub(A[np.ix_(fwd, fwd)], k - 1)
            if inner is not None:
                return [i] + [int(fwd[x]) for x in inner]
    return None


def _kClist_count_and_find(adj: BoolArray, k: int) -> tuple[int, Sequence[int] | None]:
    """General k-clique count plus one witness clique, via vertex ordering:
    each vertex v owns the cliques in which it is the lowest-ranked member,
    counted inside its forward neighbourhood (degeneracy rank in general,
    plain degree rank at k = 4 where it measured faster). CS note: this is
    the standard ordered-enumeration trick that makes clique counting
    O(n * per-neighbourhood work) instead of naive C(n, k)."""
    n = adj.shape[0]
    if n < k:
        return 0, None

    rank = _degree_rank(adj, n) if k == 4 else _degeneracy_rank(adj, n)

    total = 0
    sample = None
    for v in range(n):
        fwd = np.where(adj[v] & (rank > rank[v]))[0]
        if fwd.size < k - 1:
            continue
        sub = adj[np.ix_(fwd, fwd)]
        c = _count_sub(sub, k - 1)
        total += c
        if sample is None and c > 0:
            inner = _find_sub(sub, k - 1)
            if inner is not None:
                sample = tuple(sorted([v] + [int(fwd[i]) for i in inner]))
    return total, sample


def _count_from_adj(adj: BoolArray, k: int) -> tuple[int, Sequence[int] | None]:
    """(number of k-cliques, one sample clique or None) with hand-rolled fast
    paths: k <= 0 counts the empty clique, k = 1 counts vertices, k = 2 edges,
    k = 3 triangles via row intersections, and everything above delegates to
    `_kClist_count_and_find`. The count is the hard-sphere partition number
    Z_k of the pool -- the quantity the acceptance ratio consumes."""
    n = adj.shape[0]
    if k <= 0:
        return 1, None
    if n < k:
        return 0, None
    if k == 1:
        return (n, (0,)) if n > 0 else (0, None)
    if k == 2:
        count = int(np.count_nonzero(adj)) // 2
        if count == 0:
            return 0, None
        r0 = int(np.argmax(adj.any(axis=1)))
        c0 = int(np.argmax(adj[r0]))
        return count, tuple(sorted((r0, c0)))
    if k == 3:
        Af = adj.astype(np.float64)
        M = Af * (Af @ Af)
        count = int(np.sum(M)) // 6
        if count == 0:
            return 0, None
        for i in range(n - 1):
            row = M[i, i + 1 :]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(adj[i, :j] & adj[j, :j])[0]
            if len(cn) > 0:
                return count, tuple(sorted((int(cn[0]), i, j)))
            cn = np.where(adj[i, j + 1 :] & adj[j, j + 1 :])[0]
            if len(cn) > 0:
                return count, tuple(sorted((i, j, int(cn[0]) + j + 1)))
        return count, None
    return _kClist_count_and_find(adj, k)


def _sample_sub(A: BoolArray, k: int, rng: _RNG) -> list[int] | None:
    """Draw one k-clique UNIFORMLY from a small adjacency, or None if there is
    none. Counting descent: `_count_sub` already partitions the k-cliques by
    their lowest-index member, so choosing that member in proportion to its
    share and recursing inside its forward neighbourhood visits every clique
    with probability exactly 1 / (number of k-cliques). Costs one extra counting
    pass per level, i.e. the same order as the count itself."""
    d = A.shape[0]
    if k <= 0:
        return []
    if d < k:
        return None
    if k == 1:
        return [int(rng.integers(d))]
    counts: list[int] = []
    fwds: list[IntArray] = []
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1 :])[0] + (i + 1)
        fwds.append(fwd)
        counts.append(
            _count_sub(A[np.ix_(fwd, fwd)], k - 1) if fwd.size >= k - 1 else 0
        )
    total = int(sum(counts))
    if total == 0:
        return None
    draw = int(rng.integers(total))
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if draw < acc:
            fwd = fwds[i]
            inner = _sample_sub(A[np.ix_(fwd, fwd)], k - 1, rng)
            assert inner is not None  # c > 0 guarantees a completion exists
            return [i] + [int(fwd[x]) for x in inner]
    raise AssertionError("unreachable: draw < total")


def _uniform_from_adj(
    adj: BoolArray, k: int, rng: _RNG
) -> tuple[int, Sequence[int] | None]:
    """(number of k-cliques, one UNIFORMLY drawn k-clique). Same ownership
    scheme as `_kClist_count_and_find` -- each vertex owns the cliques in which
    it is lowest-ranked -- so the per-vertex counts it already computes are
    exactly the weights the first choice needs."""
    n = adj.shape[0]
    if k <= 0:
        return 1, None
    if n < k:
        return 0, None
    if k == 1:
        return n, ((int(rng.integers(n)),) if n > 0 else None)

    rank = _degree_rank(adj, n) if k == 4 else _degeneracy_rank(adj, n)
    counts = np.zeros(n, dtype=np.int64)
    fwds: list[IntArray] = []
    for v in range(n):
        fwd = np.where(adj[v] & (rank > rank[v]))[0]
        fwds.append(fwd)
        if fwd.size >= k - 1:
            counts[v] = _count_sub(adj[np.ix_(fwd, fwd)], k - 1)
    total = int(counts.sum())
    if total == 0:
        return 0, None
    v = int(np.searchsorted(np.cumsum(counts), rng.integers(total), side="right"))
    fwd = fwds[v]
    inner = _sample_sub(adj[np.ix_(fwd, fwd)], k - 1, rng)
    assert inner is not None  # counts[v] > 0 guarantees a completion
    return total, tuple(sorted([v] + [int(fwd[i]) for i in inner]))


def _parse_box(box_lengths: float | FloatArray, dim: int) -> FloatArray:
    """Broadcast a scalar box length to a (dim,) array; pass arrays through."""
    box = np.atleast_1d(np.asarray(box_lengths, dtype=np.float64))
    if box.size == 1:
        box = np.full(dim, float(box[0]))
    return box


def count_and_sample(
    centers: FloatArray,
    k: int,
    overlap_distance: float,
    *,
    box_lengths: float | FloatArray,
    rng: _RNG | None = None,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
) -> tuple[int, Sequence[int] | None]:
    """Count non-overlapping k-disk configurations and return one example.
    Adjacency means a pair is closer than `overlap_distance` (the exclusion
    floor: the hard-core diameter, or the 0.85 LJ edge).
    Returns (count, sample) with sample a sorted index tuple or None.

    With `rng` the example is drawn UNIFORMLY over the k-cliques, which is what
    the acceptance ratio's log Z term assumes; the engine always passes one. The
    rng-free path keeps the original's deterministic lowest-in-order witness and
    exists only for the `orig/` count-parity check -- do not sample with it. The
    witness is NOT uniform (measured: at k = 3 over a third of the cliques are
    unreachable at any vertex labelling), which biased the equilibrium density.

    TYPED (`species` and `floor`): conflict is decided by the PER-PAIR floor
    `c*sigma_ij`. Only the adjacency changes -- the counting and the uniform draw
    over cliques are graph operations and know nothing about species, which is
    what `multispecies_plan.md` P5 means by the DP being type-blind once the
    conflict graph is built."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    box = _parse_box(box_lengths, centers.shape[1] if n > 0 else 1)
    if k <= 0 or n < k:
        return 0, None
    if k == 1:
        # one disk has no pair, so no overlap floor is consulted at all and the
        # typed and untyped answers are identical by construction
        return n, ((int(rng.integers(n)) if rng is not None else 0,) if n > 0 else None)
    adj, _ = _build_adj(centers, overlap_distance, box, species, floor)
    if rng is not None:
        return _uniform_from_adj(adj, k, rng)
    return _count_from_adj(adj, k)


def count_and_sample_branch(
    centers: FloatArray,
    k: int,
    overlap_distance: float,
    *,
    box_lengths: float | FloatArray,
) -> tuple[int, int, Sequence[int] | None, Sequence[int] | None]:
    """Count k- and (k+1)-cliques over the same centres with one adjacency build.
    Returns (count_k, count_k1, sample_k, sample_k1)."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    dim = centers.shape[1] if n > 0 else 1
    box = _parse_box(box_lengths, dim)
    if k <= 0:
        return 0, 0, None, None
    adj, _ = _build_adj(centers, overlap_distance, box)
    count_k, sample_k = _count_from_adj(adj, k)
    count_k1, sample_k1 = _count_from_adj(adj, k + 1)
    return count_k, count_k1, sample_k, sample_k1


# ---------------------------------------------------------------------------
# True LJ + branch (no auxiliary): Boltzmann-weighted sum over k-cliques.
#
# Adjacency is the SAME non-overlap graph the hard-core path uses (built by
# _build_adj at the exclusion floor `lower` = 0.9, where LJ repulsion already
# makes the weight negligible): an edge means a pair is at least `lower` apart,
# so a k-clique is k mutually non-overlapping trial centres, a group that can be
# inserted at once. There is no upper bound -- two members far apart simply do
# not interact and contribute a pair factor of 1, so cliques may span the box.
#
# Each clique's weight is exp(-beta * U_group) = (member-vs-system) *
# (intra-group). The member-vs-system factor was already computed once per trial
# in the neighbour build and is carried in `member_weights`; only the cheap
# intra-group pairs (within rc) are evaluated here. The weighted sum is taken in
# log space (LJ weights span many orders of magnitude) and one clique is
# reservoir-sampled proportional to its weight in a single streaming pass.
# ---------------------------------------------------------------------------


def _enumerate_cliques(adj: BoolArray, k: int) -> Iterator[tuple[int, ...]]:
    """Yield every k-clique as a sorted tuple of vertex indices. Uses the same
    degeneracy-ordered recursion as the counter, but emits members at the leaf
    instead of incrementing. Streaming: a generator, nothing stored."""
    n = adj.shape[0]
    if k <= 0 or n < k:
        return
    if k == 1:
        for v in range(n):
            yield (v,)
        return
    rank = _degeneracy_rank(adj, n)

    def recurse(
        cand: list[int], need: int, prefix: Sequence[int]
    ) -> Iterator[tuple[int, ...]]:
        if need == 0:
            yield tuple(sorted(prefix))
            return
        if len(cand) < need:
            return
        for ii, v in enumerate(cand):
            nxt = [u for u in cand[ii + 1 :] if adj[v, u]]
            if len(nxt) >= need - 1:
                yield from recurse(nxt, need - 1, [*prefix, v])

    for v in range(n):
        fwd = [u for u in range(n) if adj[v, u] and rank[u] > rank[v]]
        if len(fwd) >= k - 1:
            yield from recurse(fwd, k - 1, [v])


def _intra_lj_factor(
    members: FloatArray,
    box: FloatArray,
    rc: float,
    beta: float,
    member_species: IntArray | None = None,
    table: PairTable | None = None,
) -> float:
    """exp(-beta * LJ energy among `members`): pairs within rc, minimum image.
    Returns 1.0 for fewer than two members (no pairs), so depth-0 weights are
    exactly the carried single-particle weight with nothing added.

    TYPED (`member_species` and `table` given): each pair's energy is the
    per-pair 12-6 form for the two species, grouped by species pair so the table
    is read once per occupied combination rather than once per pair. At one
    species with a unit table this is `lj_pair_energy` bit for bit, which is what
    keeps the single-species clique weights unchanged.
    """
    m = len(members)
    if m < 2:
        return 1.0
    ii, jj = np.triu_indices(m, k=1)
    dvec = members[ii] - members[jj]
    dvec -= np.round(dvec / box) * box
    r = np.sqrt((dvec**2).sum(axis=1))
    inside = (r < rc) & (r > 0)
    if not inside.any():
        return 1.0
    if member_species is None or table is None:
        u = float(np.sum(lj_pair_energy(r[inside])))
    else:
        # ONE gather, no grouping loop. This function is called once per CLIQUE
        # and `ljbranch_weight_sum` enumerates every clique, so a Python loop over
        # type combinations here sits inside the enumeration: it cost 53 s on a
        # single depth-4 QA matrix point before this was vectorised.
        spc = np.asarray(member_species, dtype=np.int64)
        u = float(
            np.sum(
                table_pair_energy_vec(
                    r[inside], spc[ii][inside], spc[jj][inside], table
                )
            )
        )
    return float(np.exp(-beta * u))


# NOTE FOR ANYONE EDITING THE TWO KERNELS BELOW. `count_and_sample` and
# `ljbranch_weight_sum` are AST-LOCKED: `tests/unit/test_locked_invariants.py`
# hashes each function's parsed body against a committed baseline, and that hash
# is the whole guarantee that the combinatorial core has not drifted. The lock
# covers the DOCSTRING too, because a docstring is part of the body.
#
# So the internal stage tags inside them stay, where every other one in the
# package was removed for the public release: rewriting that prose moves the
# hash. This was tried during that cleanup and the gate caught it, which is the
# gate doing its job rather than being in the way. If a kernel genuinely has to
# change, re-baseline the hash in the test with the cause written down first.
def ljbranch_weight_sum(
    centers: FloatArray,
    member_weights: FloatArray,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
    *,
    rng: _RNG | None = None,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
    table: PairTable | None = None,
) -> tuple[float, int, FloatArray | None, list[int] | None]:
    """Sum the Boltzmann weights of every non-overlapping k-clique.

    Adjacency is non-overlap at `lower` (the same graph the hard-core counter
    builds, via _build_adj); there is no upper cutoff. A clique's weight is the
    product of its members' carried single-particle weights (member-vs-system,
    already computed in the neighbour build) times the intra-group LJ factor
    (member pairs within rc). The sum is accumulated in log space.

    If `rng` is given, one clique is reservoir-sampled proportional to its
    weight and its member centres are returned (the forward / insertion use);
    otherwise `chosen` is None (the reverse use needs only the sum).

    Returns (logW, count, chosen_coords_or_None, chosen_rows_or_None);
    (-inf, 0, None, None) if no clique exists.

    THE FOURTH RETURN VALUE, added at S3 with a deliberate re-baseline of the
    locked-body hash. `chosen_rows` is the drawn clique's row indices into
    `centers`. The typed insert path needs them: the species of an inserted atom
    is the species of the pool row it came from, and M21 decides provenance by
    row and explicitly never by comparing coordinates -- a reselected member and
    a trial atom that lands on it are indistinguishable by position.

    Recovering them outside this function would have meant either an exact-float
    coordinate lookup (the identification M21 forbids) or a second copy of the
    clique enumeration. `idx` is already computed on every iteration; the change
    is to stop throwing it away.

    Nothing about the weights or the draw moved. The reservoir test, its rng
    consumption and the accumulation order are untouched, and the counter digests
    are the evidence: 13/13 unchanged across this commit.

    TYPED (`species`, `floor` and `table`). Two things become per-pair, and they
    are the same two the multi-species sweep overrides: conflict is decided at
    `c*sigma_ij` rather than one `lower`, and the intra-clique factor is the
    per-pair 12-6 energy rather than the single-type one. `member_weights` needs
    no change -- the caller already folds the species-aware member-vs-system
    factor into it, exactly as it folds the single-type one.

    The clique ENUMERATION is untouched and type-blind, which is the whole reason
    this is a small change: once the conflict graph carries the species
    information, counting and drawing cliques is a graph operation.
    """
    centers = np.asarray(centers, float)
    member_weights = np.asarray(member_weights, float)
    box = _parse_box(box, centers.shape[1] if len(centers) else 1)
    adj, _ = _build_adj(centers, lower, box, species, floor)
    spc = None if species is None else np.asarray(species, dtype=np.int64)
    logW = -np.inf
    count = 0
    chosen = None
    chosen_rows = None
    for clique in _enumerate_cliques(adj, k):
        idx = list(clique)
        wprod = float(np.prod(member_weights[idx]))
        if wprod <= 0.0:
            continue
        members = centers[idx]
        w = wprod * _intra_lj_factor(
            members, box, rc, beta, None if spc is None else spc[idx], table
        )
        if w <= 0.0:
            continue
        count += 1
        logw = np.log(w)
        new_logW = np.logaddexp(logW, logw)
        # weighted reservoir draw in log space: keep this clique with
        # probability w / W_running = exp(logw - new_logW)
        if rng is not None and (logw - new_logW) >= np.log(rng.uniform(0.0, 1.0)):
            chosen = members
            chosen_rows = idx
        logW = new_logW
    if count == 0:
        return -np.inf, 0, None, None
    return float(logW), count, chosen, chosen_rows


# ---------------------------------------------------------------------------
# LJ + branch WITHOUT clique enumeration: the same sum, evaluated the way the
# hard-core counter evaluates its own.
#
# THE ASYMMETRY THIS REMOVES. `_count_from_adj` never visits a clique. Every
# hard-core clique weighs 1, so W_k is a COUNT, and a count is matrix algebra:
# triangles are `np.sum(A * (A @ A)) // 6`, and k >= 4 recurses per vertex into
# forward neighbourhoods and bottoms out in that same trace. Measured, M = 67,
# k = 4: 2.7 ms.
#
# `ljbranch_weight_sum` does the opposite. It walks all 375 758 cliques of that
# same pool in Python and spends ~49 us on each -- about twenty numpy calls on
# six-element arrays -- for 18.5 s. The bare generator walk is 476 ms of that,
# so 97% is per-clique dispatch overhead, not arithmetic.
#
# It was NOT the graph. Both paths call `_build_adj`; only the floor differs
# (1.0 sigma hard core against `lower` = 0.9 for LJ). Measured densities 0.843
# against 0.887, and 1.4x the 4-cliques. `rc` never enters the graph at all: a
# pair beyond it contributes a factor of 1, which is why a clique may span the
# box. The difference was never sparsity -- it was enumerate against don't.
#
# WHY IT DOES NOT HAVE TO ENUMERATE. The LJ clique weight has the same shape the
# count has:
#
#     w(C) = prod_{i in C} h_i  *  prod_{i<j in C} G_ij
#
#     h_i  = member_weights[i]           member-vs-system, already computed in
#                                        the neighbour build
#     G_ij = adj_ij * exp(-beta u_ij)    the conflict graph, edge-weighted;
#                                        exactly adj_ij beyond rc
#
# A product over vertices and edges is what matrix algebra sums. So W_k is the
# recursion `_count_sub` already runs, with one extra ingredient: descending into
# fwd(v), fold the edges from v into the subproblem's vertex weights,
# h' = h[fwd] * G[v, fwd]. The hard-core counter is the special case h = 1,
# G = adj, where the weight of every clique is 1 and W_k is its cardinality.
#
# The ownership argument is unchanged and is what makes this exact: every clique
# has a unique lowest-ordered member, so summing each vertex's forward
# contribution visits each clique exactly once.
#
# NOT AN APPROXIMATION. Same sum, same draw, different evaluation order.
# `_enumerate_cliques` and `_intra_lj_factor` stay exactly where they are as the
# oracle the agreement test runs against, and `ljbranch_weight_sum` keeps its
# locked body: nothing above this line moved.
#
#     M = 30, k = 4     584 ms  ->   2.1 ms      logW agrees to 2e-15
#     M = 45, k = 4    3893 ms  ->   2.9 ms      logW agrees to 5e-15
#     M = 67, k = 4   19019 ms  ->   5.5 ms      logW agrees to 9e-15
# ---------------------------------------------------------------------------


def _edge_weights(
    centers: FloatArray,
    adj: BoolArray,
    box: FloatArray,
    rc: float,
    beta: float,
    species: IntArray | None = None,
    table: PairTable | None = None,
) -> FloatArray:
    """G_ij = adj_ij * exp(-beta u_ij), in one O(M^2) pass over the whole pool.

    The truncation rule is `_intra_lj_factor`'s, verbatim: 0 < r < rc carries an
    energy, everything else a pair factor of 1.

    THIS IS WHERE THE TYPED PATH STOPS BEING EXPENSIVE. `table_pair_energy_vec`
    is read once per POOL here. The S3 report's 53 s finding was the same lookup
    sitting once per CLIQUE inside the enumeration; with no enumeration there is
    nowhere for that to recur.
    """
    d = centers[:, None, :] - centers[None, :, :]
    d -= np.round(d / box) * box
    r = np.sqrt((d**2).sum(-1))
    inside = (r < rc) & (r > 0)
    g = np.ones_like(r)
    if inside.any():
        if species is None or table is None:
            u = lj_pair_energy(r[inside])
        else:
            spc = np.asarray(species, dtype=np.int64)
            si = np.broadcast_to(spc[:, None], r.shape)[inside]
            sj = np.broadcast_to(spc[None, :], r.shape)[inside]
            u = table_pair_energy_vec(r[inside], si, sj, table)
        g[inside] = np.exp(-beta * u)
    g = g * adj
    np.fill_diagonal(g, 0.0)
    return g


def _wsum(g: FloatArray, h: FloatArray, k: int) -> float:
    """Sum of prod(h_i) prod(g_ij) over k-cliques. `_count_sub`, weighted.

    Each closed form is the hard-core one with the boolean adjacency replaced by
    the weighted one, and returns `_count_sub`'s integer at h = 1, g = adj.
    """
    d = h.shape[0]
    if k <= 0:
        return 1.0
    if d < k:
        return 0.0
    if k == 1:
        return float(h.sum())
    if k == 2:
        return 0.5 * float(h @ g @ h)
    if k == 3:
        s = np.sqrt(h)
        sym = g * s[:, None] * s[None, :]
        return float(np.sum(sym * (sym @ sym))) / 6.0
    total = 0.0
    for i in range(d - k + 1):
        fwd = np.where(g[i, i + 1 :] > 0.0)[0] + (i + 1)
        if fwd.size >= k - 1:
            total += h[i] * _wsum(g[fwd][:, fwd], h[fwd] * g[i, fwd], k - 1)
    return total


def _wladder(g: FloatArray, h: FloatArray, kmax: int) -> FloatArray:
    """[W_0, .., W_kmax] in one recursion: `_wsum` for every rung at once.

    `_wsum(g, h, k)` partitions the k-cliques by their lowest member and
    recurses, and a ladder built from it repeats that recursion once per rung:
    the tree for rung k contains the trees for every rung below it. Here one
    descent returns the whole vector, so the ladder costs what its top rung
    costs instead of the sum over rungs.

    BIT-IDENTICAL to `_wsum`, rung by rung, by construction rather than by
    tolerance: rungs 1-3 ARE `_wsum`'s closed forms on the same arrays, and
    every higher rung adds the same products in the same order of lowest
    member, skipping exactly the terms `_wsum` skips (whose value is zero).
    """
    d = h.shape[0]
    out = np.zeros(kmax + 1)
    out[0] = 1.0
    for k in range(1, min(kmax, 3) + 1):
        out[k] = _wsum(g, h, k)
    if kmax < 4 or d < 4:
        return out
    for i in range(d - 3):
        fwd = np.where(g[i, i + 1 :] > 0.0)[0] + (i + 1)
        if fwd.size < 3:
            continue
        sub = _wladder(g[fwd][:, fwd], h[fwd] * g[i, fwd], kmax - 1)
        for k in range(4, min(kmax, fwd.size + 1) + 1):
            out[k] += h[i] * sub[k - 1]
    return out


def _wsample(g: FloatArray, h: FloatArray, k: int, rng: _RNG) -> list[int] | None:
    """One k-clique drawn proportional to its weight. `_sample_sub`, weighted.

    `_wsum` already partitions the cliques by their lowest-ordered member, so
    choosing that member in proportion to its share and recursing draws every
    clique with probability exactly w(C) / W_k -- the same counting descent, with
    counts replaced by weights. One extra summing pass per level, so the same
    order as the sum, and no reservoir: the draw no longer consumes one rng value
    per clique.
    """
    d = h.shape[0]
    if k <= 0:
        return []
    if d < k:
        return None
    if k == 1:
        tot = float(h.sum())
        if tot <= 0.0:
            return None
        return [int(np.searchsorted(np.cumsum(h), rng.uniform(0.0, tot)))]
    share = np.zeros(d)
    fwds: list[IntArray | None] = [None] * d
    for i in range(d - k + 1):
        fwd = np.where(g[i, i + 1 :] > 0.0)[0] + (i + 1)
        fwds[i] = fwd
        if fwd.size >= k - 1:
            share[i] = h[i] * _wsum(g[fwd][:, fwd], h[fwd] * g[i, fwd], k - 1)
    total = float(share.sum())
    if total <= 0.0:
        return None
    i = int(np.searchsorted(np.cumsum(share), rng.uniform(0.0, total)))
    fwd = fwds[i]
    assert fwd is not None  # share[i] > 0 means the branch above ran
    inner = _wsample(g[fwd][:, fwd], h[fwd] * g[i, fwd], k - 1, rng)
    assert inner is not None  # share[i] > 0 guarantees a completion
    return [i] + [int(fwd[x]) for x in inner]


def lj_clique_weight_sum(
    centers: FloatArray,
    member_weights: FloatArray,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
    *,
    rng: _RNG | None = None,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
    table: PairTable | None = None,
) -> tuple[float, int, FloatArray | None, list[int] | None]:
    """Signature-compatible replacement for `ljbranch_weight_sum`.

    Returns (logW, count, chosen_coords_or_None, chosen_rows_or_None).

    `count` IS NOT A CLIQUE COUNT ANY MORE: it is 1 when W_k > 0 and 0 otherwise.
    Both call sites (`_ljbranch_forward`, `_ljbranch_reverse`) only ever test it
    against zero. If a caller ever needs the cardinality, that is `_wsum` with
    h = 1 and g = adj -- which is `_count_from_adj`, so call that.

    SCALE, and why the logaddexp chain is gone. W_k is homogeneous of degree k in
    h, so dividing h by its maximum and adding k*log(c) back afterwards is exact.
    That is what the log-space accumulation was protecting against: member
    weights spanning decades. The edge weights need no such protection -- g lies
    between exp(-beta u(lower)) and exp(beta eps), both O(1) at the temperatures
    this samples. Below about T* = 0.4 the attractive well makes g > 1 large
    enough that a big-k product could overflow; rescaling g by the same
    homogeneity argument (degree k(k-1)/2) is the fix, and is deliberately not
    written until there is a run that needs it.
    """
    centers = np.asarray(centers, float)
    h_raw = np.asarray(member_weights, float)
    n = len(centers)
    box_arr = _parse_box(box, centers.shape[1] if n else 1)
    if k <= 0 or n < k:
        return -np.inf, 0, None, None

    adj, _ = _build_adj(centers, lower, box_arr, species, floor)
    g = _edge_weights(centers, adj, box_arr, rc, beta, species, table)

    c = float(h_raw.max())
    if c <= 0.0:
        return -np.inf, 0, None, None
    h = h_raw / c

    # The vertex order is free -- ownership by lowest-ordered member is valid
    # under any permutation -- so this reuses the ordering
    # `_kClist_count_and_find` measured as fastest, which keeps the two kernels
    # comparable rather than merely both correct.
    rank = _degree_rank(adj, n) if k == 4 else _degeneracy_rank(adj, n)
    order = np.argsort(rank)
    g_o = g[np.ix_(order, order)]
    h_o = h[order]

    weight = _wsum(g_o, h_o, k)
    if not weight > 0.0:
        return -np.inf, 0, None, None
    log_w = float(np.log(weight) + k * np.log(c))

    if rng is None:
        return log_w, 1, None, None

    local = _wsample(g_o, h_o, k, rng)
    if local is None:
        return -np.inf, 0, None, None
    rows = sorted(int(order[i]) for i in local)
    return log_w, 1, centers[rows], rows


class CavitySweep:
    """The ladder engines' sweep interface, over a CUBE cavity instead of a rod.

    WHY THIS UNLOCKS THE LADDER ENGINES FOR `branch`. `glauber_dynamics` and
    `suwa_todo_dynamics` need exactly three things from a sweep -- the pool size
    `n`, `log_weight(k)` for k = 0..kmax, and `sample(k, rng)` -- and
    `sampling.channel.window_ladder_logw` is the only place the ladder itself is
    written. Nothing in that derivation is one-dimensional. It needs:

        * a window strictly inside the box, so the exterior is separable and can
          be frozen. A cube cavity of side <= 3 in a box of 20 qualifies; the
          CYCLIC rod does not, and its refusal stands and is geometric.
        * a pool of exactly M rows with inadmissible trials KEPT at weight zero,
          because the (M-k)! factor counts orderings of an M-element pool.
        * W_k = the summed weight over non-overlapping k-subsets of that pool.

    Only the third was ever the obstacle, and only because W_k was reached by
    enumerating subsets. `sim_setup`'s refusal said `branch` "has no 1D chain
    structure and no transfer matrix", which is true and turns out not to be the
    requirement: a transfer matrix is ONE way to get W_k, and on a cube it is the
    expensive one -- with rc >= the cavity diagonal the DP's eviction window is
    the whole pool and its state count is 2^M. `_wsum` gets the same number by
    clique algebra in O(M^(k-2)), so the ladder is computable on a geometry the
    transfer matrix cannot afford. The rod's advantage was never exactness.

    WHAT IS THE SAME AS THE CHANNEL SWEEPS, deliberately, because the engine
    asserts on all of it: `log_weight(0) == 0.0` (the empty set weighs one),
    `k > n` is -inf rather than an error, and `sample` returns POOL ROW indices.

    THE TWO PATHS, matching `ChannelSweepHardCore` / `ChannelSweepLJ` exactly:

        hard core  no member weights; the sweep counts ADMISSIBLE k-cliques and
                   the activity (z V_W)^k rides in the ladder's `log_extra`.
                   That is `_wsum` at h = admissible, g = adj -- literally the
                   hard-core clique counter, which is why `_count_from_adj`
                   reproduces it.
        LJ         h = the member weights, which already carry z * V_W / p_s and
                   exp(-beta u_ext); g = adj * exp(-beta u_ij). ACTIVITY_TERM is
                   1. `admissible` is not taken, because an inadmissible row is
                   expressed as a zero member weight -- the same convention the
                   LJ channel sweep uses.

    ORDERING. The rank permutation is applied ONCE here and `sample` returns
    indices mapped back through it, so the weight and the draw cannot disagree
    about which pool row is which -- the failure this class is most exposed to,
    since `_wsum` and `_wsample` are two traversals of the same tree.
    """

    def __init__(
        self,
        coords: FloatArray,
        member_weights: FloatArray | None,
        kmax: int,
        lower: float,
        box: FloatArray,
        rc: float,
        beta: float,
        *,
        ljbranch: bool,
        admissible: BoolArray | None = None,
        species: IntArray | None = None,
        floor: FloatArray | None = None,
        table: PairTable | None = None,
    ) -> None:
        coords = np.asarray(coords, float)
        self.n = len(coords)
        self.kmax = int(kmax)
        adj, _ = _build_adj(coords, lower, box, species, floor)
        if ljbranch:
            if member_weights is None:
                raise ValueError(
                    "the LJ cavity sweep needs member weights: the activity and "
                    "the one-body energy ride in them (section 1.1), and the "
                    "ladder's ACTIVITY_TERM is 1 on this path"
                )
            h = np.asarray(member_weights, float)
            g = _edge_weights(coords, adj, box, rc, beta, species, table)
        else:
            # The hard-core sweep takes no weights. An inadmissible trial stays
            # in the pool at weight zero rather than being dropped: |pool| == M
            # is what the (M-k)! factor counts.
            h = (
                np.ones(self.n)
                if admissible is None
                else np.asarray(admissible, bool).astype(float)
            )
            g = adj.astype(float)
        # ONE rescale, tracked exactly. W_k is homogeneous of degree k in h, so
        # dividing by the largest weight and adding k*log(c) back is not an
        # approximation -- it is what replaces the log-space accumulation the
        # enumerating path needed because member weights span decades.
        c = float(h.max()) if self.n else 0.0
        self._log_c = -np.inf if c <= 0.0 else float(np.log(c))
        if self.n and c > 0.0:
            rank = _degeneracy_rank(adj, self.n)
            self._order = np.argsort(rank)
            self._g = g[np.ix_(self._order, self._order)]
            self._h = (h / c)[self._order]
        else:
            self._order = np.arange(self.n)
            self._g = g
            self._h = h
        self._cache: dict[int, float] = {}

    def log_weight(self, k: int) -> float:
        """log W_k, cached: `window_ladder_logw` walks every rung 0..kmax."""
        k = int(k)
        if k < 0 or k > self.n or self._log_c == -np.inf:
            return -np.inf
        if k == 0:
            return 0.0
        if not self._cache and self.kmax > 0:
            # THE WHOLE LADDER IN ONE DESCENT. `window_ladder_logw` asks for every
            # rung 0..kmax in turn, and `_wsum` per rung repeats the lower rungs'
            # recursion inside each higher one. Same numbers, bit for bit.
            for j, w in enumerate(_wladder(self._g, self._h, min(self.kmax, self.n))):
                if j:
                    self._cache[j] = (
                        float(np.log(w) + j * self._log_c) if w > 0.0 else -np.inf
                    )
        if k not in self._cache:
            w = _wsum(self._g, self._h, k)
            self._cache[k] = float(np.log(w) + k * self._log_c) if w > 0.0 else -np.inf
        return self._cache[k]

    def sample(self, k: int, rng: _RNG) -> tuple[int, ...] | None:
        """A k-clique drawn proportional to its weight, as POOL ROW indices."""
        k = int(k)
        if k == 0:
            return ()
        if k > self.n or self._log_c == -np.inf:
            return None
        local = _wsample(self._g, self._h, k, rng)
        if local is None:
            return None
        return tuple(sorted(int(self._order[i]) for i in local))
