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
import numpy as np
from scipy.spatial import KDTree

from ..geometry import overlap_radius
from ..potentials.lennard_jones import lj_pair_energy


def _build_adj(centers, overlap_distance, box):
    n = len(centers)
    shifted = np.mod(centers, box)
    tree = KDTree(shifted, boxsize=box)
    pairs = tree.query_pairs(r=overlap_radius(overlap_distance), output_type="ndarray")
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    if len(pairs) > 0:
        adj[pairs[:, 0], pairs[:, 1]] = False
        adj[pairs[:, 1], pairs[:, 0]] = False
    return adj, shifted


def _degeneracy_rank(adj, n):
    """Bucket-queue degeneracy ordering."""
    nbrs = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = [len(nb) for nb in nbrs]
    max_d = max(deg) if n > 0 else 0
    buckets = [set() for _ in range(max_d + 1)]
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


def _degree_rank(adj, n):
    """Simple degree sort. One vectorized call."""
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(adj.sum(axis=1))] = np.arange(n)
    return rank


def _count_sub(A, k):
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
        fwd = np.where(A[i, i + 1:])[0] + (i + 1)
        if fwd.size >= k - 1:
            total += _count_sub(A[np.ix_(fwd, fwd)], k - 1)
    return total


def _find_sub(A, k):
    """Find one k-clique in a small adjacency. Returns index list or None."""
    d = A.shape[0]
    if d < k or k <= 0:
        return None
    if k == 1:
        return [0]
    if k == 2:
        for i in range(d - 1):
            c = np.argmax(A[i, i + 1:])
            if A[i, i + 1 + c]:
                return [i, i + 1 + c]
        return None
    if k == 3:
        Af = A.astype(np.float64) if A.dtype != np.float64 else A
        M = Af * (Af @ Af)
        for i in range(d - 1):
            row = M[i, i + 1:]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(A[i, :j] & A[j, :j])[0]
            if len(cn) == 0:
                cn = np.where(A[i, j + 1:] & A[j, j + 1:])[0]
                if len(cn) > 0:
                    return [i, j, int(cn[0]) + j + 1]
            else:
                return [int(cn[0]), i, j]
        return None
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1:])[0] + (i + 1)
        if fwd.size >= k - 1:
            inner = _find_sub(A[np.ix_(fwd, fwd)], k - 1)
            if inner is not None:
                return [i] + [int(fwd[x]) for x in inner]
    return None


def _kClist_count_and_find(adj, k):
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


def _count_from_adj(adj, k):
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
            row = M[i, i + 1:]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(adj[i, :j] & adj[j, :j])[0]
            if len(cn) > 0:
                return count, tuple(sorted((int(cn[0]), i, j)))
            cn = np.where(adj[i, j + 1:] & adj[j, j + 1:])[0]
            if len(cn) > 0:
                return count, tuple(sorted((i, j, int(cn[0]) + j + 1)))
        return count, None
    return _kClist_count_and_find(adj, k)


def _parse_box(box_lengths, dim):
    box = np.atleast_1d(np.asarray(box_lengths, dtype=np.float64))
    if box.size == 1:
        box = np.full(dim, float(box[0]))
    return box


def count_and_sample(centers, k, overlap_distance, *, box_lengths):
    """Count non-overlapping k-disk configurations and return one example.
    Adjacency means a pair is closer than `overlap_distance` (the exclusion
    floor: the hard-core diameter, or the 0.85 LJ edge).
    Returns (count, sample) with sample a sorted index tuple or None."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    box = _parse_box(box_lengths, centers.shape[1] if n > 0 else 1)
    if k <= 0 or n < k:
        return 0, None
    if k == 1:
        return n, ((0,) if n > 0 else None)
    adj, _ = _build_adj(centers, overlap_distance, box)
    return _count_from_adj(adj, k)


def count_and_sample_branch(centers, k, overlap_distance, *, box_lengths):
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


def _enumerate_cliques(adj, k):
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

    def recurse(cand, need, prefix):
        if need == 0:
            yield tuple(sorted(prefix))
            return
        if len(cand) < need:
            return
        for ii, v in enumerate(cand):
            nxt = [u for u in cand[ii + 1:] if adj[v, u]]
            if len(nxt) >= need - 1:
                yield from recurse(nxt, need - 1, prefix + [v])

    for v in range(n):
        fwd = [u for u in range(n) if adj[v, u] and rank[u] > rank[v]]
        if len(fwd) >= k - 1:
            yield from recurse(fwd, k - 1, [v])


def _intra_lj_factor(members, box, rc, beta):
    """exp(-beta * LJ energy among `members`): pairs within rc, minimum image.
    Returns 1.0 for fewer than two members (no pairs), so depth-0 weights are
    exactly the carried single-particle weight with nothing added."""
    m = len(members)
    if m < 2:
        return 1.0
    ii, jj = np.triu_indices(m, k=1)
    dvec = members[ii] - members[jj]
    dvec -= np.round(dvec / box) * box
    r = np.sqrt((dvec ** 2).sum(axis=1))
    r = r[(r < rc) & (r > 0)]
    if r.size == 0:
        return 1.0
    u = float(np.sum(lj_pair_energy(r)))
    return float(np.exp(-beta * u))


def ljbranch_weight_sum(centers, member_weights, k, lower, box, rc, beta,
                        *, rng=None):
    """Sum the Boltzmann weights of every non-overlapping k-clique.

    Adjacency is non-overlap at `lower` (the same graph the hard-core counter
    builds, via _build_adj); there is no upper cutoff. A clique's weight is the
    product of its members' carried single-particle weights (member-vs-system,
    already computed in the neighbour build) times the intra-group LJ factor
    (member pairs within rc). The sum is accumulated in log space.

    If `rng` is given, one clique is reservoir-sampled proportional to its
    weight and its member centres are returned (the forward / insertion use);
    otherwise `chosen` is None (the reverse use needs only the sum).

    Returns (logW, count, chosen_coords_or_None); (-inf, 0, None) if no clique
    exists.
    """
    centers = np.asarray(centers, float)
    member_weights = np.asarray(member_weights, float)
    box = _parse_box(box, centers.shape[1] if len(centers) else 1)
    adj, _ = _build_adj(centers, lower, box)
    logW = -np.inf
    count = 0
    chosen = None
    for clique in _enumerate_cliques(adj, k):
        idx = list(clique)
        wprod = float(np.prod(member_weights[idx]))
        if wprod <= 0.0:
            continue
        members = centers[idx]
        w = wprod * _intra_lj_factor(members, box, rc, beta)
        if w <= 0.0:
            continue
        count += 1
        logw = np.log(w)
        new_logW = np.logaddexp(logW, logw)
        # weighted reservoir draw in log space: keep this clique with
        # probability w / W_running = exp(logw - new_logW)
        if rng is not None and (logw - new_logW) >= np.log(rng.uniform(0.0, 1.0)):
            chosen = members
        logW = new_logW
    if count == 0:
        return -np.inf, 0, None
    return float(logW), count, chosen
