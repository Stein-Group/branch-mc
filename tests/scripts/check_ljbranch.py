"""LJ-branch fix: weights are reused, not recomputed, and there is no upper
cutoff on clique adjacency.

These tests are self-contained: they check the new path against brute force and
against the engine's own group-energy routine, not against the original monolith
(whose LJ-branch path is the thing being fixed). Four checks, all fast:

  A  reuse == recompute      the carried single-particle weight equals the
                             member-vs-system part of _lj_set_energy, and the
                             intra factor equals its intra part, so a clique's
                             reused weight equals the full recompute exactly.
  B  sum vs brute force      ljbranch_weight_sum over the non-overlap graph (no
                             upper bound) equals a direct sum over every
                             non-overlapping k-subset.
  C  sampling distribution   the reservoir draw picks each clique with frequency
                             proportional to its weight.
  D  spanning cliques        two far-apart non-overlapping centres DO form a
                             clique (the old band graph wrongly excluded them).
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scipy.spatial import cKDTree

from gcmc.MC_engine import _lj_set_energy, _lj_singleparticle_weight
from gcmc.potentials import make_potential
from gcmc.samplers import branch

LOWER = 0.9  # the auxiliary exclusion floor
RC = 2.5  # LJ cutoff
BETA = 1.0 / 0.85  # T* = 0.85
NN = 256  # high neighbour cap, as in production
fail = 0


def carried_weights(pool, rest, box):
    """The single-particle weights the neighbour build would store: each pool
    point's LJ Boltzmann factor against `rest` within RC. Mirrors the engine's
    find_neighbors -> _lj_singleparticle_weight path."""
    pot = make_potential("lj", beta=BETA, upper_cutoff=RC, diameter=1.0)
    if len(rest) == 0:
        return np.ones(len(pool))
    tree = cKDTree(rest, boxsize=box)
    dists, idxs = tree.query(pool, k=NN, distance_upper_bound=RC)
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    return _lj_singleparticle_weight(dists, idxs, pot, None)


def min_image_pdist(members, box):
    out = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            d = members[i] - members[j]
            d -= np.round(d / box) * box
            out.append(np.sqrt((d * d).sum()))
    return np.array(out) if out else np.array([np.inf])


def brute_force_logW(pool, weights, rest, k, box):
    """Direct sum over every non-overlapping k-subset of exp(-beta U_group),
    U_group = members-vs-rest + intra. Returns (logW, {idx_tuple: weight})."""
    terms = {}
    for combo in itertools.combinations(range(len(pool)), k):
        members = pool[list(combo)]
        if np.any(min_image_pdist(members, box) < LOWER - 1e-9):
            continue  # overlapping subset, not a clique
        u = _lj_set_energy(members, rest, box, RC)
        terms[combo] = float(np.exp(-BETA * u))
    if not terms:
        return -np.inf, terms
    return float(np.log(sum(terms.values()))), terms


# --------------------------------------------------------------------------- A
print("A  reuse == recompute (single-particle, intra, and composed clique)")
rg = np.random.default_rng(0)
maxerr = 0.0
for _trial in range(8):
    box = 8.0
    rest = rg.uniform(0, box, size=(rg.integers(20, 60), 3))
    pool = rg.uniform(0, box, size=(rg.integers(8, 14), 3))
    # drop pool points that overlap rest (those never survive the build)
    tree = cKDTree(rest, boxsize=box)
    near = tree.query_ball_point(pool, LOWER - 1e-9)
    pool = pool[[len(n) == 0 for n in near]]
    if len(pool) < 4:
        continue
    w = carried_weights(pool, rest, box)
    # (a) carried single-particle weight == exp(-beta * member-vs-rest)
    for i in range(len(pool)):
        truth = np.exp(-BETA * _lj_set_energy(pool[i : i + 1], rest, box, RC))
        maxerr = max(maxerr, abs(w[i] - truth))
    # (b) intra factor == exp(-beta * intra energy) and (c) composed == full
    for k in (2, 3):
        for combo in itertools.combinations(range(len(pool)), k):
            members = pool[list(combo)]
            if np.any(min_image_pdist(members, box) < LOWER - 1e-9):
                continue
            intra = branch._intra_lj_factor(members, np.full(3, box), RC, BETA)
            intra_truth = np.exp(
                -BETA * _lj_set_energy(members, np.empty((0, 3)), box, RC)
            )
            composed = float(np.prod(w[list(combo)])) * intra
            full = float(np.exp(-BETA * _lj_set_energy(members, rest, box, RC)))
            maxerr = max(maxerr, abs(intra - intra_truth), abs(composed - full))
print(f"     max abs error over all checks: {maxerr:.2e}")
if maxerr < 1e-9:
    print("     ok\n")
else:
    print("     FAIL: reuse does not reproduce recompute\n")
    fail += 1

# --------------------------------------------------------------------------- B
print("B  ljbranch_weight_sum == brute force over non-overlapping subsets")
rg = np.random.default_rng(1)
maxrel = 0.0
for _trial in range(8):
    box = 8.0
    rest = rg.uniform(0, box, size=(rg.integers(15, 40), 3))
    pool = rg.uniform(0, box, size=(rg.integers(12, 16), 3))
    tree = cKDTree(rest, boxsize=box)
    near = tree.query_ball_point(pool, LOWER - 1e-9)
    pool = pool[[len(n) == 0 for n in near]]
    if len(pool) < 5:
        continue
    w = carried_weights(pool, rest, box)
    for k in (1, 2, 3, 4, 5):  # past the old depth-3 cap
        logW, terms = brute_force_logW(pool, w, rest, k, box)
        logW2, count, _, _ = branch.ljbranch_weight_sum(
            pool, w, k, LOWER, box, RC, BETA, rng=None
        )
        if not np.isfinite(logW) and not np.isfinite(logW2):
            continue
        rel = abs(np.exp(logW2) - np.exp(logW)) / max(np.exp(logW), 1e-300)
        maxrel = max(maxrel, rel)
        if count != len(terms):
            print(f"     FAIL: clique count {count} != brute {len(terms)} (k={k})")
            fail += 1
print(f"     max relative error in W: {maxrel:.2e}")
if maxrel < 1e-9:
    print("     ok\n")
else:
    print("     FAIL: sum disagrees with brute force\n")
    fail += 1

# --------------------------------------------------------------------------- C
print("C  reservoir draw frequency proportional to clique weight")
rg = np.random.default_rng(2)
box = 6.0
rest = rg.uniform(0, box, size=(20, 3))
pool = rg.uniform(0, box, size=(11, 3))
tree = cKDTree(rest, boxsize=box)
near = tree.query_ball_point(pool, LOWER - 1e-9)
pool = pool[[len(n) == 0 for n in near]]
w = carried_weights(pool, rest, box)
K = 3
_, terms = brute_force_logW(pool, w, rest, K, box)
total = sum(terms.values())
exact = {c: v / total for c, v in terms.items()}
# index lookup so a returned coord-set maps back to its clique tuple
key = {tuple(np.round(pool[i], 9)): i for i in range(len(pool))}
draws = 30000
draw_rng = np.random.default_rng(7)
tally = dict.fromkeys(terms, 0)
for _ in range(draws):
    _, _, chosen, _ = branch.ljbranch_weight_sum(
        pool, w, K, LOWER, box, RC, BETA, rng=draw_rng
    )
    combo = tuple(sorted(key[tuple(np.round(p, 9))] for p in chosen))
    tally[combo] += 1
tv = 0.5 * sum(abs(tally[c] / draws - exact[c]) for c in terms)
print(f"     {len(terms)} cliques, {draws} draws, total-variation dist: {tv:.4f}")
if tv < 0.02:
    print("     ok\n")
else:
    print("     FAIL: sampling not proportional to weight\n")
    fail += 1

# --------------------------------------------------------------------------- D
print("D  far-apart non-overlapping centres form a clique (no upper bound)")
box = 50.0
# two points ~25 apart: overlapping under no rule, far beyond any cavity span
pool = np.array([[1.0, 1.0, 1.0], [26.0, 1.0, 1.0]])
w = np.array([1.0, 1.0])  # rest empty -> unit weights
logW, count, _, _ = branch.ljbranch_weight_sum(
    pool, w, 2, LOWER, box, RC, BETA, rng=None
)
# one 2-clique exists; both far beyond RC so intra factor is exactly 1 -> W = 1
ok = (count == 1) and abs(np.exp(logW) - 1.0) < 1e-12
print(f"     2-cliques found: {count}, W: {np.exp(logW):.6f}")
if ok:
    print("     ok\n")
else:
    print("     FAIL: spanning clique missing or mis-weighted\n")
    fail += 1

# ---------------------------------------------------------------------------
if fail == 0:
    print("ALL LJ-BRANCH FIX TESTS PASSED")
else:
    print(f"{fail} LJ-BRANCH CHECK(S) FAILED")
    raise SystemExit(1)
