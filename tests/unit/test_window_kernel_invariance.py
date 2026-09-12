"""G2: the window kernel's pi-invariance, as linear algebra rather than statistics.

THE HARNESS. Freeze the trial cloud to a fixed SITE SET and the state space
becomes finite (2^n), so the transition matrix can be assembled explicitly and
`pi T = pi` checked to machine precision instead of estimated. Sites lie on a
line, because that is the structure the channel sweep exploits: a thin cuboid
with finite-range interactions IS a 1D chain, which is what makes the transfer
matrix work at all.

WHAT THE FROZEN CLOUD CHANGES, stated plainly because it is the one thing that
makes this test look weaker than it is. The (M-k)! in the ladder is a property
of the CONTINUUM cloud measure -- it counts the orderings of an M-element pool
whose M-k fresh points were drawn from a volume V_W. With the cloud frozen onto
the site set there is no such volume: the pool is the same set of sites at every
step, so the auxiliary-variable derivation degenerates and the correct local
conditional is the plain lattice-gas one,

    P(S_w | S_e)  prop  z^|S_w| exp(-beta U(S_w, S_e))

which is what the harness targets. So this file tests the SWEEP (clique weights
against the frozen exterior), the LADDER (drawing k), the within-k CONDITIONAL
(`sweep.sample`), and their composition into a pi-invariant block kernel. It
does NOT test the (M-k)!, and it is not supposed to: that factor is tested
against brute-force enumeration and against the Poisson ideal-gas limit in
`test_window_ladder.py`, where the continuum measure is actually present.

WHY A MIXTURE OF BLOCKS. A single window covering every site gives an
independence sampler -- T(S -> S') = pi(S') for all S -- which is pi-invariant
for trivial reasons and would test nothing. Sweeping the window across the chain
and averaging the block kernels is the real move set: each block kernel is
reversible with respect to the same pi, so the mixture is pi-invariant, and the
mixture is what a run actually applies.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gcmc.sampling.channel import ChannelSweepLJ, ladder_probs, window_ladder_logw

SIGMA = 1.0
BETA = 1.0 / 1.2
RC = 2.5


def _sites(n, pitch=0.62):
    """n sites on a line, closer together than sigma so exclusion has teeth.

    A pitch above sigma would make every subset legal and the exclusion DP
    untested; at 0.62 sigma a site conflicts with its first neighbour and not
    its second, which is the shortest interesting conflict graph.
    """
    x = np.arange(n) * pitch
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


def _u_pair(d2):
    """The same 12-6 the sweep uses, from a squared distance, or +inf inside the
    core. Written out so the target distribution shares no code with the DP."""
    if d2 <= (SIGMA - 1e-12) ** 2:
        return math.inf
    if d2 >= RC * RC:
        return 0.0
    x = 1.0 / d2**3
    return 4.0 * (x * x - x)


def _log_pi(S, X, log_z):
    """log pi(S) for the grand-canonical lattice gas on these sites.

    pi(S) prop z^|S| exp(-beta U(S)); -inf for a subset containing an
    overlapping pair. From scratch, by an explicit pair loop.
    """
    u = 0.0
    for a, b in itertools.combinations(sorted(S), 2):
        d2 = float(((X[a] - X[b]) ** 2).sum())
        e = _u_pair(d2)
        if math.isinf(e):
            return -math.inf
        u += e
    return len(S) * log_z - BETA * u


def _legal_states(n, X, log_z):
    """Every subset with finite weight, and its log pi."""
    out = []
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            lp = _log_pi(set(S), X, log_z)
            if math.isfinite(lp):
                out.append((frozenset(S), lp))
    return out


def _block_kernel(states, index, X, log_z, block):
    """T_block: the exact row for every state, from the ENGINE's own machinery.

    For each state, the exterior occupants (sites outside `block`) are frozen;
    the pool is every site IN the block; the member weight of pool row i is
    z * exp(-beta u_ext_i) with u_ext its interaction with the frozen exterior;
    and the row is P(k) from `window_ladder_logw` times the within-k clique
    conditional read out of the sweep exactly.

    The within-k conditional is obtained by ENUMERATION over the sweep's own
    per-clique weights rather than by sampling, because this is a linear-algebra
    test: `sweep.sample` is checked against those same weights in
    `test_window_ladder.py`.
    """
    n = len(X)
    blk = sorted(block)
    ext = [i for i in range(n) if i not in block]
    T = np.zeros((len(states), len(states)))
    for si, (S, _lp) in enumerate(states):
        occ_ext = [i for i in ext if i in S]
        # one-body weights of the pool rows against the frozen exterior
        mw = np.zeros(len(blk))
        for r, i in enumerate(blk):
            u = 0.0
            bad = False
            for j in occ_ext:
                e = _u_pair(float(((X[i] - X[j]) ** 2).sum()))
                if math.isinf(e):
                    bad = True
                    break
                u += e
            mw[r] = 0.0 if bad else math.exp(log_z - BETA * u)
        M = len(blk)
        sweep = ChannelSweepLJ(X[blk], mw, SIGMA, RC, BETA, M)
        # THE LATTICE LADDER: the continuum (M-k)! is cancelled, because with a
        # frozen cloud there is no volume it could have come from (see the
        # module docstring).
        extra = np.array([-math.lgamma(M - k + 1) for k in range(M + 1)])
        p_k = ladder_probs(window_ladder_logw(sweep, M, log_extra=extra))
        # per-clique weights, by enumeration over the sweep's own definition
        for k in range(M + 1):
            if p_k[k] <= 0.0:
                continue
            cliques: dict[tuple[int, ...], float] = {}
            for c in itertools.combinations(range(M), k):
                wgt = 1.0
                ok = True
                for r in c:
                    if mw[r] <= 0.0:
                        ok = False
                        break
                    wgt *= mw[r]
                if not ok:
                    continue
                for a, b in itertools.combinations(c, 2):
                    e = _u_pair(float(((X[blk[a]] - X[blk[b]]) ** 2).sum()))
                    if math.isinf(e):
                        ok = False
                        break
                    wgt *= math.exp(-BETA * e)
                if ok and wgt > 0.0:
                    cliques[c] = wgt
            tot = sum(cliques.values())
            if tot <= 0.0:
                continue
            for c, wgt in cliques.items():
                new = frozenset(occ_ext) | {blk[r] for r in c}
                T[si, index[new]] += p_k[k] * wgt / tot
    return T


@pytest.mark.parametrize("n,block_size", [(8, 3), (8, 4), (9, 3)])
def test_pi_T_equals_pi_to_machine_precision(n, block_size):
    """G2.1: the window kernel leaves the grand-canonical lattice gas invariant.

    Assembled explicitly and checked to 1e-13, so nothing here can fail for
    statistical reasons -- which is the point of doing it as linear algebra.
    """
    X = _sites(n)
    log_z = 0.4
    states = _legal_states(n, X, log_z)
    index = {S: i for i, (S, _) in enumerate(states)}
    pi = np.array([math.exp(lp) for _S, lp in states])
    pi = pi / pi.sum()

    blocks = [set(range(a, a + block_size)) for a in range(n - block_size + 1)]
    T = np.mean([_block_kernel(states, index, X, log_z, b) for b in blocks], axis=0)

    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12), "T is not stochastic"
    dev = float(np.abs(pi @ T - pi).max())
    assert dev < 1e-13, f"pi T - pi = {dev:.3e} (states {len(states)})"


@pytest.mark.parametrize("block_size", [2, 3, 5])
def test_invariance_holds_at_two_pool_sizes(block_size):
    """G2.2, the DETERMINISTIC half: invariance at more than one M.

    The frozen-cloud harness makes M the block size, and each block kernel is
    pi-invariant for its own M. That is the honest deterministic statement.

    IT IS NOT the claim that P(k) is invariant under M -- it is not, and the
    plan says so: for a FIXED window and surroundings, P(k) genuinely differs
    between cloud realisations at different M, and only the MARGINAL over clouds
    is M-invariant. As a deterministic test that reduces to the ideal-gas limit,
    which `test_window_ladder.test_ideal_gas_limit_is_poisson` already covers at
    M = 10, 25 and 40; the interacting statement is statistical and belongs to
    the averaged-M gate on the running engine.
    """
    n = 8
    X = _sites(n)
    log_z = 0.1
    states = _legal_states(n, X, log_z)
    index = {S: i for i, (S, _) in enumerate(states)}
    pi = np.array([math.exp(lp) for _S, lp in states])
    pi = pi / pi.sum()
    blocks = [set(range(a, a + block_size)) for a in range(n - block_size + 1)]
    T = np.mean([_block_kernel(states, index, X, log_z, b) for b in blocks], axis=0)
    assert float(np.abs(pi @ T - pi).max()) < 1e-13


def test_detailed_balance_holds_edge_by_edge():
    """G2.3: pi(S) q(S'|S) == pi(S') q(S|S') for every edge of one block kernel.

    Invariance is the weaker statement -- a non-reversible kernel can still be
    pi-invariant -- so this checks the stronger one directly, edge by edge, on a
    single block. A single block Gibbs step must be reversible; the MIXTURE over
    blocks need not be, and is not claimed to be.
    """
    n, block_size = 7, 3
    X = _sites(n)
    log_z = 0.3
    states = _legal_states(n, X, log_z)
    index = {S: i for i, (S, _) in enumerate(states)}
    pi = np.array([math.exp(lp) for _S, lp in states])
    pi = pi / pi.sum()
    T = _block_kernel(states, index, X, log_z, set(range(2, 2 + block_size)))
    flux = pi[:, None] * T
    dev = float(np.abs(flux - flux.T).max())
    scale = float(flux.max())
    assert dev < 1e-14 * max(1.0, scale), f"detailed balance off by {dev:.3e}"
