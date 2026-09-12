"""Flat-histogram biasing in N, and the transition-matrix solve for ln Pi(N).

Two objects that are usually confused and are kept apart here because the whole
method claim of the design plan turns on the difference:

  the BIAS      eta[N], added to the log weight of a macrostate so the chain
                visits the whole N range instead of camping in the mode. It
                changes the chain and nothing else.
  the ESTIMATOR C[i, j], the collection matrix. Rows are accumulated with
                UNBIASED transition probabilities, so ln Pi comes out of it
                free of whatever eta was doing at the time. That is Errington's
                central trick and it is why the bias can be refreshed mid-run
                without discarding data.

REFERENCES, and what each one is load-bearing for:

  Errington, J. Chem. Phys. 118, 9915 (2003)
    GC-TMMC. Settings this module's defaults follow: the collection matrix is
    incremented with the unbiased acceptance for EVERY proposal including
    rejected ones, and eq. 11 -- the sequential ln Pi recursion -- requires the
    transition matrix to be TRI-BANDED. His words: restricting transitions to
    neighbouring macrostates "limits the macrostate transition probability
    matrix to three bands, which enables one to obtain a unique solution";
    without it "this is an overspecified problem and a minimization technique
    needs to be implemented."
  Siderius & Shen, J. Phys. Chem. C 117, 5861 (2013)
    The same machinery on pores, plus the overlapping-subdomain parallel merge
    (exact for transition-matrix data, ill-defined for visited-states data).

SO THERE ARE TWO SOLVES, and which one applies is a property of the DATA:

  `ln_pi_three_band`  Errington eq. 11 verbatim. Only valid when C has no
                      entries outside |i - j| <= 1; asserts that rather than
                      silently ignoring the far bands.
  `ln_pi_multiband`   the overspecified case. Every ordered pair (i, j) with
                      traffic BOTH ways contributes one equation
                      x_j - x_i = ln[P(i->j) / P(j->i)], weighted by how much
                      traffic it saw, and the system is solved by weighted
                      least squares with the gauge x_0 = 0. The window heat
                      bath changes N by up to M in one step, so this is the
                      solve G9 needs and the three-band recursion does not
                      apply to it at all.
  `ln_pi_eigenvector` the same data through a different route: the stationary
                      left eigenvector of the row-normalised C. Independent of
                      the least-squares residual weighting, so agreement
                      between the two is a real check on the solve rather than
                      on the arithmetic.

WHY THE BIAS OBJECT LIVES AT L5. It is proposal machinery: `eta` multiplies the
weight of a state in the sampler and is divided back out (or, in the Glauber
case, simply not used) by the estimator. The engine reads it; nothing below the
engine does. The solve rides along in the same module because it consumes the
same array and has no other home -- it is not analysis over a finished chain,
it is what the chain reads back to refresh its own bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._types import FloatArray

_NINF = -math.inf


def _row_normalise(C: FloatArray) -> FloatArray:
    """C with every row scaled to sum to 1; all-zero rows left as zeros.

    An unvisited macrostate has an all-zero row, and dividing it by its sum
    would put NaN into the solve at a state that carries no information. Zeros
    are the honest value: no transition probability was measured.
    """
    A = np.asarray(C, dtype=np.float64)
    s = A.sum(axis=1)
    out = np.zeros_like(A)
    nz = s > 0.0
    out[nz] = A[nz] / s[nz, None]
    return out


def ln_pi_three_band(C: FloatArray) -> FloatArray:
    """Errington eq. 11: the sequential recursion, for a TRI-BANDED C.

    ln Pi(n+1) - ln Pi(n) = ln[ P(n -> n+1) / P(n+1 -> n) ], walked upward from
    ln Pi(0) = 0. Unique because the three-band restriction makes the system
    exactly determined; that is the whole reason single-particle GC-TMMC
    restricts its exchange moves to dN = 1.

    Raises on a matrix with traffic outside the three bands rather than dropping
    it: a dN > 1 move set makes this recursion the WRONG solve, not an
    approximate one, and silently ignoring the far bands would return a
    plausible curve with no relation to the data.
    """
    A = np.asarray(C, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"collection matrix must be square, got {A.shape}")
    off = A.copy()
    for d in (-1, 0, 1):
        idx = np.arange(max(0, -d), min(n, n - d))
        off[idx, idx + d] = 0.0
    if float(np.abs(off).max(initial=0.0)) > 0.0:
        bad = np.argwhere(np.abs(off) > 0.0)[:5].tolist()
        raise ValueError(
            "ln_pi_three_band was given a collection matrix with traffic "
            f"outside |i-j| <= 1 (e.g. {bad}). Errington eq. 11 is a "
            "three-band recursion and does not apply: use ln_pi_multiband, "
            "which solves the overspecified system instead."
        )
    P = _row_normalise(A)
    ln_pi = np.zeros(n)
    for i in range(n - 1):
        up, dn = P[i, i + 1], P[i + 1, i]
        if up > 0.0 and dn > 0.0:
            ln_pi[i + 1] = ln_pi[i] + math.log(up) - math.log(dn)
        else:
            # No two-way traffic across this edge: the ratio is unmeasured. Carry
            # the previous value and let the caller see the flat segment rather
            # than inventing a jump.
            ln_pi[i + 1] = ln_pi[i]
    return ln_pi


def ln_pi_multiband(
    C: FloatArray, *, min_count: float = 1.0
) -> tuple[FloatArray, float]:
    """Weighted least squares over EVERY two-way edge. Returns (ln Pi, residual).

    One equation per ordered pair with traffic both ways:

        x_j - x_i = ln[ P(i->j) / P(j->i) ]

    weighted by w_ij = harmonic mean of the two raw counts, which is small
    exactly where one direction is rare and the log-ratio is noisy. Gauge fixed
    by x_0 = 0 (an additive constant in ln Pi is unobservable), added as one
    hard row rather than by pinning a column, so the normal equations stay
    well conditioned when state 0 itself was never visited.

    `residual` is the weighted RMS of the equation residuals: zero for a
    consistent (e.g. three-band) system, and a direct measure of how much the
    multi-band data disagrees with ANY single ln Pi. A large residual means the
    C rows are not a reversible chain's -- the diagnostic the plan's G0.7 spike
    exists to establish.
    """
    A = np.asarray(C, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"collection matrix must be square, got {A.shape}")
    P = _row_normalise(A)
    rows: list[FloatArray] = []
    rhs: list[float] = []
    wts: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            cij, cji = A[i, j], A[j, i]
            if cij < min_count or cji < min_count:
                continue
            if P[i, j] <= 0.0 or P[j, i] <= 0.0:
                continue
            r = np.zeros(n)
            r[j] = 1.0
            r[i] = -1.0
            rows.append(r)
            rhs.append(math.log(P[i, j]) - math.log(P[j, i]))
            wts.append(2.0 * cij * cji / (cij + cji))
    # the gauge: x_0 = 0, with a weight large enough to pin it exactly
    gauge = np.zeros(n)
    gauge[0] = 1.0
    if not rows:
        return np.zeros(n), 0.0
    scale = max(1.0, max(wts))
    M = np.vstack([*rows, gauge])
    b = np.array([*rhs, 0.0])
    w = np.sqrt(np.array([*wts, scale * 1e6]))
    sol, *_ = np.linalg.lstsq(M * w[:, None], b * w, rcond=None)
    res = (M[:-1] @ sol) - b[:-1]
    wr = np.array(wts)
    rms = float(np.sqrt((wr * res * res).sum() / wr.sum())) if len(res) else 0.0
    return np.asarray(sol - sol[0], dtype=np.float64), rms


def ln_pi_eigenvector(C: FloatArray) -> FloatArray:
    """ln Pi from the stationary LEFT eigenvector of the row-normalised C.

    Pi P = Pi, solved by the power method on the visited block. A second,
    structurally different route to the same object as `ln_pi_multiband`: it
    uses no log-ratios and no weighting, so the two agreeing is evidence about
    the solve rather than about one implementation.

    Unvisited macrostates come back as -inf, which is the truth about them.
    """
    A = np.asarray(C, dtype=np.float64)
    n = A.shape[0]
    seen = A.sum(axis=1) > 0.0
    out = np.full(n, _NINF)
    if not seen.any():
        return out
    P = _row_normalise(A)[np.ix_(seen, seen)]
    # renormalise the restricted block: probability leaving the visited set is
    # reflected back, which is the standard treatment of a truncated window
    s = P.sum(axis=1)
    s[s <= 0.0] = 1.0
    P = P / s[:, None]
    v = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(20000):
        nv = v @ P
        nv = nv / nv.sum()
        if float(np.abs(nv - v).max()) < 1e-15:
            v = nv
            break
        v = nv
    with np.errstate(divide="ignore"):
        lv = np.log(v)
    out[seen] = lv - lv[0]
    return out


@dataclass
class FlatHistogram:
    """The bias array, the collection matrix and the visit histogram in N.

    `n_min` .. `n_max` INCLUSIVE, so the arrays have `n_max - n_min + 1` rows
    and macrostate N sits at row `N - n_min`. N_min = 0 is required for the
    pressure reference and for the grand-free-energy relation (Errington 2003),
    so it is the default and a nonzero floor has to be asked for.

    THE BIAS IS ADDITIVE IN LOG WEIGHT. `log_bias(N)` returns eta[N]; a sampler
    adds `eta[N_new] - eta[N_old]` to its log acceptance, or -- for the heat
    bath, which draws from an exact conditional rather than testing -- adds the
    whole eta slice to the ladder before the categorical draw. Out-of-range N
    returns -inf, which refuses the move rather than silently sampling a
    macrostate the histogram cannot record.

    THE MATRIX IS UNBIASED. `record` takes the UNBIASED log acceptance and
    stores min(1, exp(.)) in the off-diagonal and the complement on the
    diagonal. Passing the biased value here is the one mistake that makes the
    whole method wrong while still producing a smooth curve, so the argument is
    named `log_acc_unbiased` and nothing else.
    """

    n_min: int = 0
    n_max: int = 100
    eta: FloatArray = field(default_factory=lambda: np.zeros(0))
    C: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    visits: FloatArray = field(default_factory=lambda: np.zeros(0))
    #: transitions offered outside [n_min, n_max]; a nonzero count here means the
    #: window is too narrow and the measured ln Pi is truncated, not converged.
    out_of_range: int = 0

    def __post_init__(self) -> None:
        """Size the three arrays from the window if the caller did not."""
        k = self.n_states
        if self.eta.size == 0:
            self.eta = np.zeros(k)
        if self.C.size == 0:
            self.C = np.zeros((k, k))
        if self.visits.size == 0:
            self.visits = np.zeros(k)
        for name, arr, want in (
            ("eta", self.eta, (k,)),
            ("C", self.C, (k, k)),
            ("visits", self.visits, (k,)),
        ):
            if arr.shape != want:
                raise ValueError(f"{name} has shape {arr.shape}, expected {want}")

    @property
    def n_states(self) -> int:
        """Number of macrostates in the window, inclusive of both ends."""
        return self.n_max - self.n_min + 1

    def row(self, n: int) -> int:
        """Row index of macrostate `n`, or -1 when it is outside the window."""
        i = n - self.n_min
        return i if 0 <= i < self.n_states else -1

    def log_bias(self, n: int) -> float:
        """eta[n]; -inf outside the window, which refuses the transition."""
        i = self.row(n)
        return _NINF if i < 0 else float(self.eta[i])

    def bias_slice(self, n_lo: int, count: int) -> FloatArray:
        """eta for macrostates n_lo .. n_lo+count-1, with -inf off the ends.

        The Glauber engine needs the whole slice at once (one entry per rung of its
        ladder), and getting it as an array keeps the -inf padding in one place
        instead of at every call site.
        """
        out = np.full(count, _NINF)
        for k in range(count):
            out[k] = self.log_bias(n_lo + k)
        return out

    def visit(self, n: int) -> None:
        """Count one sampled configuration at macrostate `n`."""
        i = self.row(n)
        if i < 0:
            self.out_of_range += 1
            return
        self.visits[i] += 1.0

    def record(self, n_from: int, n_to: int, log_acc_unbiased: float) -> None:
        """One proposal's contribution to the collection matrix.

        `log_acc_unbiased` is log of the UNBIASED Metropolis ratio for
        n_from -> n_to. Both the transition and its complement are stored, and
        both for every proposal whether it was accepted or not: that is what
        makes C an estimate of the transition MATRIX rather than of the realised
        chain, and it is why C can be merged across runs exactly while a visit
        histogram cannot.
        """
        i, j = self.row(n_from), self.row(n_to)
        if i < 0:
            self.out_of_range += 1
            return
        p = 1.0 if log_acc_unbiased >= 0.0 else math.exp(log_acc_unbiased)
        if j < 0:
            # the proposal left the window: it still failed to move the chain
            # within it, so the retained probability is the honest record
            self.out_of_range += 1
            self.C[i, i] += 1.0
            return
        self.C[i, j] += p
        self.C[i, i] += 1.0 - p

    def record_exact_row(self, n_from: int, n_lo: int, probs: FloatArray) -> None:
        """A whole row at once: P(n_from -> n_lo + k) = probs[k], exactly.

        This is what the Glauber engine can do and single-particle TMMC
        cannot. TMMC increments C with the acceptance probability of the ONE
        move it proposed; the Glauber engine knows the exact conditional probability
        of every reachable macrostate given the pool, so one step fills a whole
        row. Probabilities that fall outside the window are folded onto the
        diagonal, which keeps the row normalised and records the truncation in
        `out_of_range`.
        """
        i = self.row(n_from)
        if i < 0:
            self.out_of_range += 1
            return
        p = np.asarray(probs, dtype=np.float64)
        lost = 0.0
        for k in range(p.size):
            j = self.row(n_lo + k)
            if j < 0:
                lost += float(p[k])
                continue
            self.C[i, j] += float(p[k])
        if lost > 0.0:
            self.C[i, i] += lost
            self.out_of_range += 1

    # -- the solve -------------------------------------------------------
    def is_three_band(self) -> bool:
        """Does C have traffic only at |i - j| <= 1?

        Decides which solve applies. Asked rather than assumed because the same
        object serves both engines: single-particle TMMC fills three bands, the
        Glauber engine fills the whole row.
        """
        A = self.C
        n = A.shape[0]
        off = A.copy()
        for d in (-1, 0, 1):
            idx = np.arange(max(0, -d), min(n, n - d))
            off[idx, idx + d] = 0.0
        return bool(float(np.abs(off).max(initial=0.0)) == 0.0)

    def ln_pi(self) -> FloatArray:
        """ln Pi(N) over the window, gauged to ln Pi(n_min) = 0.

        Picks the solve from the data: the three-band recursion when C is
        tri-banded (it is exact and unique there), the weighted least-squares
        multi-band solve otherwise.
        """
        if self.is_three_band():
            return ln_pi_three_band(self.C)
        return ln_pi_multiband(self.C)[0]

    def refresh(self, *, damping: float = 1.0) -> FloatArray:
        """Set eta = -ln Pi from the current C and return the new eta.

        A flat histogram in N is exactly the statement that the sampled weight
        Pi(N) exp(eta[N]) is constant, so eta = -ln Pi. `damping` < 1 mixes with
        the previous eta, which is worth having early in a run when C is thin.

        THE COLLECTION MATRIX IS NOT RESET. Refreshing is a change of chain, not
        of estimator: C's rows are unbiased transition probabilities, so the
        rows collected under the old eta and the new one estimate the same
        matrix and merging them is exact. Discarding C at each refresh is the
        mistake this docstring exists to prevent -- it is the whole reason
        Errington's scheme can refresh often and cheaply.
        """
        lp = self.ln_pi()
        target = -np.where(np.isfinite(lp), lp, 0.0)
        self.eta = (1.0 - damping) * self.eta + damping * target
        # an additive constant in eta is unobservable; anchor it so the numbers
        # in a log stay readable across refreshes
        self.eta = self.eta - self.eta.min()
        return self.eta

    def flatness(self) -> float:
        """max(visits) / min(visits) over VISITED macrostates; inf if any is 0.

        The G9.2 criterion in one number. Restricted to visited states because a
        window wider than the chain ever reached would report inf forever and
        say nothing about the states that were sampled.
        """
        v = self.visits
        seen = v > 0.0
        if not seen.any():
            return float("inf")
        if int(seen.sum()) < self.n_states:
            return float("inf")
        return float(v.max() / v.min())

    def drop_edge_rows(self, k: int) -> FlatHistogram:
        """Zero the outermost `k` ROWS of the collection matrix, in place.

        WHY ROWS AND NOT COLUMNS, and why at all. A row of C is the average
        transition probability measured WHILE THE CHAIN SAT at that macrostate,
        so its quality depends on how often the chain sat there. The outermost
        macrostates of a bounded window are badly sampled, and the two kernels
        fail there in opposite ways: a single-particle chain piles UP against
        the wall (every move out is refused, so it stays), while a multi-band
        kernel JUMPS OVER them and reaches them only when a window draws exactly
        the right k. Either way the row is thin and its configuration
        distribution at fixed N is not the equilibrium one.

        The COLUMNS are unaffected: an entry C[i, j] is an exact conditional
        probability computed from a pool at macrostate i, and j being near an
        edge does not make it worse.

        This is what an overlap is FOR. Discarding each subdomain's edge rows
        leaves those macrostates to be determined by the neighbouring
        subdomain, where they sit in the interior -- so the overlap has to be
        wider than the edge that is discarded, and (for the Glauber engine)
        at least kmax wide, since kmax is how far a single step can jump.

        Rows are ZEROED rather than deleted so the grid is unchanged and `merge`
        still applies.
        """
        if k <= 0:
            return self
        if 2 * k >= self.n_states:
            raise ValueError(
                f"dropping {k} rows at each edge of a {self.n_states}-macrostate "
                "window leaves nothing; the window is too narrow to trim"
            )
        self.C[:k, :] = 0.0
        self.C[self.n_states - k :, :] = 0.0
        return self

    def merge(self, other: FlatHistogram) -> FlatHistogram:
        """Add another window's C and visits into this one (Siderius & Shen).

        Exact for the collection matrix, which is the property that makes the
        overlapping-subdomain parallel strategy work: transition-matrix data
        merges by addition, visited-states data does not merge at all. Windows
        must agree on (n_min, n_max); overlapping subdomains are merged by
        embedding each into the full window first.
        """
        if (self.n_min, self.n_max) != (other.n_min, other.n_max):
            raise ValueError(
                f"cannot merge windows [{self.n_min},{self.n_max}] and "
                f"[{other.n_min},{other.n_max}]: embed both in the full window "
                "first. Transition-matrix data merges by addition only over the "
                "same macrostate grid."
            )
        self.C = self.C + other.C
        self.visits = self.visits + other.visits
        self.out_of_range += other.out_of_range
        return self


def embed(fh: FlatHistogram, n_min: int, n_max: int) -> FlatHistogram:
    """Copy `fh` into a wider window, so subdomains can be merged by addition.

    The parallel strategy in G10 runs overlapping N subdomains independently;
    each comes back on its own grid, and `merge` refuses mismatched grids on
    purpose. This is the one legal way to make them match.
    """
    if n_min > fh.n_min or n_max < fh.n_max:
        raise ValueError(
            f"embed target [{n_min},{n_max}] does not contain [{fh.n_min},{fh.n_max}]"
        )
    out = FlatHistogram(n_min=n_min, n_max=n_max)
    lo = fh.n_min - n_min
    k = fh.n_states
    out.C[lo : lo + k, lo : lo + k] = fh.C
    out.visits[lo : lo + k] = fh.visits
    out.eta[lo : lo + k] = fh.eta
    out.out_of_range = fh.out_of_range
    return out
