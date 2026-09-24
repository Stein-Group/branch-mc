"""Solving the bias exponent from the trial weights instead of reading it.

`_GAMMA_AUX_3D` is nine hand-fitted numbers keyed by Rosenbluth level. They were
fitted at one trial count and one box, and they are entangled with both: the g3
floor used to be `1/n_trials`, so changing k moved the floor, which moved the
weight distribution, which invalidated the fitted gamma. Stage 1 pinned the
floor. This module removes the fitted numbers, replacing all nine with one
target: the fraction of the trial pool that should carry the selection
probability.

For normalised trial weights p_i = w_i / sum_j w_j,

    PR = 1 / sum_i p_i^2 = (sum_i w_i)^2 / sum_i w_i^2

is the PARTICIPATION RATIO, between 1 and the pool size. `PR / pool size` is the
PARTICIPATION FRACTION, and `f_pr` is the target for it. PR = 1 means one trial
takes every pick and the cloud is wasted; PR = k means the weights are flat and
the bias is doing nothing. The gamma that hits a chosen fraction is solved per
level from the weights already in memory, so the decay across levels emerges
instead of being fitted.

PR is algebraically Kong's effective sample size (Kong, Liu & Wong, JASA 89:278,
1994), as adaptive-SMC tempering uses it. It is called participation here
because this project already reports a chain effective sample size from
autocorrelation (`gcmc/analysis/autocorr.py`), and two quantities called ESS in
one paper is a referee comment waiting to happen.

`f_pr` is a VARIANCE KNOB ONLY. The bias is divided back out of the acceptance,
so no choice of `f_pr` can move an equilibrium average; only the variance and
the acceptance rate move. Gate G6 is what proves that, and it is the gate that
matters.

Three rules hold this module's shape, and all three are about detailed balance
rather than about numerics:

1.  STATELESS. Every function here is pure. Nothing is cached, warm-started or
    carried between calls, levels, moves or passes. A gamma that depended on
    chain history would turn a fixed kernel into an adaptive one and forfeit
    exact stationarity, and it would do so while looking entirely reasonable.
2.  NO BISECTION. PR(gamma) is not monotone. With w_i = a_i * b_i^gamma,
    d log(PR) / d gamma = 2 (<log b>_w - <log b>_{w^2}), whose sign follows the
    correlation between the bias and the weight: where the bias likes what the
    Boltzmann factor dislikes, PR RISES with gamma before falling. A fixed grid
    is exactly reproducible and cannot fail to bracket; bisection assumes a
    unique root that need not exist.
3.  PERMUTATION-INVARIANT BIT-FOR-BIT. The engine permutes the pool
    (`pool = pool[perm]`), so the same physical set of trials arrives in
    different orders. `np.sum` is order-dependent, so every reduction here sorts
    ascending first. Two orders of one pool must give one gamma, or the kernel
    depends on the RNG draw that produced the permutation.

Vocabulary note, because this codebase overloads one letter: the number of
Rosenbluth levels is `n_levels` (it equals `n_atoms`, the number of molecules
the move inserts or removes) and it is never called L. `L` in this project is
the trial box side (`sampler.trial_box_side`), which has nothing to do with any
of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._types import FloatArray


def participation_ratio(w: FloatArray) -> float:
    """PR = (sum w)^2 / sum w^2 over non-negative weights.

    Sorted ascending before either reduction, so the answer is the same float
    for every ordering of the same pool (rule 3). Sorting by w also sorts w^2,
    because the weights cannot be negative.

    A non-positive or non-finite total returns 0.0 rather than NaN. That is the
    exhausted pool: every trial annihilated, which the caller is about to reject
    the move over anyway (`EnergyWeigher.selection` returns no probability for
    it). Returning 0.0 keeps the solver's arithmetic finite and its answer
    deterministic on the way to that rejection; returning NaN would make the
    grid comparison below undefined and the chosen gamma a matter of luck.
    """
    ws = np.sort(np.asarray(w, dtype=float))
    total = np.sum(ws)
    total_sq = np.sum(ws * ws)
    if not (total > 0.0) or not np.isfinite(total) or not (total_sq > 0.0):
        return 0.0
    return float(total * total / total_sq)


def participation_fraction(w: FloatArray) -> float:
    """PR / pool size, in [0, 1]. The quantity `f_pr` targets under the default
    criterion. An empty pool has no participation to speak of and returns 0.0.
    Kept as its own name because it is the thing the method is named after;
    `spread_fraction` is the general form."""
    n = len(np.atleast_1d(np.asarray(w, dtype=float)))
    if n == 0:
        return 0.0
    return participation_ratio(w) / n


def perplexity(w: FloatArray) -> float:
    """exp(-sum p_i log p_i) over non-negative weights: the other criterion.

    The FALLBACK the plan names, and it is here as a peer rather than a patch
    because which one is right is an empirical question. Both measure how spread
    the selection probability is and both equal the pool size on flat weights,
    so `f_pr` means the same kind of thing under either. They differ in what
    they punish: the participation ratio is a second moment, so one dominant
    trial pulls it almost to 1 whatever the rest look like, while perplexity is
    an entropy and keeps reading the shape of the tail. At small k the plug-in PR
    also carries O(1/k) bias, which is the drift the plan's efficiency note
    warns about.

    Same sorted reduction as `participation_ratio`, for the same reason (rule 3).
    Zero weights contribute nothing: p log p -> 0 as p -> 0, so they are dropped
    rather than producing a NaN from log(0) times 0.
    """
    ws = np.sort(np.asarray(w, dtype=float))
    total = np.sum(ws)
    if not (total > 0.0) or not np.isfinite(total):
        return 0.0
    p = ws / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(np.exp(-np.sum(p * np.log(p))))


#: The two criteria, by the name the config uses. Each maps weights to a number
#: between 1 and the pool size, which divided by the pool size is the fraction
#: `pr_target` names. Adding a third is a function and an entry.
CRITERIA = {
    "participation": participation_ratio,
    "perplexity": perplexity,
}


def spread_fraction(w: FloatArray, criterion: str = "participation") -> float:
    """The chosen criterion divided by the pool size, in [0, 1]. An empty pool
    has no spread to speak of and returns 0.0."""
    try:
        measure = CRITERIA[criterion]
    except KeyError:
        raise ValueError(
            f"unknown criterion {criterion!r}; have {sorted(CRITERIA)}"
        ) from None
    n = len(np.atleast_1d(np.asarray(w, dtype=float)))
    if n == 0:
        return 0.0
    return measure(w) / n


def _fraction_from_logs(log_w: FloatArray, criterion: str = "participation") -> float:
    """The criterion's fraction for weights given by their logs.

    Done in log space and shifted by the maximum before exponentiating. PR is
    invariant under a common scale factor, so the shift is free, and it is what
    keeps the grid search finite: `log_bias` summed over thirty neighbours
    reaches a few hundred, and `exp` of that times a Boltzmann factor overflows
    to inf, which would turn the whole reduction into NaN. The shift is a
    scalar, so it cannot disturb rule 3.
    """
    lw = np.asarray(log_w, dtype=float)
    if lw.size == 0:
        return 0.0
    top = np.max(lw)
    if not np.isfinite(top):
        # every weight is zero (-inf) or the pool carries a NaN; either way
        # there is no spread to measure and no gamma worth solving
        return 0.0
    return spread_fraction(np.exp(lw - top), criterion)


def _fractions_on_grid(
    log_base: FloatArray,
    log_bias: FloatArray,
    grid: FloatArray,
    criterion: str,
) -> FloatArray:
    """The spread fraction at EVERY grid point, in one pass.

    THIS IS THE HOT PATH AND IT WAS THE WHOLE COST OF THE FEATURE. The obvious
    way to write the solver is a Python loop over the grid calling
    `spread_fraction` once per point, and that is what it was. Measured on a
    real run -- hard cores, k = 200, `full_repacking`, so every move walks about
    six levels over pools of up to 200 -- it cost 40 to 50% of move time
    against the 5% the method was budgeted, because 17 grid points times 6
    levels is a hundred Python-level calls per move, each sorting its own array.
    The per-move acceptance was competitive; the wall clock was not, and ESJD
    per second is a ratio of the two.

    So the grid is one (n_grid, k) matrix and the reductions are four
    vectorised calls. Same arithmetic, same answers, one C dispatch per stage
    instead of a hundred.

    The sort stays, and stays per row: it is what makes the reduction
    order-independent (rule 3), and `np.sort` along an axis is a single C call
    over the whole matrix rather than 17 Python ones. The max-shift is per row
    for the same reason it was scalar before -- PR is scale-invariant, and
    without it `exp` of a log_bias summed over thirty neighbours overflows.
    """
    lw = log_base[None, :] + grid[:, None] * log_bias[None, :]
    top = np.max(lw, axis=1, keepdims=True)
    # A row whose maximum is not finite has no weight at all (every trial
    # annihilated) or carries a NaN; either way there is no spread to measure.
    dead = ~np.isfinite(top[:, 0])
    if dead.any():
        top = np.where(np.isfinite(top), top, 0.0)
    w = np.sort(np.exp(lw - top), axis=1)
    total = np.sum(w, axis=1)
    n = w.shape[1]
    live = np.isfinite(total) & (total > 0.0) & ~dead
    out = np.zeros(grid.shape[0], dtype=float)
    if not live.any():
        return out
    if criterion == "participation":
        sq = np.sum(w * w, axis=1)
        ok = live & (sq > 0.0)
        out[ok] = (total[ok] * total[ok]) / (sq[ok] * n)
    elif criterion == "perplexity":
        pr = w[live] / total[live, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(pr > 0.0, pr * np.log(pr), 0.0)
        out[live] = np.exp(-np.sum(plogp, axis=1)) / n
    else:
        raise ValueError(f"unknown criterion {criterion!r}; have {sorted(CRITERIA)}")
    return out


def gamma_grid(gamma_max: float, n_grid: int = 17) -> FloatArray:
    """The candidate gammas: exactly 0, then `n_grid - 1` points spaced
    logarithmically from 1e-3 to `gamma_max`.

    Ascending, fixed, and a constant of the run. Zero is a grid point in its own
    right because "no bias at this level" is a real answer and the fitted
    schedule used it at levels 7 through 9; the log spacing then covers the
    small gammas where the fitted numbers actually lived (0.0037 to 0.29) at the
    same resolution as the large ones, which a linear grid would not.
    """
    if not (gamma_max > 0.0):
        raise ValueError(f"gamma_max must be > 0, got {gamma_max!r}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be at least 2, got {n_grid!r}")
    return np.concatenate(
        [[0.0], np.logspace(-3.0, np.log10(float(gamma_max)), n_grid - 1)]
    )


#: Pool sizes above which the reductions below build their products in numpy
#: rather than in a list comprehension. Purely a crossover: at the median level
#: pool of about twenty trials the comprehension wins, at two hundred numpy
#: does, and the two agree to the bit because `math.fsum` is exact either way.
#: Not a config key: it changes the cost and not the answer.
_VECTOR_FROM = 24


def _exact_sums(a: FloatArray, *cols: FloatArray) -> list[float]:
    """`math.fsum` of `a`, then of `a` times each column, exactly rounded.

    THIS IS WHAT REPLACED THE SORT. `np.sum` is pairwise and therefore
    order-dependent, and the engine permutes the pool (`pool = pool[perm]`), so
    a plain reduction would make gamma a function of the RNG draw that produced
    the permutation (rule 3). The grid solver bought invariance by sorting
    inside every reduction, which cost a sort per grid point.

    `math.fsum` is CORRECTLY ROUNDED: it returns the nearest double to the exact
    sum, which is a property of the multiset and not of the order. So it is
    permutation-invariant by construction, and strictly more so than sorting --
    sorting only guarantees that two permutations take the same path through the
    same rounding errors, while this has none to take. Measured at the median
    pool size it is also four times faster than the sorted numpy version it
    replaced, because a level pool is about twenty trials and numpy's per-call
    overhead dominates everything at that size.
    """
    out = [math.fsum(a.tolist())]
    if a.size >= _VECTOR_FROM:
        for col in cols:
            out.append(math.fsum((a * col).tolist()))
    else:
        al = a.tolist()
        for col in cols:
            out.append(math.fsum([x * y for x, y in zip(al, col.tolist())]))
    return out


def _fraction_at(log_w: FloatArray, n: int, criterion: str) -> float:
    """The criterion's fraction of `n`, from log weights, reduced with `fsum`.

    The same quantity `_fraction_from_logs` returns, without the sorted numpy
    reductions: this one is called on the clamp path of the closed-form solver,
    where it is the whole cost, so it goes through `_exact_sums` like everything
    else there. Exactly rounded reductions, so it is permutation-invariant for
    the same reason (rule 3).

    Shifted by the largest log weight before exponentiating, because both
    criteria are invariant under a common factor and `log_bias` summed over
    thirty neighbours reaches a few hundred, where `exp` overflows.
    """
    if log_w.size == 0 or n <= 0:
        return 0.0
    top = float(np.max(log_w))
    if not math.isfinite(top):
        return 0.0
    lw = log_w - top
    w = np.exp(lw)
    if criterion == "participation":
        sw, sww = _exact_sums(w, w)
        if not (sw > 0.0) or not (sww > 0.0) or not math.isfinite(sw):
            return 0.0
        return (sw * sw) / (sww * n)
    sw, swl = _exact_sums(w, lw)
    if not (sw > 0.0) or not math.isfinite(sw):
        return 0.0
    return math.exp(math.log(sw) - swl / sw) / n


def _quadratic_root(a: float, b: float, c: float, gamma_max: float) -> float | None:
    """Smallest root of a x^2 + b x = c in [0, gamma_max], or None.

    Written out rather than handed to `np.roots`, because the degenerate cases
    are the common ones: a == 0 happens whenever the bias has no spread left,
    and both zero happens on a pool of identical weights.
    """
    tiny = 1e-300
    if abs(a) < tiny:
        if abs(b) < tiny:
            return 0.0 if abs(c) < tiny else None
        x = c / b
        return x if 0.0 <= x <= gamma_max else None
    disc = b * b + 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    xs = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
    for x in xs:
        if 0.0 <= x <= gamma_max:
            return float(x)
    return None


def solve_gamma_moments(
    base_w: FloatArray,
    log_bias: FloatArray,
    f_target: float,
    gamma_max: float,
    criterion: str = "participation",
) -> float:
    """The exponent in closed form, from two moments of the pool. No grid.

    WHY THIS EXISTS. The grid search evaluates the spread at 17 candidate
    exponents, which is 17 k exponentials and 17 sorted reductions per level.
    Measured on a real run -- hard cores, k = 200, `full_repacking` -- that was
    1450 exponentials per move and 13% of move cost, to answer a question that
    does not need them.

    THE QUESTION HAS A CLOSED FORM. Write u_i = log b_i and let P be the pool
    weighted by the gamma-free part a_i, Q the pool weighted by a_i^2. Then the
    participation ratio of w_i = a_i exp(gamma u_i) obeys

        log( PR(gamma)/k ) = log( PR(0)/k ) + 2 kappa_P(gamma) - kappa_Q(2 gamma)

    exactly, where kappa is a cumulant generating function. Expanding to second
    order,

        log( F(gamma) / F(0) ) = 2 gamma (m_P - m_Q) + gamma^2 (v_P - 2 v_Q)

    with m, v the mean and variance of u under each weighting. That is a
    quadratic in gamma: two moments, one square root. Perplexity gives the same
    shape with A = -v_P / 2 and B = -Cov_P(log a, u).

    Note what falls out of the leading term. With a_i constant -- every
    admissible hard-core trial, at level 1 -- P and Q coincide, B vanishes and

        gamma = sqrt( log(1/f) / Var(log b) )

    which is the whole solver for that case: one variance and a square root.

    ACCURACY IS NOT THE POINT, and this is why an expansion is legitimate rather
    than a corner cut. Gamma has to be a deterministic, permutation-invariant
    function of the level pool and nothing else; detailed balance does not care
    which function. A gamma that lands a few percent from the ideal moves the
    variance of the estimator and nothing else, because the bias is divided back
    out of the acceptance. What would break the method is a gamma that depended
    on chain history, and this depends on the pool alone.

    The expansion is exact to O(gamma^3), so it is at its best where the tuned
    schedule actually lives (gamma 0.03 to 0.29) and degrades towards the top of
    the bracket. `solve_gamma` remains available for a pilot run that wants the
    grid's answer to compare against.
    """
    a = np.asarray(base_w, dtype=float)
    u = np.asarray(log_bias, dtype=float)
    if a.shape != u.shape:
        raise ValueError(
            f"base_w has shape {a.shape} and log_bias {u.shape}; they are "
            f"the two factors of one weight per trial and must be row-aligned. "
            f"A shape mismatch here usually means unreferenced trials were "
            f"dropped from one of them instead of contributing log_bias = 0."
        )
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; have {sorted(CRITERIA)}")
    if a.size < 2:
        # An empty pool has no spread and a pool of one participates fully at
        # every exponent, so the answer is the least biased one. Short-circuited
        # because a level pool of one is common at depth.
        return 0.0
    if not (0.0 < f_target <= 1.0):
        raise ValueError(
            f"pr_target must satisfy 0 < pr_target <= 1, got {f_target!r}: it is "
            f"a fraction of the trial pool, not a gamma."
        )
    if not (gamma_max > 0.0):
        raise ValueError(f"gamma_max must be > 0, got {gamma_max!r}")
    n = a.size
    top = float(np.max(a))
    if not (top > 0.0) or not math.isfinite(top):
        # every trial annihilated, or the pool carries an inf or a NaN: there is
        # no spread to tune and the caller is about to reject the move anyway
        return 0.0
    # Rescaled by the largest weight before anything is squared. Every quantity
    # below is invariant under a common factor on `a` -- the moments of u are,
    # and so is each f_zero -- so this costs nothing and it is what keeps
    # `a * a` from overflowing on a pool whose Boltzmann factors span
    # hundreds of orders of magnitude. The grid solver's max-shift, in the one
    # place this solver can overflow.
    a = a / top
    if criterion == "perplexity":
        # log(0) would poison the perplexity term. An annihilated trial has
        # weight 0 at every gamma, so dropping it changes no moment; `n` stays
        # the full pool size, because the fraction is measured against the pool
        # the sampler proposed.
        keep = a > 0.0
        if not bool(keep.all()):
            a, u = a[keep], u[keep]
            if a.size < 2:
                return 0.0
    uu = u * u
    sa, s_au, s_auu = _exact_sums(a, u, uu)
    if not (sa > 0.0) or not math.isfinite(sa):
        return 0.0
    m_p = s_au / sa
    v_p = max(0.0, s_auu / sa - m_p * m_p)
    if criterion == "participation":
        sa2, s_a2u, s_a2uu = _exact_sums(a * a, u, uu)
        if not (sa2 > 0.0) or not math.isfinite(sa2):
            return 0.0
        m_q = s_a2u / sa2
        v_q = max(0.0, s_a2uu / sa2 - m_q * m_q)
        f_zero = (sa * sa) / (sa2 * n)
        coef_a = v_p - 2.0 * v_q
        coef_b = 2.0 * (m_p - m_q)
    else:
        log_a = np.log(a)
        _, s_al, s_alu = _exact_sums(a, log_a, log_a * u)
        mean_log_a = s_al / sa
        f_zero = math.exp(math.log(sa) - mean_log_a) / n
        cov = s_alu / sa - mean_log_a * m_p
        coef_a = -0.5 * v_p
        coef_b = -cov
    if not (f_zero > 0.0) or not np.isfinite(f_zero):
        return 0.0
    target = math.log(f_target) - math.log(f_zero)
    root = _quadratic_root(coef_a, coef_b, target, gamma_max)
    if root is not None:
        return root
    # NO EXPONENT IN THE BRACKET REACHES THE TARGET, which is ordinary (see
    # `solve_gamma`): a target above the fraction at gamma = 0 is out of reach
    # when the spread falls with gamma, and below it when the spread rises.
    #
    # The answer is then one END of the bracket, and the expansion is the wrong
    # thing to pick it with. Its coefficients are fitted at gamma = 0 and it is
    # exact to O(gamma^3), so extrapolating it to gamma_max is exactly where it
    # is least trustworthy -- and it is monotone past its vertex while the true
    # spread is not (rule 2), so it will happily claim gamma_max is closer when
    # the spread has already turned back up.
    #
    # So the candidates are evaluated honestly, at the real criterion. There
    # are three of them: the two ends, and the quadratic's VERTEX, which is the
    # expansion's own estimate of where the spread turns. The vertex is what
    # makes the answer competitive with the grid on an unreachable target --
    # for the pool this was measured on the spread peaks at 0.41 around
    # gamma = 0.35 and a target of 0.9 cannot be met anywhere, and the vertex
    # is the point the grid would have found. Three exponential passes over the
    # pool against the grid's seventeen, and only on this path, which a level
    # that solved never takes.
    with np.errstate(divide="ignore"):
        log_a = np.log(a)
    cands = [0.0, float(gamma_max)]
    if abs(coef_a) > 1e-300:
        vertex = -coef_b / (2.0 * coef_a)
        if 0.0 < vertex < gamma_max:
            cands.append(float(vertex))
    miss = [abs(_fraction_at(log_a + g * u, n, criterion) - f_target) for g in cands]
    # Ties go to the smallest candidate: least bias among equally good answers,
    # which is the grid's convention (`argmin` over an ascending grid).
    best = min(range(len(cands)), key=lambda i: (miss[i], cands[i]))
    return cands[best]


def solve_gamma(
    base_w: FloatArray,
    log_bias: FloatArray,
    f_target: float,
    gamma_max: float,
    n_grid: int = 17,
    criterion: str = "participation",
) -> float:
    """The grid point whose participation fraction is closest to `f_target`.

    `base_w` is the part of each trial's weight gamma does not touch and
    `log_bias` the log of the part it does, so the candidate weight at gamma g
    is `base_w * exp(g * log_bias)`. No interpolation and no refinement: the
    answer is one of the grid points, which makes it exactly reproducible.

    Ties go to the smallest gamma, because the grid is ascending and `argmin`
    returns the first minimum. Least bias among equally good answers.

    UNREACHABLE TARGETS ARE NORMAL. At gamma = 0 the weights are the Boltzmann
    factors alone and their participation is whatever it is; a target below that
    is out of reach if PR falls with gamma, and a target above it is out of
    reach if PR rises. Either way this returns the closest grid point rather
    than raising, and a return of exactly 0.0 is the signal the caller counts as
    a clamped level. The grid decides it rather than a special case, because PR
    is not monotone (rule 2): "the fraction at gamma = 0 is below the target"
    does not imply "gamma = 0 is the closest point", and a hard clamp would
    return a worse answer than the grid already has in hand.
    """
    base = np.asarray(base_w, dtype=float)
    logb = np.asarray(log_bias, dtype=float)
    if base.shape != logb.shape:
        raise ValueError(
            f"base_w has shape {base.shape} and log_bias {logb.shape}; they are "
            f"the two factors of one weight per trial and must be row-aligned. "
            f"A shape mismatch here usually means unreferenced trials were "
            f"dropped from one of them instead of contributing log_bias = 0."
        )
    if base.size == 0:
        return 0.0
    if base.size == 1:
        # One trial always participates fully, whatever the exponent, so every
        # grid point ties and the tie goes to the smallest. Short-circuited
        # because a level pool of one is common at depth and the grid would be
        # seventeen exponentials to reach a foregone answer.
        return 0.0
    if not (0.0 < f_target <= 1.0):
        raise ValueError(
            f"pr_target must satisfy 0 < pr_target <= 1, got {f_target!r}: it is "
            f"a fraction of the trial pool (PR / k), not a gamma."
        )
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; have {sorted(CRITERIA)}")
    with np.errstate(divide="ignore"):
        log_base = np.log(base)
    grid = gamma_grid(gamma_max, n_grid)
    miss = np.abs(_fractions_on_grid(log_base, logb, grid, criterion) - f_target)
    return float(grid[int(np.argmin(miss))])


#: Default number of grid points. A constant of the METHOD rather than a config
#: key: 17 points over [0, gamma_max] put the answer within a grid step of the
#: ideal gamma, and a gamma one step off changes the variance of the estimator
#: and nothing else, because the bias is divided back out of the acceptance.
#: Exposing it would add an identity-bearing field whose only honest use is a
#: pilot study, and a pilot study can construct a `ParticipationTarget`
#: directly.
N_GRID = 17

#: Default bracket. The fitted 3D schedule saturates by gamma = 2, so 3 leaves
#: room above every value that was ever useful.
GAMMA_MAX = 3.0

#: How the exponent is obtained, by the name the config uses. `moments` is the
#: closed form: two weighted moments of log b and one square root, no
#: exponentials and no grid. `grid` is the original search over
#: `gamma_grid(gamma_max, n_grid)`, kept because a pilot run comparing the two
#: is the only honest way to show the expansion costs nothing that matters, and
#: a solver that exists only in a git history cannot be run.
#:
#: `moments` is the DEFAULT because the grid was the cost of the method: 17 grid
#: points times k trials times the levels a repacking move walks came to about
#: 1450 exponentials per move, 13% of move time, and ESJD per second is a ratio
#: in which that denominator shows up directly. The two solvers answer the same
#: question and land equally close to the target (the expansion is sometimes
#: closer, because the grid can only return a grid point); they are separate
#: run identities because they select differently, not because one is right.
SOLVERS = ("moments", "grid")


@dataclass(frozen=True, slots=True)
class ParticipationTarget:
    """The three constants of an adaptive-gamma run, and the solve they define.

    Frozen on purpose. `f_pr`, `gamma_max` and the grid are constants of the
    run: they are tuned in pilot runs and then they are part of the method.
    Updating any of them from accumulated acceptance statistics mid-run is the
    one change that produces a result which looks fine and is wrong, because it
    makes the transition kernel a function of chain history.

    Holding them together rather than passing three arguments is not tidiness.
    The forward pass and the reverse pass MUST solve against the same
    constants, and the way that goes wrong is one call site being updated and
    the other not; one object threaded to both cannot come apart.
    """

    #: target spread fraction (the criterion divided by k), in (0, 1]
    f_pr: float
    gamma_max: float = GAMMA_MAX
    n_grid: int = N_GRID
    #: which spread the target is measured with; see `CRITERIA`. A constant of
    #: the run like the other three, and part of the run identity: the same
    #: target under two criteria proposes two different ways.
    criterion: str = "participation"
    #: which solver, see `SOLVERS`. `n_grid` is dead weight under "moments" and
    #: is kept on the object anyway, because the field is what lets a pilot run
    #: construct both targets from one set of constants.
    solver: str = "moments"

    def __post_init__(self) -> None:
        if self.criterion not in CRITERIA:
            raise ValueError(
                f"unknown criterion {self.criterion!r}; have {sorted(CRITERIA)}"
            )
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown solver {self.solver!r}; have {sorted(SOLVERS)}")

    def solve(self, base_w: FloatArray, log_bias: FloatArray) -> float:
        """The exponent for one level's pool, at these constants.

        Stateless: this object carries no solver state and calling it twice on
        one pool returns one float. The criterion and the solver apply at EVERY
        level, depth included: both are properties of the run, not of the
        level."""
        if self.solver == "moments":
            return solve_gamma_moments(
                base_w,
                log_bias,
                self.f_pr,
                self.gamma_max,
                self.criterion,
            )
        return solve_gamma(
            base_w,
            log_bias,
            self.f_pr,
            self.gamma_max,
            self.n_grid,
            self.criterion,
        )
