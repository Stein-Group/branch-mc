"""Estimators for record runs: equilibration cut, blocked error, isosteric heat.

These operate on the per-window series that run_mof already streams to its CSV
(N_s, u_total), so a production run needs no extra bookkeeping. Nothing here runs
a simulation; it is pure post-processing, unit-tested on synthetic series.

  geweke_burnin       R3.3 (Q11): pick the equilibration cut by the Geweke
                      z-score comparing an early and a late block of the series;
                      never applied silently, the index is reported.
  blocked_stderr      standard error under autocorrelation via block averaging.
  isosteric_heat      Q9 / figure P8: Q_st = RT - cov(U, N) / var(N), the
                      fluctuation formula, from the U and N series.
"""

from __future__ import annotations

import numpy as np

# gas constant in kJ/(mol K)
R_KJ_PER_MOL_K = 8.314462618e-3


def geweke_burnin(
    series: np.ndarray,
    first: float = 0.1,
    last: float = 0.5,
    threshold: float = 2.0,
    step_frac: float = 0.05,
) -> int:
    """Smallest truncation index at which the series looks equilibrated by the
    Geweke test: the mean of the first `first` fraction of the REMAINING series
    agrees with the mean of the last `last` fraction to within `threshold`
    standard errors. Advances the cut in steps of `step_frac` of the length.
    Returns the burn-in index (0 if already stationary, len-1 if never)."""
    x = np.asarray(series, float)
    n = len(x)
    if n < 20:
        return 0
    step = max(1, int(step_frac * n))
    for cut in range(0, n - 10, step):
        tail = x[cut:]
        m = len(tail)
        a = tail[: max(1, int(first * m))]
        b = tail[-max(1, int(last * m)) :]
        va = a.var(ddof=1) / len(a) if len(a) > 1 else 0.0
        vb = b.var(ddof=1) / len(b) if len(b) > 1 else 0.0
        denom = np.sqrt(va + vb)
        if denom == 0.0:
            return cut
        z = (a.mean() - b.mean()) / denom
        if abs(z) < threshold:
            return cut
    return n - 1


def blocked_stderr(series: np.ndarray, n_blocks: int = 20) -> float:
    """Standard error of the mean under autocorrelation, via block averaging:
    split into `n_blocks` contiguous blocks, take the std error of the block
    means. Falls back to the plain stderr when the series is too short."""
    x = np.asarray(series, float)
    n = len(x)
    if n < 2:
        return 0.0
    nb = min(n_blocks, n)
    edges = np.array_split(x, nb)
    means = np.array([e.mean() for e in edges if len(e)])
    if len(means) < 2:
        return float(x.std(ddof=1) / np.sqrt(n))
    return float(means.std(ddof=1) / np.sqrt(len(means)))


def isosteric_heat(
    u_kelvin: np.ndarray, n_total: np.ndarray, temperature_k: float
) -> float:
    """Isosteric heat of adsorption Q_st in kJ/mol from the fluctuation formula
    Q_st = RT - (<UN> - <U><N>) / (<N^2> - <N>^2), with U the total guest energy
    in kelvin and N the total loading per window. Returns nan for a degenerate
    (zero-variance N) series. An ideal gas (U == 0) gives exactly RT."""
    u = np.asarray(u_kelvin, float)
    n = np.asarray(n_total, float)
    var_n = n.var()
    if var_n == 0.0:
        return float("nan")
    cov_un = np.mean(u * n) - u.mean() * n.mean()
    q_st_kelvin = temperature_k - cov_un / var_n
    return float(R_KJ_PER_MOL_K * q_st_kelvin)


def summarize_series(
    series: np.ndarray, n_blocks: int = 20
) -> dict[str, float]:
    """Mean + blocked stderr of the equilibrated tail, with the Geweke cut."""
    x = np.asarray(series, float)
    cut = geweke_burnin(x)
    tail = x[cut:]
    return {
        "burn_in": float(cut),
        "mean": float(tail.mean()) if len(tail) else float("nan"),
        "stderr": blocked_stderr(tail, n_blocks),
        "n_samples": float(len(tail)),
    }
