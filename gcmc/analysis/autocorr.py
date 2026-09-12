"""Integrated autocorrelation time of a scalar chain observable, and its cost.

WHY THIS MODULE EXISTS. Every performance claim in the design plan
is stated as "tau_N per CPU-second", and before G0 there was no estimator for
it anywhere in the tree. Acceptance ratio is not a substitute: the Glauber engine has
no acceptance test at all, so the only thing the two engines can be compared on
is how much CPU time one statistically independent value of N costs.

THE DEFINITION, stated once so nothing downstream has to guess.

    rho(t)   normalised autocorrelation of the sampled series x_0..x_{n-1}
    tau_int  1/2 + sum_{t=1}^{W} rho(t)                     [in SAMPLES]
    tau_cpu  tau_int * (cpu_seconds / n_samples)            [in CPU-SECONDS]

`tau_int` counts SAMPLES, so it depends on how often the caller sampled; two
runs that logged N at different cadences are not comparable in that unit. Only
`tau_cpu` is, which is why it is what the gates quote. The factor of 1/2 is the
usual convention that makes tau_int = 1/2 for white noise and makes

    var(mean)  =  2 * tau_int * var(x) / n

exact in the large-n limit; `sem` below is that expression, and it is the only
honest error bar for a correlated chain (the naive var/n underestimates it by
2*tau_int, which at tau_int = 40 is a factor of 80).

THE WINDOW. Summing rho(t) to the end of the series adds n/2 terms of pure
noise and the estimate diverges. Sokal's automatic windowing is used: take the
smallest W with W >= c * tau_int(W), c = 5 for an exponential-ish decay. The
alternative -- cutting at the first negative rho -- is cheaper and much noisier
for the slowly-decaying series this project produces, so it is kept only as the
fallback when the windowing condition never fires.

RELIABILITY IS REPORTED, NOT ASSUMED. `tau_int` returns a `Tau` carrying the
window it chose and whether the series was long enough for it (`n >= 50 * tau`
is the usual rule of thumb). A tau estimated from a series shorter than that is
biased LOW, which flatters whichever engine is being measured, so the gates read
`reliable` and refuse to quote a number that is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._types import FloatArray

#: Sokal's windowing constant. 5 is the standard choice for a series whose
#: autocorrelation is roughly exponential; larger c windows later (less bias,
#: more variance).
SOKAL_C = 5.0

#: How many integrated autocorrelation times a series must span before its own
#: tau estimate is trusted. Below this the estimate is biased low.
MIN_SPAN = 50.0


@dataclass(frozen=True)
class Tau:
    """An integrated autocorrelation time and everything needed to judge it.

    tau : float          integrated autocorrelation time, in SAMPLES
    window : int         the lag the sum was truncated at
    n : int              length of the series
    var : float          sample variance of the series
    mean : float         sample mean
    reliable : bool      n >= MIN_SPAN * tau, i.e. the series is long enough
                         for its own tau estimate to be trusted
    """

    tau: float
    window: int
    n: int
    var: float
    mean: float
    reliable: bool

    @property
    def n_eff(self) -> float:
        """Effective number of independent samples, n / (2 tau)."""
        return self.n / (2.0 * self.tau) if self.tau > 0.0 else float(self.n)

    @property
    def sem(self) -> float:
        """Standard error of the mean, sqrt(2 tau var / n).

        The naive sqrt(var/n) is wrong by sqrt(2 tau) for a correlated chain and
        is what makes two engines look like they disagree when they do not.
        """
        if self.n == 0:
            return float("nan")
        return math.sqrt(max(0.0, 2.0 * self.tau * self.var / self.n))


def autocorr(x: FloatArray, max_lag: int | None = None) -> FloatArray:
    """Normalised autocorrelation rho(t), t = 0..max_lag, by FFT.

    rho(0) is 1 by construction. A constant series has zero variance and no
    autocorrelation function; it comes back as rho(0) = 1 and zeros, which makes
    `tau_int` return 1/2 rather than a NaN that propagates into a gate.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n < 2:
        return np.ones(1)
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    d = a - a.mean()
    v = float(d @ d)
    if v <= 0.0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    # zero-pad to a power of two so the circular correlation the FFT computes is
    # the linear one over the lags asked for
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(d, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[: max_lag + 1]
    return np.asarray(acf / acf[0], dtype=np.float64)


def tau_int(x: FloatArray, c: float = SOKAL_C) -> Tau:
    """Integrated autocorrelation time in SAMPLES, with Sokal windowing.

    The window is the smallest W with W >= c * tau(W). When that never fires --
    a series far shorter than its own correlation time -- the sum is truncated
    at the first non-positive rho instead, and `reliable` comes back False, so
    the caller can see that the number is a lower bound rather than a
    measurement.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n < 2:
        return Tau(0.5, 0, n, 0.0, float(a[0]) if n else float("nan"), False)
    rho = autocorr(a)
    csum = 0.0
    window = 0
    fired = False
    for t in range(1, rho.size):
        csum += float(rho[t])
        tau = 0.5 + csum
        if t >= c * max(tau, 1e-12):
            window = t
            fired = True
            break
        if rho[t] <= 0.0 and not fired:
            # remember the first sign change as the fallback truncation
            window = window or t
    if not fired:
        # windowing never satisfied: truncate at the first sign change (or the
        # whole series) and say the estimate is not trustworthy
        stop = window if window else rho.size - 1
        csum = float(rho[1 : stop + 1].sum())
        window = stop
    else:
        csum = float(rho[1 : window + 1].sum())
    tau = max(0.5, 0.5 + csum)
    var = float(a.var(ddof=1)) if n > 1 else 0.0
    return Tau(
        tau=tau,
        window=int(window),
        n=int(n),
        var=var,
        mean=float(a.mean()),
        reliable=bool(fired and n >= MIN_SPAN * tau),
    )


def tau_cpu_seconds(x: FloatArray, cpu_seconds: float, c: float = SOKAL_C) -> float:
    """tau_N in CPU-SECONDS: the CPU cost of one statistically independent value.

    `cpu_seconds` is the CPU time that produced the whole series, so
    `cpu_seconds / n` is the cost of one sample and the product is the cost of
    one independent sample. This is THE metric for every performance claim in
    the plan, and it is deliberately independent of the sampling cadence: a run
    that logs ten times more often gets ten times the tau in samples and the
    same tau in seconds.
    """
    t = tau_int(x, c)
    if t.n == 0:
        return float("nan")
    return t.tau * (float(cpu_seconds) / t.n)


def block_average(x: FloatArray, n_blocks: int = 20) -> tuple[float, float]:
    """(mean, standard error) by blocking, as an independent check on `sem`.

    Blocking and the windowed tau are two different estimators of the same
    quantity; they should agree to within a few tens of percent, and when they
    do not the series is too short for either. Cheap enough to compute both
    every time a gate quotes an error bar.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n < n_blocks or n_blocks < 2:
        return (float(a.mean()) if n else float("nan"), float("nan"))
    size = n // n_blocks
    trimmed = a[: size * n_blocks].reshape(n_blocks, size)
    means = trimmed.mean(axis=1)
    return float(a[: size * n_blocks].mean()), float(
        means.std(ddof=1) / math.sqrt(n_blocks)
    )
