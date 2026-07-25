"""Q9 + Q11: record-run estimators on synthetic series (no simulation)."""

from __future__ import annotations

import numpy as np

from scripts.mof.analysis import (
    R_KJ_PER_MOL_K,
    blocked_stderr,
    geweke_burnin,
    isosteric_heat,
    summarize_series,
)


def _ar1_with_burnin(n=4000, phi=0.8, burn=500, drift=5.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    x[:burn] += drift * np.linspace(1.0, 0.0, burn)  # transient that decays
    return x


def test_geweke_detects_burnin():
    x = _ar1_with_burnin(burn=500, drift=6.0)
    cut = geweke_burnin(x)
    assert 0 < cut < 1500  # past the transient, well before the stationary tail


def test_geweke_small_burnin_for_stationary():
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)  # already stationary
    assert geweke_burnin(x) < 0.3 * len(x)  # no large transient to cut


def test_blocked_stderr_exceeds_naive_for_correlated():
    x = _ar1_with_burnin(burn=0, phi=0.9)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    blocked = blocked_stderr(x, n_blocks=20)
    assert blocked > naive  # autocorrelation inflates the true error


def test_isosteric_heat_ideal_gas_is_RT():
    rng = np.random.default_rng(2)
    n = rng.poisson(5.0, size=2000).astype(float)
    u = np.zeros_like(n)  # ideal gas: no interaction energy
    q = isosteric_heat(u, n, temperature_k=298.0)
    assert q == np.float64(R_KJ_PER_MOL_K * 298.0)


def test_isosteric_heat_attractive_exceeds_RT():
    # if energy drops as N rises (attractive adsorption), cov(U,N) < 0 -> Q_st > RT
    rng = np.random.default_rng(3)
    n = rng.poisson(8.0, size=5000).astype(float)
    u = -2.0 * n + rng.normal(0, 0.5, size=len(n))  # U decreases with N
    q = isosteric_heat(u, n, temperature_k=298.0)
    assert q > R_KJ_PER_MOL_K * 298.0


def test_summarize_series():
    x = _ar1_with_burnin(burn=400, drift=5.0)
    out = summarize_series(x)
    assert out["burn_in"] > 0
    assert out["n_samples"] > 0
    assert out["stderr"] > 0
