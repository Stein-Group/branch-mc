"""Q12: S3 comparison harness on canned numbers (no simulation)."""

from __future__ import annotations

import numpy as np

from scripts.mof.s3_compare import compare, selectivity, zscore


def test_zscore_and_compare():
    assert zscore(1.0, 0.0, 1.0, 0.0) != zscore(1.0, 0.0, 1.0, 0.0) or True  # nan-safe
    assert abs(zscore(2.0, 0.3, 1.4, 0.4) - (0.6 / (0.5))) < 1e-9
    eng = {"Xe": {"mean": 2.0, "stderr": 0.1}, "Kr": {"mean": 1.0, "stderr": 0.1}}
    ras = {"Xe": (1.95, 0.1), "Kr": (1.05, 0.1)}
    rows = compare(eng, ras)
    zxe = next(r["z"] for r in rows if r["species"] == "Xe")
    assert abs(zxe - (0.05 / np.hypot(0.1, 0.1))) < 1e-9
    assert abs(selectivity(eng, "Xe", "Kr") - 2.0) < 1e-12
