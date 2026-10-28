"""The run log as the analysis scripts read it.

`load` returns a DataFrame under the current column names (see
`gcmc.run.runner`). A log written before the raw-column format is translated
name by name, so the scripts read old and new runs alike. Columns the old
format derived (`pckng_frctn`) are kept as they are; `packing_fraction` uses
them when present and computes eta from the counts otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OLD_NAMES = {
    "MC_cycle(M)": "MC_cycle",
    "add_stp": "insert_tried",
    "add_att": "insert_evaluated",
    "add_acc": "insert_accepted",
    "rm_stp": "delete_tried",
    "rm_acc": "delete_accepted",
    "mv_stp": "translate_tried",
    "mv_acc": "translate_accepted",
    "glb_same": "ladder_unchanged",
    "glb_dN": "ladder_sum_abs_dN",
    "glb_kmax": "ladder_max_k",
    "glb_cut_k": "ladder_cut_kmax",
    "glb_cut_M": "ladder_cut_pool",
    "glb_infeas": "ladder_infeasible",
}


def load(path: str | Path, min_rows: int = 0) -> pd.DataFrame | None:
    """The log at `path` (a run.log or its run folder); None below `min_rows`."""
    path = Path(path)
    if path.is_dir():
        path = path / "run.log"
    with path.open() as fh:
        names = fh.readline().lstrip("#").split()
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=names)
    if len(df) < min_rows:
        return None
    return df.rename(columns=OLD_NAMES)


def packing_fraction(cols: Mapping[str, Any], params: dict[str, Any]) -> np.ndarray:
    """eta per row of a log (a DataFrame or a dict of columns): the logged
    column of an old run, else computed from the counts."""
    if "pckng_frctn" in cols:
        return np.asarray(cols["pckng_frctn"], float)
    from gcmc.run.sim_setup import from_params

    cfg = from_params(params)
    if "N" in cols:
        return np.array([cfg.packing_fraction(n) for n in cols["N"]])
    assert cfg.species is not None
    counts = np.column_stack([cols[f"N_{s}"] for s in cfg.species.names])
    return np.array([cfg.packing_fraction(c) for c in counts])


def total_n(cols: Mapping[str, Any]) -> np.ndarray:
    """N per row: the N column, or the sum of the N_<species> columns."""
    if "N" in cols:
        return np.asarray(cols["N"], float)
    return np.sum([np.asarray(cols[c], float) for c in cols if c.startswith("N_")], 0)
