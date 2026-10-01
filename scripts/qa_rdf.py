#!/usr/bin/env python3
"""Average g11 (fluid-fluid) and g01 (fluid-matrix) over the matrix realizations
in a QA target sweep, matching Lomba 1993's procedure (average over matrix
configurations). Run on the cluster against your gcmc install.

Set RUN below to pick the state, then just:

    python qa_rdf.py

Each realization gives ONE g(r), built from all its post-burn frames against its
own frozen matrix. The 16 realization curves are the sample set; the error bar is
their scatter across realizations (matrix-to-matrix spread), not anything from
inside a single trajectory. g01 must stay per-realization because it is the fluid
measured against that run's matrix, so the curves are averaged, not the raw frames.

ERROR picks what the error column means:
    "spread" -> std across realizations (ddof=1): how much the matrices differ
    "sem"    -> std / sqrt(n_realizations): error on the matrix-averaged mean

Output columns: r g11_mean g11_err g01_mean g01_err.
"""
import sys, os, glob
import numpy as np
from ase.io import read as ase_read
from gcmc.rdf import RDFAccumulator
from gcmc import io

# ============================== CONFIG ===============================
# Flip RUN to switch state. Both share BURN_M / R_MAX / N_BINS / ERROR.
RUN = "both"              # "0.3_0.3", "0.6_0.1", or "both"

STATES = {
    "0.3_0.3": dict(root="runs/N_sweep_0.3_0.3", box=9.48, out="rdf_0.3_0.3.csv"),
    "0.6_0.1": dict(root="runs/N_sweep_0.6_0.1", box=9.41, out="rdf_0.6_0.1.csv"),
}

BURN_M = 0.25            # burn-in, MC-cycle millions
R_MAX  = 4.0            # max r (paper plots to 4)
N_BINS = 160
ERROR  = "sem"          # "sem" (std/sqrt n: accuracy of the mean) or "spread" (std)
# =====================================================================


def realization_dirs(root):
    # each realization is a run dir holding matrix.npz + trajectory.extxyz
    ds = sorted(d for d in glob.glob(os.path.join(root, "*"))
                if os.path.isfile(os.path.join(d, "trajectory.extxyz"))
                and os.path.isfile(os.path.join(d, "matrix.npz")))
    return ds


def gr_for_realization(run_dir, box, burn_M, r_max, n_bins, dim=3):
    matrix = io.load_matrix(os.path.join(run_dir, "matrix.npz"), dim)
    acc = RDFAccumulator(box=box, dimension=dim, r_max=r_max, n_bins=n_bins)
    acc.set_matrix(matrix)
    frames = ase_read(os.path.join(run_dir, "trajectory.extxyz"),
                      index=":", format="extxyz")
    used = 0
    for fr in frames:
        step = fr.info.get("step", None)
        # burn-in is given in MC-cycle millions; step is in cycles
        if step is not None and float(step) < burn_M * 1e6:
            continue
        acc.accumulate(fr.get_positions())
        used += 1
    r, g11, g01 = acc.finalize()
    return r, g11, g01, used, len(matrix)


def main(root, box, burn_M, r_max, n_bins, out):
    dirs = realization_dirs(root)
    if not dirs:
        sys.exit(f"no realization dirs (with matrix.npz + trajectory.extxyz) under {root}")
    print(f"{len(dirs)} realizations under {os.path.basename(root)}, error={ERROR}")

    g11s, g01s, r = [], [], None
    for d in dirs:
        r, g11, g01, used, nm = gr_for_realization(d, box, burn_M, r_max, n_bins)
        g11s.append(g11); g01s.append(g01)
        print(f"  {os.path.basename(d):40s} frames_used={used:5d} matrix_N={nm}")

    g11s, g01s = np.array(g11s), np.array(g01s)
    n = len(dirs)
    if n < 2:
        sys.exit(f"need >=2 realizations for a spread; found {n}")

    # spread across realizations (ddof=1); SEM is that divided by sqrt(n)
    g11_sd, g01_sd = g11s.std(0, ddof=1), g01s.std(0, ddof=1)
    scale = np.sqrt(n) if ERROR == "sem" else 1.0
    g11_m, g11_e = g11s.mean(0), g11_sd / scale
    g01_m, g01_e = g01s.mean(0), g01_sd / scale

    M = np.column_stack([r, g11_m, g11_e, g01_m, g01_e])
    np.savetxt(out, M, header="r g11_mean g11_err g01_mean g01_err", comments="")
    print(f"wrote {out}: {n_bins} bins to r_max={r_max}, "
          f"{ERROR} across {n} realizations")


if __name__ == "__main__":
    todo = list(STATES) if RUN == "both" else [RUN]
    for key in todo:
        cfg = STATES[key]
        main(cfg["root"], cfg["box"], BURN_M, R_MAX, N_BINS, cfg["out"])
