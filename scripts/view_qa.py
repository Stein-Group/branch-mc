#!/usr/bin/env python3
"""Write a viewable extxyz showing the frozen matrix (species X) together with
the mobile fluid (species H), so you can confirm the matrix is frozen and only
the fluid moves. The matrix is NOT stored in the normal trajectory (it is
constant); this combines them only for visualization.

    python view_qa.py RUN_DIR [--frames last|all] [-o combined.extxyz]

RUN_DIR is a QA run directory (contains matrix.npz and trajectory.extxyz).
"""
import sys, os, argparse
import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write


def main(run_dir, which, out):
    snap = os.path.join(run_dir, "matrix.npz")
    traj = os.path.join(run_dir, "trajectory.extxyz")
    if not os.path.exists(snap):
        sys.exit(f"no matrix.npz in {run_dir} (not a QA run, or older build)")
    with np.load(snap, allow_pickle=False) as z:
        M = np.asarray(z["positions"], float)
        if M.ndim == 1 or (M.ndim == 2 and M.shape[0] == 1):
            M = M.reshape(-1, 3)            # tolerate flattened snapshots
    frames = ase_read(traj, index=(":" if which == "all" else "-1:"), format="extxyz")
    box = frames[0].cell.lengths()[0] if frames[0].cell.rank else 12.0
    combined = []
    for fr in frames:
        F = fr.get_positions()
        # matrix = X (frozen), fluid = H (mobile)
        atoms = Atoms("X" * len(M) + "H" * len(F),
                      positions=np.vstack([M, F]),
                      cell=[box] * 3, pbc=True)
        atoms.info["n_matrix"] = len(M)
        atoms.info["n_fluid"] = len(F)
        combined.append(atoms)
    ase_write(out, combined, format="extxyz")
    print(f"wrote {out}: {len(combined)} frame(s), "
          f"{len(M)} frozen matrix (X) + fluid (H, ~{len(frames[-1])})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames", choices=["last", "all"], default="all")
    ap.add_argument("-o", "--out", default="qa_combined.extxyz")
    a = ap.parse_args()
    main(a.run_dir, a.frames, a.out)
