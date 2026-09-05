"""ms/step for the two production channel modes at three system sizes.

    python tests/bench_neighbors.py                    # print + write timing.json
    python tests/bench_neighbors.py --out ref/baseline_timing.json
    python tests/bench_neighbors.py --quick            # short windows, for smoke runs

A "step" is one exchange step (the window counter); translations are
interspersed and not counted, exactly as `sampling_window` defines it. Each
configuration is lattice-seeded at the target N and given warmup windows before
the measured ones, so the tree-build cost is measured at the production density
rather than at whatever an unequilibrated start happens to give.

Both modes are muVT. Hard spheres run at the Carnahan-Starling mu for the seeded
density so N is near-stationary across the measurement; LJ runs at a fixed mu and
drifts a little, which is reproducible and therefore fine for a before/after
comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H

from gcmc.engine import MC_engine

# (label, box, N)
SIZES = [("small", 12.0, 700), ("medium", 20.0, 4000), ("large", 30.0, 15000)]
LJ_RHO = 0.6


def hs_channel_params(box, n, window):
    return H._p(
        H._HS,
        {
            "type": "channel",
            "depth": 1,
            "trial_density": 3.0,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        {"mu": H.cs_mu(n / box**3), "dimension": 3, "box": box},
        {"sampling_window": window},
    )


def lj_full_params(box, n, window):
    return H._p(
        H._LJ,
        {
            "type": "channel",
            "depth": 1,
            "trial_density": 0.6,
            "channel_length": "full",
            "channel_width": 1.2,
        },
        {"mu": -3.2, "dimension": 3, "box": box},
        {"sampling_window": window},
    )


MODES = {"hs_channel": hs_channel_params, "lj_full_span": lj_full_params}


def measure(make_params, box, n, window, warmup, repeat, seed=7, engine=None):
    rng = np.random.default_rng(seed)
    pos = H.lattice_seed(box, n, 3, rng)
    eng = None if engine in (None, "current") else H.load_engine(engine)
    advance = (MC_engine if eng is None else eng[1]).advance
    cfg, pot, wgh, st = H.build(
        make_params(box, n, window), seed, positions=pos, engine=eng
    )
    for _ in range(warmup):
        advance(st, cfg, pot, wgh)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        advance(st, cfg, pot, wgh)
        times.append((time.perf_counter() - t0) * 1000.0 / window)
    return {
        "box": box,
        "n_seed": n,
        "n_final": len(st.positions),
        "window": window,
        "ms_per_step": float(np.mean(times)),
        "ms_per_step_sd": float(np.std(times)),
        "samples": times,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="one mode name")
    ap.add_argument(
        "--engines", default="current", help="comma-separated: current,stageA,baseline"
    )
    a = ap.parse_args(argv)

    window, warmup, repeat = (150, 1, 2) if a.quick else (500, 2, 3)
    modes = {a.only: MODES[a.only]} if a.only else MODES
    engines = [e.strip() for e in a.engines.split(",")]

    out = {}
    for mode, fn in modes.items():
        out[mode] = {}
        for label, box, n in SIZES:
            # LJ is benchmarked at its own density, not the hard-sphere one
            nn = int(round(LJ_RHO * box**3)) if mode == "lj_full_span" else n
            row = {}
            for e in engines:
                # interleaved, not one engine after the other: the machine drifts
                row[e] = measure(fn, box, nn, window, warmup, repeat, engine=e)
            out[mode][label] = row[engines[0]] if len(engines) == 1 else row
            cells = "  ".join(f"{e}={row[e]['ms_per_step']:7.3f}" for e in engines)
            speed = ""
            if "baseline" in row and "current" in row:
                speed = (
                    f"   speedup "
                    f"{row['baseline']['ms_per_step'] / row['current']['ms_per_step']:.2f}x"
                )
            print(
                f"{mode:14s} {label:7s} box={box:4.0f} "
                f"N={row[engines[0]]['n_final']:6d}  {cells} ms/step{speed}"
            )

    if a.out:
        path = (
            a.out
            if os.path.isabs(a.out)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
