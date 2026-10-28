"""Time several run configs back to back, in ONE process, on ONE core.

Wall times from separate jobs carry whatever else was on the node. Running every
config here, one after the other, takes that out: what is left is the code.

    taskset -c 0 python scripts/bench_dynamics.py --start-dir equilibrated/lj/T_1.5/box_20 \
        --mu 1.0 -6.0 params/lj_channel_*_T1_5.json params/lj_branch_*_T1_5.json

Prints ms per MC step (per exchange step; translations are extra on every
engine) and, for the ladder engines, the mean pool size and body length so two
runs of the "same" config can be checked against each other.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcmc import config_specs  # noqa: E402
from gcmc.engine import glauber_dynamics, metropolis_dynamics  # noqa: E402
from gcmc.run import sim_setup  # noqa: E402
from gcmc.system.state import ChainState  # noqa: E402


def load_xyz(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2 : 2 + n]])


def start_file(start_dir: Path, mu: float) -> Path:
    hits = sorted(start_dir.glob(f"mu_{mu:.1f}_*.xyz"))
    if not hits:
        raise SystemExit(f"no start for mu={mu} in {start_dir}")
    return hits[0]


def build(params: dict, mu: float, window: int):
    p = copy.deepcopy(params)
    p["ensemble"]["mu"] = float(mu)
    for k in ("n_cores", "initial_config", "traj_every", "checkpoint_every", "target_cycles"):
        p["run"].pop(k, None)
    p["run"]["sampling_window"] = window
    return sim_setup.from_params(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--start-dir", type=Path, required=True)
    ap.add_argument("--mu", type=float, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=200, help="exchange steps per timed window")
    ap.add_argument("--windows", type=int, default=3, help="timed windows; the median is reported")
    a = ap.parse_args()

    for mu in a.mu:
        pos0 = load_xyz(start_file(a.start_dir, mu))
        for path in a.configs:
            cfg = build(json.loads(path.read_text()), mu, a.window)
            pot = cfg.build_potential()
            wgh = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
            st = ChainState.fresh(cfg.dimension, seed=1)
            st.positions = pos0.copy()
            adv = (
                metropolis_dynamics.advance
                if cfg.engine == "metropolis_dynamics"
                else glauber_dynamics.advance
            )
            ts, agg = [], {}
            for _ in range(a.windows):
                t = time.perf_counter()
                out = adv(st, cfg, pot, wgh)
                ts.append(time.perf_counter() - t)
                for k, v in out["columns"].items():
                    agg[k] = agg.get(k, 0.0) + v
            extra = ""
            # every step of a ladder window that is not a translation is a
            # window move
            moves = a.windows * a.window - agg.get("translate_tried", 0.0)
            if "ladder_sum_pool" in agg and moves:
                extra = f"  pool M {agg['ladder_sum_pool'] / moves:.1f}"
                if "ladder_sum_length" in agg:
                    extra += f"  body len {agg['ladder_sum_length'] / moves:.2f}"
            print(
                f"mu={mu:5.1f}  {path.stem:32s} {1e3 * np.median(ts) / a.window:8.2f} ms/step"
                f"  N={out['N']}{extra}",
                flush=True,
            )


if __name__ == "__main__":
    main()
