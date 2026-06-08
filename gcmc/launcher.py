"""Launcher for the single-core sweep, the thin replacement for wrapper.py.

It owns only the sweep axis (a list of mu values) and the core assignment. For
each mu it writes a concrete params.json (the base with that mu substituted) and
launches one `python -m gcmc run` pinned to a core, with the BLAS thread count
forced to one so sixteen processes on a node do not oversubscribe. Each run has
a distinct mu, hence a distinct id and its own directory, so the sixteen never
collide and restart is just re-running the launcher.

    python -m gcmc.launcher base.json --mus -3.0 -3.25 -3.5 --root runs

The mu list may instead live in the base file under a top-level "sweep": {"mu":
[...]}, which is stripped before each run since it is not part of a run's params.
Core pinning uses `taskset -c <core>` as a command prefix, which is race-free,
and is skipped automatically if taskset is unavailable or --no-pin is given.
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _params_for(base, mu):
    p = copy.deepcopy(base)
    p.pop("sweep", None)
    p.setdefault("ensemble", {})["mu"] = float(mu)
    return p


def launch_sweep(base, mus, *, root="runs", launch_dir="launch", threads=1,
                 pin=True, cores=None, wait=True):
    """Launch one single-core run per mu. Returns the list of return codes if
    `wait`, else the list of Popen handles."""
    os.makedirs(launch_dir, exist_ok=True)
    taskset = shutil.which("taskset") if pin else None

    procs = []
    for i, mu in enumerate(mus):
        params = _params_for(base, mu)
        pf = os.path.join(launch_dir, f"params_mu_{mu}.json")
        with open(pf, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)

        env = dict(os.environ)
        for v in _THREAD_VARS:
            env[v] = str(threads)

        cmd = [sys.executable, "-m", "gcmc", "run", pf, "--root", root]
        if taskset is not None:
            core = cores[i] if cores else i
            cmd = [taskset, "-c", str(core)] + cmd
        procs.append(subprocess.Popen(cmd, env=env))

    if wait:
        return [pr.wait() for pr in procs]
    return procs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gcmc.launcher")
    ap.add_argument("base", help="base params.json (mu is supplied by the sweep)")
    ap.add_argument("--mus", nargs="+", type=float, default=None,
                    help="mu values; if omitted, read from base['sweep']['mu']")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--launch-dir", default="launch")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-pin", action="store_true", help="do not pin cores")
    ap.add_argument("--no-wait", action="store_true", help="launch and return")
    args = ap.parse_args(argv)

    with open(args.base) as f:
        base = json.load(f)
    mus = args.mus if args.mus is not None else base.get("sweep", {}).get("mu")
    if not mus:
        ap.error("no mu values: pass --mus or set base['sweep']['mu']")

    result = launch_sweep(base, mus, root=args.root, launch_dir=args.launch_dir,
                          threads=args.threads, pin=not args.no_pin,
                          wait=not args.no_wait)
    if args.no_wait:
        print(f"launched {len(result)} runs")
    else:
        print(f"finished {len(result)} runs, return codes {result}")
        if any(result):
            sys.exit(1)


if __name__ == "__main__":
    main()
