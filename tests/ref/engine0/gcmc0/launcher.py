"""Launcher: a Cartesian sweep over chemical potential and the cavity trial
knobs, with a hard core budget.

Sweepable axes and where they live in the base params:
    ensemble.mu                 (single value or a list; muVT)
    ensemble.packing_fraction   (single value or a list; NVT)
    ensemble.N                  (single value or a list; NVT)
    sampler.trial_density
    sampler.trial_box_side

Any axis given as a JSON list is swept; scalars are fixed. The launcher takes
the Cartesian product of all swept axes, so two list axes build a heatmap
(e.g. 4 trial_density x 4 trial_box_side = 16 runs). Each combination becomes a
concrete params.json with scalars in place, a distinct id, and its own run
directory, so combinations never collide and restart is just re-running.

Core budget (required, --cores N, or run.n_cores in the base):
    cores_per_run = run.n_workers   if run.kind == "parallel"   else 1
    total_needed  = (number of combinations) * cores_per_run
The launcher refuses to start if total_needed > N, then runs at most
floor(N / cores_per_run) combinations concurrently so the node is never
oversubscribed; the rest start as earlier ones finish.

Examples (serial unless noted):
    1 mu, 1 density, 1 box        -> 1  * 1  = 1 core
    16 mu                         -> 16 * 1  = 16 cores
    4 density x 4 box             -> 16 * 1  = 16 cores
    1 mu, parallel n_workers=30   -> 1  * 30 = 30 cores

    python -m gcmc.launcher base.json --cores 16 --root runs
"""

from __future__ import annotations
import argparse
import copy
import itertools
import json
import os
import shutil
import subprocess
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# (axis label, params section, key). Order fixes the product / id ordering.
_AXES = (("mu", "ensemble", "mu"),
         ("packing_fraction", "ensemble", "packing_fraction"),
         ("N", "ensemble", "N"),
         ("target_N", "ensemble", "target_N"),
         ("target_eta", "ensemble", "target_eta"),
         ("target_density", "ensemble", "target_density"),
         ("depth", "sampler", "depth"),
         ("seed", "run", "seed"),
         ("trial_density", "sampler", "trial_density"),
         ("trial_box_side", "sampler", "trial_box_side"),
         ("channel_length", "sampler", "channel_length"),
         ("channel_width", "sampler", "channel_width"))

# nested pore axes (ensemble.pore.{H,R}); handled separately since they live one
# level deeper than the flat _AXES entries.
_PORE_AXES = (("poreH", "H"), ("poreR", "R"))


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _matrix_files_in(folder):
    """A matrix folder (e.g. an NVT runs root) holds one subdirectory per
    realization, each with a trajectory.extxyz (the frozen matrix; its last
    frame is used). Return the per-realization trajectory paths, sorted for a
    deterministic pairing with the swept target list. Also accepts a folder of
    bare .extxyz/.npz files."""
    import glob
    folder = os.path.abspath(folder)
    # case A: subdirectories each containing trajectory.extxyz
    subs = sorted(glob.glob(os.path.join(folder, "*", "trajectory.extxyz")))
    if subs:
        return subs
    # case B: subdirectories each containing checkpoint.npz
    subs = sorted(glob.glob(os.path.join(folder, "*", "checkpoint.npz")))
    if subs:
        return subs
    # case C: a flat folder of matrix files
    flat = sorted(glob.glob(os.path.join(folder, "*.extxyz")) +
                  glob.glob(os.path.join(folder, "*.npz")))
    return flat


def expand_sweep(base):
    """Return (combos, labels): one concrete params dict per Cartesian
    combination of the swept axes, with the matching short id suffixes. A
    scalar axis contributes one value; a list axis is swept; absent axes are
    not varied."""
    present, grids = [], []
    for label, section, key in _AXES:
        sec = base.get(section, {})
        if key in sec:
            present.append((label, section, key))
            grids.append(_as_list(sec[key]))
    pore = base.get("ensemble", {}).get("pore", {})
    for label, key in _PORE_AXES:
        if key in pore:
            present.append((label, "ensemble.pore", key))   # nested marker
            grids.append(_as_list(pore[key]))
    if not present:                       # nothing sweepable present: one run
        return [copy.deepcopy(base)], ["single"]

    # only axes with more than one distinct value are "swept"; those name the
    # run dirs. single-valued axes are still applied but kept out of the label.
    varying = {i for i, g in enumerate(grids) if len({repr(x) for x in g}) > 1}
    combos, labels = [], []
    for values in itertools.product(*grids):
        p = copy.deepcopy(base)
        parts = []
        for i, ((label, section, key), val) in enumerate(zip(present, values)):
            if section == "ensemble.pore":
                p.setdefault("ensemble", {}).setdefault("pore", {})[key] = val
            else:
                p.setdefault(section, {})[key] = val
            if i in varying and section != "ensemble.pore":
                parts.append(f"{label}{val}")   # pore size is in the dir prefix
        lbl = "_".join(parts)
        if lbl:
            p.setdefault("run", {})["sweep_label"] = lbl
        combos.append(p)
        labels.append(lbl or "single")

    # matrix-folder pairing: if run.matrix_config is a FOLDER (not a file), pair
    # each swept run with one realization matrix from that folder, by index --
    # combo[i] uses matrix[i] (target[i] <-> matrix[i]). This makes a 16-target
    # sweep pick up 16 matrices and run them one-to-one.
    mcfg = base.get("run", {}).get("matrix_config")
    if mcfg is not None and os.path.isdir(mcfg):
        files = _matrix_files_in(mcfg)
        if not files:
            raise ValueError(
                f"run.matrix_config is a folder ({mcfg}) but no matrix files "
                f"were found in it (expected */trajectory.extxyz, */checkpoint.npz, "
                f"or *.extxyz/*.npz).")
        if len(files) < len(combos):
            raise ValueError(
                f"sweep needs {len(combos)} matrices (one per swept value) but "
                f"the folder {mcfg} has only {len(files)}. Provide at least as "
                f"many realizations as swept values, or shorten the sweep.")
        for i, p in enumerate(combos):
            p.setdefault("run", {})["matrix_config"] = files[i]
            # tag the realization in the label so each run dir is identifiable
            real_id = os.path.basename(os.path.dirname(files[i])) or os.path.basename(files[i])
            labels[i] = f"{labels[i]}_m{i:02d}" if labels[i] != "single" else f"m{i:02d}"

    return combos, labels


def cores_per_run(base):
    run = base.get("run", {})
    if run.get("kind") == "parallel":
        nw = run.get("n_workers")
        if not isinstance(nw, int) or nw < 1:
            raise ValueError("parallel runs require integer run.n_workers >= 1")
        return nw
    return 1


def plan(base, cores):
    """Validate the budget; return (combos, labels, per_run, concurrency)."""
    combos, labels = expand_sweep(base)
    per = cores_per_run(base)
    total = len(combos) * per
    if total > cores:
        raise ValueError(
            f"core budget exceeded: {len(combos)} run(s) x {per} core(s) each "
            f"= {total} cores needed, but only {cores} available. Reduce the "
            f"sweep, lower run.n_workers, or request more cores.")
    return combos, labels, per, max(1, cores // per)


def launch_sweep(base, cores, *, root="runs", launch_dir="launch", threads=1,
                 pin=True, wait=True):
    combos, labels, per, concurrency = plan(base, cores)
    os.makedirs(launch_dir, exist_ok=True)
    # serial runs take one core and can be pinned; parallel runs manage their
    # own workers, so they are not taskset-pinned
    taskset = shutil.which("taskset") if (pin and per == 1) else None

    next_core = [0]

    def start(params, label):
        pf = os.path.join(launch_dir, f"params_{label}.json")
        with open(pf, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        env = dict(os.environ)
        for v in _THREAD_VARS:
            env[v] = str(threads)
        cmd = [sys.executable, "-m", "gcmc", "run", pf, "--root", root]
        if taskset is not None:
            # prefix taskset, but fall back to unpinned if affinity is refused
            # (e.g. restricted cgroups): pinning is an optimisation, not a
            # correctness requirement, so a node that forbids it must not fail
            core = str(next_core[0] % cores)
            next_core[0] += 1
            probe = subprocess.run([taskset, "-c", core, "true"],
                                   capture_output=True)
            if probe.returncode == 0:
                cmd = [taskset, "-c", core] + cmd
        return subprocess.Popen(cmd, env=env)

    pending = list(zip(combos, labels))
    if not wait:
        return [start(p, l) for p, l in pending]

    done, running, used = [], [], 0
    while pending or running:
        while pending and used + per <= cores and len(running) < concurrency:
            p, l = pending.pop(0)
            running.append((start(p, l), per))
            used += per
        pr, c = running.pop(0)
        done.append(pr.wait())
        used -= c
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gcmc.launcher")
    ap.add_argument("base", help="base params.json; sweepable axes may be lists")
    ap.add_argument("--cores", type=int, default=None,
                    help="cores available this job (required; or run.n_cores)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--launch-dir", default="launch")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without launching")
    args = ap.parse_args(argv)

    with open(args.base) as f:
        base = json.load(f)
    cores = args.cores
    if cores is None:
        cores = base.get("run", {}).get("n_cores")
    if not isinstance(cores, int) or cores < 1:
        ap.error("cores is required: pass --cores N or set run.n_cores in the base")

    combos, labels, per, concurrency = plan(base, cores)
    print(f"{len(combos)} run(s) x {per} core(s) = {len(combos) * per} of "
          f"{cores} cores; up to {concurrency} concurrent")
    if args.dry_run:
        for l in labels:
            print("  ", l)
        return

    result = launch_sweep(base, cores, root=args.root, launch_dir=args.launch_dir,
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
