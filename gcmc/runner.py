"""Runner: the one loop that makes a params.json runnable.

Fresh and restart are the same path. Compute the run id, look for a checkpoint,
load it if present or initialise if not, then run `while step < target`. A
finished run re-entered is a no-op, so requeues and array jobs are idempotent.
The single generator's state is checkpointed, so a restart continues the exact
Markov chain rather than reseeding.

A lockfile (pid, host, time) in the run directory stops two jobs resuming the
same id; a stale lock past the timeout is taken over.
"""

from __future__ import annotations
import json
import os
import socket
import time
import numpy as np

from . import config, identity, io, engine
from .state import ChainState, zero_counters

LOG_HEADER = ("#MC_cycle(M)  pckng_frctn  N  add_acc add_att add_stp "
              "rm_acc rm_stp mv_acc mv_stp")


def _init_config(cfg, rng):
    # start from a pre-equilibrated configuration when one is given, else one
    # seed molecule, as the original starts (insertion needs an occupied
    # cavity, so an empty system never grows).
    #
    # initial_config is either a directory or a file, decided by whether it has
    # a file extension (not by os.path.isdir, which silently mis-resolves when a
    # job's working directory differs from where it was launched). A directory
    # selects the file per run by chemical potential, system and box:
    #     <dir>/<subfolder>/mu_<mu.2f>_<family>_<dim>d_box<L>.xyz
    # e.g. equilibrated/hard_disk_3d/mu_3.25_hard_3d_box20.xyz
    # so every mu in a sweep loads its own start. Use an absolute path to be
    # safe across SLURM / subprocess working directories.
    spec = cfg.initial_config
    if not spec:
        return rng.uniform(0, cfg.box, size=(1, cfg.dimension))

    if os.path.splitext(spec)[1]:                      # has an extension -> file
        path = spec
    else:                                              # directory -> build path
        lj = cfg.potential_type == "lj"
        family = "lj" if lj else "hard"
        subfolder = "lj" if lj else f"hard_disk_{cfg.dimension}d"
        b = int(cfg.box) if float(cfg.box).is_integer() else cfg.box
        fname = f"mu_{cfg.chem_pot:.2f}_{family}_{cfg.dimension}d_box{b}.xyz"
        path = os.path.join(spec, subfolder, fname)

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"initial_config not found: {path}\n"
            f"  (resolved from cwd={os.getcwd()}). If runs are launched from a "
            f"different directory than the config tree, set execution."
            f"initial_config to an absolute path.")
    pos = io.load_config_file(path, cfg.dimension)
    print(f"loaded initial config: {len(pos)} molecules from {path}")
    return pos


def acquire_lock(paths, timeout=3600):
    if os.path.exists(paths.lock):
        age = time.time() - os.path.getmtime(paths.lock)
        if age < timeout:
            raise RuntimeError(f"run is locked ({age:.0f}s old): "
                               f"{open(paths.lock).read().strip()}")
    with open(paths.lock, "w") as f:
        f.write(f"{os.getpid()} {socket.gethostname()} {time.time():.0f}")


def release_lock(paths):
    try:
        os.remove(paths.lock)
    except FileNotFoundError:
        pass


def _save(paths, cfg, state, params):
    io.save_checkpoint(paths, positions=state.positions, step=state.step,
                       rng=io.rng_state(state.rng), counters=state.counters,
                       params=params)


def run(params, root="runs"):
    if isinstance(params, str):
        with open(params) as f:
            params = json.load(f)

    cfg = config.from_params(params)
    paths = identity.run_paths(cfg, root)
    io.ensure_run_dir(paths, params)
    acquire_lock(paths)

    try:
        ckpt = io.load_checkpoint(paths)
        if ckpt is None:
            seed = params.get("execution", {}).get("seed")
            state = ChainState.fresh(cfg.dimension, seed=seed)
            state.positions = _init_config(cfg, state.rng)
        else:
            gen = io.set_rng_state(np.random.default_rng(), ckpt["rng"])
            state = ChainState(positions=ckpt["positions"], step=int(ckpt["step"]),
                               rng=gen, counters=dict(ckpt["counters"]))
            io.trim_log(paths.log, state.step / 1e6)

        potential = cfg.build_potential()
        weigher = engine.make_weigher(cfg, potential)
        if cfg.mode == "parallel":
            from . import parallel
            advance = parallel.advance_parallel
        else:
            advance = engine.advance
        log = io.Log(paths.log, LOG_HEADER, flush_every=1)
        info = {"step": state.step, "mu": cfg.chem_pot,
                "potential": cfg.potential_type, "box": cfg.box}

        win = 0
        while state.step < cfg.target_cycles:
            stats = advance(state, cfg, potential, weigher)
            for k, v in stats["counters"].items():
                state.counters[k] += v

            if state.step >= cfg.burn_in:
                cc = stats["counters"]
                log.write(f"{state.step/1e6:.6f}  {stats['packing_fraction']:.8f}  "
                          f"{stats['N']}  {cc['add_accept']} {cc['add_attempt']} "
                          f"{cc['add_step']} {cc['remove_accept']} {cc['remove_step']} "
                          f"{cc['move_accept']} {cc['move_step']}")

            win += 1
            if win % cfg.checkpoint_every == 0:
                _save(paths, cfg, state, params)
            if win % cfg.traj_every == 0:
                info["step"] = state.step
                io.write_frame(paths, state.positions, two_dim=cfg.two_dim,
                               box=cfg.box, info=info, append=cfg.full_trajectory)

        log.close()
        _save(paths, cfg, state, params)
        info["step"] = state.step
        io.write_frame(paths, state.positions, two_dim=cfg.two_dim, box=cfg.box,
                       info=info, append=cfg.full_trajectory)
        return state, paths
    finally:
        release_lock(paths)
