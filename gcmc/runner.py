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
import itertools
import os
import socket
import time
import numpy as np

from . import sim_setup, hash_fingerprint, io, MC_engine
from .state import ChainState, zero_counters

LOG_HEADER = ("#MC_cycle(M)  pckng_frctn  N  add_acc add_att add_stp "
              "rm_acc rm_stp mv_acc mv_stp  wall_s")


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
        mpos = getattr(cfg, "matrix_positions", None)
        if mpos is not None and len(mpos) > 0 and getattr(cfg, "target_n", None) is None:
            # quenched-annealed muVT: seed a few valid fluid particles (the
            # depth>=1 proposal needs an anchor and cannot grow from N=0), each
            # clearing the matrix and the others. GCMC then fills the pores.
            from scipy.spatial import cKDTree as _KD
            box = cfg.box
            tree_m = _KD(np.asarray(mpos, float) % box, boxsize=[box] * cfg.dimension)
            placed = []
            target_seed = 20            # small; enough to bootstrap insertion
            for _ in range(target_seed * 200):
                if len(placed) >= target_seed:
                    break
                pt = rng.uniform(0, box, cfg.dimension)
                if tree_m.query(pt % box, k=1)[0] < cfg.diameter:
                    continue            # overlaps matrix
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * cfg.dimension)
                    if tp.query(pt % box, k=1)[0] < cfg.diameter:
                        continue        # overlaps an already-placed fluid sphere
                placed.append(pt)
            return (np.array(placed) if placed else np.empty((0, cfg.dimension)))
        if getattr(cfg, "target_n", None) is not None:
            # NVT: seed EXACTLY target_n non-overlapping particles on a grid.
            # Repacking conserves N, so the starting count must be the target;
            # the configuration then equilibrates in place from this lattice.
            n = cfg.target_n
            if n == 0:
                return np.empty((0, cfg.dimension))
            # smallest cubic grid with at least n valid sites, spacing >= diameter
            import math as _m
            per = max(2, int(_m.ceil(n ** (1.0 / cfg.dimension))))
            while True:
                step = cfg.box / per
                if step < cfg.diameter:        # too tight: would overlap
                    raise ValueError(
                        f"cannot place {n} particles in box {cfg.box} without "
                        f"overlap (grid spacing {step:.3f} < diameter "
                        f"{cfg.diameter}); reduce N/packing_fraction or enlarge box")
                g = (np.arange(per) + 0.5) * step
                pts = np.array(list(itertools.product(g, repeat=cfg.dimension)))
                if len(pts) >= n:
                    return pts[:n]
                per += 1
        if cfg.depth > 0:
            # depth >= 1 insertions need an occupied cavity and a removal at
            # N=1 can downgrade to a pure deletion, so N=0 is absorbing: a
            # near-empty start can die and never grow. Seed a hard-core-valid
            # grid at moderate density instead (the original relied on dense
            # restart files for the same reason).
            g = np.arange(0.65, cfg.box, (4.0 / 3.0) * cfg.diameter)
            g = g[g < cfg.box - 0.5 * cfg.diameter]
            pts = np.array(list(itertools.product(g, repeat=cfg.dimension)))
            return pts
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
            f"different directory than the config tree, set run."
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


def _find_matrix_snapshot(params, root, dimension):
    """On resume with the external matrix gone, locate the run directory whose
    params match and return its matrix.npz snapshot path (or None)."""
    import os, json, glob
    target = json.dumps(params, sort_keys=True)
    for snap in glob.glob(os.path.join(root, "*", "matrix.npz")):
        pj = os.path.join(os.path.dirname(snap), "params.json")
        if os.path.exists(pj):
            try:
                if json.dumps(json.load(open(pj)), sort_keys=True) == target:
                    return snap
            except Exception:
                continue
    return None


def run(params, root="runs"):
    if isinstance(params, str):
        with open(params) as f:
            params = json.load(f)

    cfg = sim_setup.from_params(params)
    if cfg.matrix_config is not None:
        import dataclasses, os
        import numpy as _np
        mpath = os.path.abspath(cfg.matrix_config)
        # On a fresh launch we load the external matrix to compute the
        # content-addressed run dir, then snapshot it inside. On a resume the
        # snapshot in the (params-determined) run dir is authoritative, so the
        # external file may be gone -- find the run dir by probing the snapshot.
        if os.path.exists(mpath):
            mpos = io.load_matrix(mpath, cfg.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            snap = os.path.join(paths.directory, "matrix.npz")
            if os.path.exists(snap):
                cfg = dataclasses.replace(
                    cfg, matrix_positions=io.load_matrix(snap, cfg.dimension))
            else:
                _np.savez(snap, positions=_np.asarray(mpos, float))
            print(f"loaded frozen matrix: {len(mpos)} obstacles from {mpath}")
        else:
            # external matrix gone: this must be a resume. Find the existing run
            # dir whose params.json matches and load its matrix snapshot.
            snap = _find_matrix_snapshot(params, root, cfg.dimension)
            if snap is None:
                raise FileNotFoundError(
                    f"matrix_config not found: {mpath}, and no existing run "
                    f"snapshot under {root!r} to resume from. Restore the matrix "
                    f"file or the run directory.")
            mpos = io.load_matrix(snap, cfg.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            print(f"resumed frozen matrix: {len(mpos)} obstacles from {snap}")
    else:
        paths = hash_fingerprint.run_paths(cfg, root)
        io.ensure_run_dir(paths, params)
    acquire_lock(paths)

    try:
        ckpt = io.load_checkpoint(paths)
        if ckpt is None:
            seed = params.get("run", {}).get("seed")
            state = ChainState.fresh(cfg.dimension, seed=seed)
            state.positions = _init_config(cfg, state.rng)
        else:
            gen = io.set_rng_state(np.random.default_rng(), ckpt["rng"])
            state = ChainState(positions=ckpt["positions"], step=int(ckpt["step"]),
                               rng=gen, counters=dict(ckpt["counters"]))
            io.trim_log(paths.log, state.step / 1e6)
            if cfg.traj_every >= 1:
                io.trim_trajectory(paths.trajectory, state.step)

        potential = cfg.build_potential()
        weigher = MC_engine.make_weigher(cfg, potential)
        if cfg.mode == "parallel":
            from . import parallel
            advance = parallel.advance_parallel
        else:
            advance = MC_engine.advance
        log = io.Log(paths.log, LOG_HEADER, flush_every=1)
        info = {"step": state.step, "mu": cfg.chem_pot,
                "potential": cfg.potential_type, "box": cfg.box}

        win = 0
        while state.step < cfg.target_cycles:
            _t = time.perf_counter()
            stats = advance(state, cfg, potential, weigher)
            win_wall = time.perf_counter() - _t
            for k, v in stats["counters"].items():
                state.counters[k] += v

            if state.step >= cfg.burn_in:
                cc = stats["counters"]
                log.write(f"{state.step/1e6:.6f}  {stats['packing_fraction']:.8f}  "
                          f"{stats['N']}  {cc['add_accept']} {cc['add_attempt']} "
                          f"{cc['add_step']} {cc['remove_accept']} {cc['remove_step']} "
                          f"{cc['move_accept']} {cc['move_step']}  {win_wall:.4f}")

            win += 1
            if win % cfg.checkpoint_every == 0:
                _save(paths, cfg, state, params)
            if cfg.traj_every >= 1 and win % cfg.traj_every == 0:
                info["step"] = state.step
                io.write_frame(paths, state.positions, two_dim=cfg.two_dim,
                               box=cfg.box, info=info, append=True)

        log.close()
        _save(paths, cfg, state, params)
        info["step"] = state.step
        if cfg.traj_every == 0:
            # save-last-frame mode: one overwritten snapshot
            io.write_frame(paths, state.positions, two_dim=cfg.two_dim,
                           box=cfg.box, info=info, append=False)
        elif win % cfg.traj_every != 0:
            # append mode: the last window was not on the cadence, so its frame
            # has not been written yet -- append it so the final state is saved
            io.write_frame(paths, state.positions, two_dim=cfg.two_dim,
                           box=cfg.box, info=info, append=True)
        return state, paths
    finally:
        release_lock(paths)
