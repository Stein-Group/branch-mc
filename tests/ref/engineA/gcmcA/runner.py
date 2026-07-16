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
import math
import os
import socket
import time
import numpy as np

from . import sim_setup, hash_fingerprint, io, MC_engine
from .state import ChainState, zero_counters

# One fixed width per column, shared by the header and every data row, so the
# labels sit directly above their values and the log reads as a clean table.
_LOG_COLS = [
    ("MC_cycle(M)", 13), ("pckng_frctn", 13), ("N", 7),
    ("add_acc", 9), ("add_att", 9), ("add_stp", 9),
    ("rm_acc", 9), ("rm_stp", 9), ("mv_acc", 9), ("mv_stp", 9),
    ("wall_s", 11), ("mu", 12), ("avgN", 11), ("avg_n_cav", 12),
    ("density", 14),
]
_LOG_WIDTHS = [w for _, w in _LOG_COLS]


def _log_header():
    h = "".join(name.rjust(w) for name, w in _LOG_COLS)
    return "#" + h[1:]          # mark as a comment without shifting the columns


def _log_row(cells):
    return "".join(str(c).rjust(w) for c, w in zip(cells, _LOG_WIDTHS))


LOG_HEADER = _log_header()



def _inside_pore_seed(pt, cfg):
    """Is a single point inside the hard pore? The centre clearance is the
    configurable sphere radius cfg.radius, not a hardcoded 0.5."""
    L = cfg.box
    cl = cfg.radius
    if cfg.pore_geometry == "slit":
        z0 = (L - cfg.pore_H) / 2.0
        return z0 + cl <= pt[2] <= z0 + cfg.pore_H - cl
    cx = cy = L / 2.0
    import math as _m
    return _m.hypot(pt[0] - cx, pt[1] - cy) <= cfg.pore_R - cl


def _save_pore_geometry(cfg, directory):
    """Write the wall geometry as an extxyz of marker points so the pore can be
    loaded next to the fluid trajectory in OVITO. Slit: two wall planes tiled
    with markers; cylinder: the lateral surface tiled in (theta, z). Markers use
    element 'W' (wall) so they are visually distinct from the fluid."""
    import numpy as _np, os as _os
    L = cfg.box
    pts = []
    n = 40
    if cfg.pore_geometry == "slit":
        z0 = (L - cfg.pore_H) / 2.0
        xs = _np.linspace(0, L, n); ys = _np.linspace(0, L, n)
        for zc in (z0, z0 + cfg.pore_H):                 # both wall surfaces
            for x in xs:
                for y in ys:
                    pts.append((x, y, zc))
    else:
        R = cfg.pore_R; cx = cy = L / 2.0
        th = _np.linspace(0, 2 * _np.pi, n, endpoint=False)
        zs = _np.linspace(0, L, n)
        for z in zs:
            for t in th:
                pts.append((cx + R * _np.cos(t), cy + R * _np.sin(t), z))
    pts = _np.asarray(pts, float)
    path = _os.path.join(directory, "pore_geometry.extxyz")
    with open(path, "w") as f:
        f.write(f"{len(pts)}\n")
        f.write(f'Lattice="{L} 0 0 0 {L} 0 0 0 {L}" '
                f'Properties=species:S:1:pos:R:3 geometry={cfg.pore_geometry}\n')
        for p in pts:
            f.write(f"W {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    return path


def _init_config(cfg, rng):
    # start from a pre-equilibrated configuration when one is given, else one
    # seed molecule, as the original starts (insertion needs an occupied
    # cavity, so an empty system never grows).
    #
    # initial_config is either a directory or a file, decided by whether it has
    # a file extension (not by os.path.isdir, which silently mis-resolves when a
    # job's working directory differs from where it was launched). A directory is
    # the equilibrated library, nested by system, temperature (LJ only) and box:
    #     equilibrated/lj/T_0.85/box_20/mu_0.0_lj_3d_box20.xyz
    #     equilibrated/hard_disk_2d/box_200/mu_1.00_hard_2d_box_200.xyz
    #     equilibrated/NVT/box_20/...           matrix/ is loaded via matrix_config
    # The per-run file is picked by matching chemical potential, box, dimension
    # and (for LJ) temperature, so every mu in a sweep loads its own start. Use an
    # absolute path to be safe across SLURM / subprocess working directories.
    spec = cfg.initial_config
    if not spec:
        if getattr(cfg, "pore_geometry", None) is not None:
            # confined pore: seed a few non-overlapping particles INSIDE the pore
            # so the depth>=1 proposal has an anchor (it cannot grow from N=0).
            from scipy.spatial import cKDTree as _KD
            box = cfg.box; dim = cfg.dimension
            placed = []
            for _ in range(20 * 300):
                if len(placed) >= 20:
                    break
                pt = rng.uniform(0, box, dim)
                if not _inside_pore_seed(pt, cfg):
                    continue
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * dim)
                    if tp.query(pt % box, k=1)[0] < cfg.diameter:
                        continue
                placed.append(pt)
            return np.array(placed) if placed else np.empty((0, dim))
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
        # Bulk muVT with no initial_config: start EMPTY. Insertion centres on a
        # uniform random point and needs nothing present, so N=0 -> 1 grows on
        # its own; removal and displacement only ever act on existing particles.
        # No lattice: this method places trials at uniform random points, so a
        # gridded start would be foreign to it. Provide initial_config only to
        # start from a pre-equilibrated structure (e.g. the metastable liquid).
        return np.empty((0, cfg.dimension))

    if os.path.splitext(spec)[1]:                      # has an extension -> file
        path = os.path.abspath(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"initial_config file not found: {path}\n"
                f"  (resolved from cwd={os.getcwd()}). Use an absolute path if the "
                f"run is launched from a different directory than the config.")
        pos = io.load_config_file(path, cfg.dimension)
        print(f"loaded initial config: {len(pos)} molecules from {path}")
        return pos

    # A directory points at the equilibrated library, which is nested by system,
    # (temperature for LJ) and box, with per-mu files whose decimal formatting
    # differs between families, e.g.
    #     equilibrated/lj/T_0.85/box_20/mu_0.0_lj_3d_box20.xyz
    #     equilibrated/hard_disk_2d/box_200/mu_1.00_hard_2d_box_200.xyz
    #     equilibrated/NVT/box_20/mu_..._box_20.xyz
    # Rather than reconstruct an exact filename (the formats are inconsistent),
    # search the tree and match on the values parsed from each candidate: chemical
    # potential and box always, dimension and (for LJ) temperature when present.
    import glob as _glob
    import re as _re
    root = os.path.abspath(spec)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"initial_config directory not found: {root} (resolved from "
            f"cwd={os.getcwd()}). Point it at the equilibrated library folder or "
            f"at a single .xyz file.")
    is_lj = cfg.potential_type in ("lj", "ljbranch", "ljaux")
    all_xyz = _glob.glob(os.path.join(root, "**", "*.xyz"), recursive=True)
    if not all_xyz:
        raise FileNotFoundError(
            f"no .xyz configurations found anywhere under initial_config directory "
            f"{root}. Expected files like mu_<mu>_<family>_<dim>d_box<L>.xyz.")

    def _num(pattern, text):
        m = _re.search(pattern, text)
        return float(m.group(1)) if m else None

    matches = []
    for f in all_xyz:
        name = os.path.basename(f)
        f_mu = _num(r"mu_(-?\d+(?:\.\d+)?)", name)
        f_box = _num(r"box_?(\d+(?:\.\d+)?)", f)       # from the path (box_<L> dir)
        f_dim = _num(r"(\d)d", name)
        f_T = _num(r"[/\\]T_(\d+(?:\.\d+)?)[/\\]", f)
        if f_mu is None or abs(f_mu - cfg.chem_pot) > 1e-6:
            continue
        if f_box is not None and abs(f_box - float(cfg.box)) > 1e-6:
            continue
        if f_dim is not None and int(f_dim) != cfg.dimension:
            continue
        if is_lj and f_T is not None and abs(f_T - float(cfg.temp)) > 1e-6:
            continue
        matches.append(f)

    if len(matches) == 0:
        sample = sorted(os.path.relpath(f, root) for f in all_xyz)[:12]
        raise FileNotFoundError(
            f"no equilibrated config under {root} matches this run "
            f"(mu={cfg.chem_pot}, box={cfg.box}, dimension={cfg.dimension}"
            + (f", T={cfg.temp}" if is_lj else "") + "). "
            f"{len(all_xyz)} .xyz files are present, for example: {sample}. "
            f"Check that a matching mu/box (and T for LJ) exists, or point "
            f"initial_config at the exact file.")
    if len(matches) > 1:
        rels = sorted(os.path.relpath(m, root) for m in matches)[:8]
        raise ValueError(
            f"initial_config is ambiguous: {len(matches)} files under {root} match "
            f"mu={cfg.chem_pot}, box={cfg.box}"
            + (f", T={cfg.temp}" if is_lj else "") + f": {rels}. Point "
            f"initial_config at a more specific subdirectory or at the exact file.")

    path = matches[0]
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
                       params=params, mu=state.mu, n_history=state.n_history)


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
        if os.path.isdir(mpath):
            raise IsADirectoryError(
                f"run.matrix_config is a folder ({mpath}). A folder of matrix "
                f"realizations is only valid as a SWEEP: give a list target "
                f"(target_N/target_eta/target_density) and launch with "
                f"`python -m gcmc.launcher`, which pairs each swept value with one "
                f"matrix from the folder. For a single run, point matrix_config "
                f"at one trajectory.extxyz / checkpoint.npz file.")
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
        if getattr(cfg, "pore_geometry", None) is not None:
            _save_pore_geometry(cfg, paths.directory)
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
                               rng=gen, counters=dict(ckpt["counters"]),
                               mu=ckpt.get("mu"),
                               n_history=list(ckpt.get("n_history", [])))
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

        # target-mu control (muVT only): drive mu so the running-mean N hits
        # cfg.target_count. state.mu carries the live mu (init from cfg on a
        # fresh run; preserved across resume). state.n_history is the rolling
        # window of N feeding the running mean the controller steers.
        targeting = cfg.target_count is not None
        if targeting and state.mu is None:
            state.mu = cfg.chem_pot
        V = cfg.total_volume

        win = 0
        while state.step < cfg.target_cycles:
            _t = time.perf_counter()
            stats = advance(state, cfg, potential, weigher)
            win_wall = time.perf_counter() - _t
            for k, v in stats["counters"].items():
                state.counters[k] += v

            # running mean of N over the control window
            state.n_history.append(stats["N"])
            if len(state.n_history) > cfg.control_window:
                state.n_history = state.n_history[-cfg.control_window:]
            avg_n = sum(state.n_history) / len(state.n_history)

            # running mean of cavity occupancy over the same window (diagnostic)
            state.ncav_history.append(stats.get("n_cav", 0.0))
            if len(state.ncav_history) > cfg.control_window:
                state.ncav_history = state.ncav_history[-cfg.control_window:]
            avg_ncav = sum(state.ncav_history) / len(state.ncav_history)

            if targeting:
                # Robbins-Monro update on log-N: mu += gamma * (lnN_target - lnN).
                # Gain decays as gamma0/(1+win/tau) so mu converges and settles.
                gamma = cfg.control_gain / (1.0 + win / cfg.control_gain_tau)
                avg_safe = max(avg_n, 0.5)         # guard log at near-empty
                state.mu += gamma * (math.log(cfg.target_count) - math.log(avg_safe))

            cur_mu = state.mu if state.mu is not None else cfg.chem_pot

            if state.step >= cfg.burn_in:
                cc = stats["counters"]
                log.write(_log_row([
                    f"{state.step/1e6:.6f}", f"{stats['packing_fraction']:.8f}",
                    stats['N'], cc['add_accept'], cc['add_attempt'],
                    cc['add_step'], cc['remove_accept'], cc['remove_step'],
                    cc['move_accept'], cc['move_step'], f"{win_wall:.4f}",
                    f"{cur_mu:.6f}", f"{avg_n:.4f}", f"{avg_ncav:.4f}",
                    f"{avg_n / V:.8f}"]))

            win += 1
            if win % cfg.checkpoint_every == 0:
                _save(paths, cfg, state, params)
            if cfg.traj_every >= 1 and win % cfg.traj_every == 0:
                info["step"] = state.step
                info["mu"] = cur_mu
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
