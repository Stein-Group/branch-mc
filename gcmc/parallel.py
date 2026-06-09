"""Parallel domain-decomposition engine (2D only, matching the original).

The original ran one Monte Carlo kernel either over the whole box (serial) or
over many cells with frozen borders (parallel). The serial kernel lives in
engine.advance and is verified against the original. This module is its cell
sibling: the same move/add/remove physics and the same Rosenbluth assembly
(reused from samplers and from engine._weighted_trials), diverging only where
decomposition demands it:

  * neighbour and overlap searches see the cell's frozen border molecules,
    stacked in front of the live ones, so trials near a cell edge feel the
    neighbours that live in the next cell;
  * periodic wrapping is cell-local (apply_PBC_single for a move, apply_pbc_array
    for the trial cloud) instead of a global modulo;
  * an insertion's cavity is placed by create_cavity inside the cell, not
    anywhere in the box;
  * mu_ideal uses the cell area, not the box volume, and the molecule count is
    the cell's live molecules alone, never the borders.

Those four points are exactly the decomposition helpers verified in
check_decomposition.py. Everything between them is the verified serial physics.

Verification here is by self-consistency rather than an exact trace: the
original's parallel path draws its cavities from an unseedable default_rng, so
no seed reproduces it. Instead a parallel-2D run must reach the same equilibrium
density as the serial-2D run at the same mu, T and box, which is the physical
invariant a decomposition must preserve and which any acceptance, wrapping or
volume bug would break.

3D parallel is an explicit error in the original ("3d paralle not implemented
yet") because the helpers are 2D-only; advance_parallel raises the same way.
"""

from __future__ import annotations
import concurrent.futures
import numpy as np
from scipy.spatial import cKDTree

from . import decomposition, geometry
from .trials import TrialSet
from .engine import _weighted_trials, make_weigher
from .samplers.rosenbluth import select_forward, select_reverse
from .state import zero_counters


def advance_cell(main, border, n_steps, cell_dims, cfg, potential, weigher, rng):
    """Run n_steps of GCMC inside one cell. `main` are the live molecules
    (added/removed/moved), `border` are frozen context for neighbour searches.
    Returns (new_main_coords, counters)."""
    if not cfg.two_dim:
        raise NotImplementedError("parallel decomposition is 2D-only")

    pos = np.asarray(main, float).reshape(-1, 2)
    bord = np.asarray(border, float).reshape(-1, 2)
    c = zero_counters()

    box = cfg.box_dims
    side = cfg.box
    diameter = cfg.diameter
    r_sq = cfg.r_squared
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    n_trials = cfg.no_of_trial_molecules
    depth = cfg.depth
    gdict = cfg.gamma_dict
    beta = cfg.beta
    mu = cfg.chem_pot
    cell_area = cfg.cell_length * cfg.cell_length
    lj = cfg.potential_type == "lj"

    def with_border(arr):
        return np.vstack((bord, arr)) if len(bord) else arr

    def rand():
        return rng.random()

    step = 0
    while step <= n_steps:
        N = len(pos)
        choice = "add" if N == 0 else rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ---------------------------------------------------------- translate
        if choice == "move" and N > 0:
            c["move_step"] += 1
            j = rng.choice(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            ranf1, ranf2 = rng.uniform(0, 1, 2)
            mx = old[0] + mf * (ranf1 - 0.5)
            my = old[1] + mf * (ranf2 - 0.5)
            mx, my = decomposition.apply_PBC_single(mx, my, cell_dims, box)
            moved = np.array([mx, my])
            system = with_border(without)
            tree = cKDTree(system, boxsize=box)
            if lj:
                d_new, _ = tree.query([moved], k=nn, distance_upper_bound=uc)
                d_new = d_new[0][np.isfinite(d_new[0])]
                if not (d_new.size and np.any(d_new < diameter)):
                    d_old, _ = tree.query([old], k=nn, distance_upper_bound=uc)
                    d_old = d_old[0][np.isfinite(d_old[0])]

                    def _u(d):
                        if d.size == 0:
                            return 0.0
                        ir6 = (1.0 / d) ** 6
                        return np.sum(4.0 * ir6 * (ir6 - 1.0))
                    dU = _u(d_new) - _u(d_old)
                    if np.log(rng.uniform(0, 1)) <= -dU / cfg.temp:
                        pos = np.append(without, [moved], axis=0)
                        c["move_accept"] += 1
            else:
                if len(tree.query_ball_point([moved], diameter)[0]) == 0:
                    pos = np.append(without, [moved], axis=0)
                    c["move_accept"] += 1
            continue

        # ---------------------------------------------------------- add/remove
        if choice == "remove" and N == 0:
            continue

        if choice == "add":
            c["add_step"] += 1
            n_add, n_remove = depth + 1, depth
        else:
            c["remove_step"] += 1
            n_add, n_remove = depth, depth + 1

        step += 1
        if step >= n_steps:
            break

        # cavity centre: inside the cell for an insertion (or LJ removal),
        # an existing molecule for a hard-sphere removal
        if choice == "add" or lj:
            center = np.asarray(decomposition.create_cavity(cell_dims, box, True))[:2]
        else:
            center = pos[rng.choice(N)]

        idx_cav = decomposition.neighbor_indices_in_cavity_parallel(
            pos, center, cfg.cavity_radius, cell_dims, box, True)
        n_cav = len(idx_cav)
        if n_cav == 0:
            continue

        if n_cav < n_remove:
            if choice == "add":
                n_add, n_remove = n_cav, n_cav - 1
            else:
                n_add, n_remove = n_cav - 1, n_cav

        if n_remove == 0:
            disk_in = np.empty((0, 2))
            disk_rest = pos
        else:
            chosen_idx = rng.choice(idx_cav, n_remove, replace=False)
            disk_in = pos[chosen_idx]
            keep = np.ones(N, dtype=bool)
            keep[chosen_idx] = False
            disk_rest = pos[keep]

        # trial cloud, wrapped into the cell
        trial_nums = rng.uniform(-cfg.cavity_radius, cfg.cavity_radius, size=(2, n_trials))
        tx = trial_nums[0] + center[0]
        ty = trial_nums[1] + center[1]
        tx, ty = decomposition.apply_pbc_array(tx, ty, cell_dims, box)
        trials = np.column_stack((tx, ty))

        keep_mask = np.ones(n_trials, dtype=bool)
        if n_remove != 0:
            keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False

        system = with_border(disk_rest)
        tree = cKDTree(system, boxsize=box)
        raw = geometry.find_neighbors(tree, trials, diameter, uc, nn, keep_mask=keep_mask)
        if raw is None:
            continue
        trial_set, km = _weighted_trials(raw, tree, cfg, potential, choice, N)

        if n_remove != 0:
            raw_in = geometry.find_neighbors(tree, disk_in, diameter, uc, nn)
            if raw_in is None:
                continue
            mol_in_cav, _ = _weighted_trials(raw_in, tree, cfg, potential, choice, N)
            if len(mol_in_cav) != n_remove:
                continue
        else:
            mol_in_cav = TrialSet(np.empty((0, 2)), np.empty(0))

        if n_remove != 0:
            pool = trial_set.mask(km).merge(mol_in_cav)
            pool = pool[rng.permutation(len(pool))]
        else:
            pool = trial_set.copy()

        if n_add != 0:
            if len(trial_set) < n_add:
                continue
            chosen, sum_new_arr = select_forward(pool, n_add, weigher, rand=rand,
                                                 side=side, r_squared=r_sq, gamma_schedule=gdict)
            if chosen is None:
                continue
            sum_new = np.sum(np.log(sum_new_arr))
            e_u_r_new = np.sum(np.log(chosen.energy_weight))
            chosen_coords = chosen.coords
        else:
            sum_new = 0.0
            e_u_r_new = 0.0
            chosen_coords = np.empty((0, 2))

        if n_remove != 0:
            trimmed = trial_set.mask(km)
            trimmed = trimmed if len(trimmed) != 0 else None
            sum_cur_arr, e_cur_arr = select_reverse(trimmed, mol_in_cav, n_remove, weigher,
                                                    side=side, r_squared=r_sq, gamma_schedule=gdict)
            sum_current = np.sum(np.log(sum_cur_arr))
            e_u_r_current = np.sum(np.log(e_cur_arr))
        else:
            sum_current = 0.0
            e_u_r_current = 0.0

        if choice == "add":
            c["add_attempt"] += 1
            mu_ideal = -np.log(cell_area / (N + 1))
            k_trials = -np.log(n_trials - n_remove)
            mu_excess = -sum_new + sum_current - k_trials
            delta_E = 0.0 if lj else (e_u_r_new - e_u_r_current)
            log_alpha = mu_ideal - beta * mu + mu_excess + delta_E
            log_alpha = -np.log(min(1, np.exp(-log_alpha)))
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                pos = np.append(disk_rest, chosen_coords, axis=0)
                c["add_accept"] += 1
        else:
            mu_ideal = -np.log(N / cell_area)
            k_trials = -np.log(n_trials - n_add)
            mu_excess = -sum_new + sum_current + k_trials
            delta_E = 0.0 if lj else (e_u_r_new - e_u_r_current)
            log_alpha = mu_ideal + beta * mu + mu_excess + delta_E
            log_alpha = -np.log(min(1, np.exp(-log_alpha)))
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                pos = disk_rest if len(chosen_coords) == 0 else np.append(disk_rest, chosen_coords, axis=0)
                c["remove_accept"] += 1

    return pos, c


# --------------------------------------------------------------------------
# pool driver: grid the system, run every cell, recombine with the remainder
# --------------------------------------------------------------------------

def _run_cell(args):
    main, border, n_steps, cell_dims, cfg, potential, seed = args
    rng = np.random.default_rng(seed)
    weigher = make_weigher(cfg, potential)
    new_main, counters = advance_cell(main, border, n_steps, cell_dims,
                                      cfg, potential, weigher, rng)
    return new_main, counters


def advance_parallel(state, cfg, potential, weigher):
    """Advance one window by domain decomposition: grid the box into cells,
    run each cell's MC with its borders frozen, then stitch the live molecules
    back together with the ungridded remainder."""
    if not cfg.two_dim:
        raise NotImplementedError(
            "parallel decomposition is 2D-only, matching the original")
    if cfg.sampler != "rosenbluth":
        raise NotImplementedError(
            f"parallel currently wires the rosenbluth sampler only; "
            f"{cfg.sampler!r} under parallel is not yet implemented")

    rng = state.rng
    decomposition.configure(cfg.cell_length)
    pos = np.asarray(state.positions, float).reshape(-1, 2)

    # grid_up_the_system draws its origin from the legacy global RNG; seed it
    # from the chain RNG so a restart still advances the stream reproducibly.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    pos3 = np.column_stack((pos, np.zeros(len(pos))))
    cells, rest3 = decomposition.grid_up_the_system(
        pos3, side=cfg.box, cell_length=cfg.cell_length)
    rest = rest3[:, :2] if len(rest3) else np.empty((0, 2))

    n_steps = cfg.sampling_window
    seeds = rng.integers(0, 2**31 - 1, size=len(cells))
    tasks = [(cell[0][:, :2], cell[1][:, :2], n_steps, cell[2], cfg, potential, int(s))
             for cell, s in zip(cells, seeds)]

    total = zero_counters()
    new_mains = []
    if cfg.n_workers > 1 and len(tasks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            for new_main, counters in ex.map(_run_cell, tasks):
                new_mains.append(new_main)
                for k in total:
                    total[k] += counters[k]
    else:
        for t in tasks:
            new_main, counters = _run_cell(t)
            new_mains.append(new_main)
            for k in total:
                total[k] += counters[k]

    parts = [m for m in new_mains if len(m)] + ([rest] if len(rest) else [])
    pos = np.vstack(parts) if parts else np.empty((0, 2))

    state.positions = pos
    state.step += cfg.sampling_window * max(1, len(cells))
    packing = (len(pos) * cfg.volume_of_molecule) / cfg.total_volume
    return {"N": len(pos), "packing_fraction": packing, "counters": total,
            "n_cells": len(cells)}
