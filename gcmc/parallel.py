"""Domain-decomposition GCMC, dimension-generic (2D and 3D).

The box is tiled into cubic cells (decomposition.grid_up_nd). Each cell runs
an independent grand-canonical chain over its own molecules while everything
within `buffer` of the cell (the border) is frozen and read-only; molecules in
the gaps between cells do not move during the window. Because energies and
overlaps are evaluated against the true frozen environment and the gap is at
least as wide as the buffer, each cell kernel preserves the global ensemble
restricted to its moves, and cells never see each other's live molecules, so
running them concurrently is exact, not approximate. A new random grid origin
every window restores ergodicity for the gap regions.

All proposal geometry lives in cell-local coordinates with the cell as its own
torus (period = cell_length per axis): insertion centres are uniform over the
cell, trial clouds and removal shadow centres wrap with the cell period, and
translations wrap the same way. That makes the insertion support exactly
cell_length**dim (the volume mu_ideal uses) and makes the removal move the
exact mirror of the insertion, which is what detailed balance requires. The
physics (distances, overlaps, energies) is always evaluated in global
coordinates with the global box PBC.

Supports hard_core and lennard_jones with the rosenbluth and branch samplers
at any validated depth (0-3). For LJ the long-range tail uses
the global molecule count at window start plus the cell's own drift; with
potential option "tail": false the tail is off entirely, which is also the
exact setting for parallel-vs-serial validation.

Requirements checked at setup/grid time: cell_length > 2 * cavity_radius (the
cell torus must hold a cavity box without self-overlap) and buffer >= the
interaction range (diameter for hard core, rc for LJ), so border rings contain
every molecule that can interact with a live one.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
from scipy.spatial import cKDTree

from . import decomposition, geometry
from ._types import BoolArray
from .MC_engine import _lj_energy, _lj_set_energy, _weighted_trials, make_weigher
from .samplers.branch import count_and_sample
from .samplers.rosenbluth import select_forward, select_reverse
from .state import zero_counters
from .trials import TrialSet


def advance_cell(main, border, origin, n_steps, cfg, potential, weigher, rng, n_offset):
    """Advance one cell by n_steps insertion/deletion proposals.

    main    : (N, dim) global coordinates of the cell's live molecules
    border  : (M, dim) frozen global coordinates within `buffer` of the cell
    origin  : (dim,)  global lower corner of the cell cube
    n_offset: global molecule count outside this cell at window start (for the
              LJ tail, which is a system-level quantity)
    Returns (new_main_global, counters).
    """
    dim = 2 if cfg.two_dim else 3
    side = cfg.box
    L_c = cfg.cell_length
    cav = cfg.cavity_radius
    depth = cfg.depth
    lower, uc, nn = cfg.lower_cutoff, cfg.upper_cutoff, cfg.no_neigh
    box = cfg.box_dims
    beta, mu, mf = cfg.beta, cfg.chem_pot, cfg.move_factor
    n_trials = cfg.no_of_trial_molecules
    v_cell = L_c**dim
    ljaux = cfg.potential_type == "ljaux"
    lj = cfg.potential_type == "lj" or ljaux
    branch = cfg.sampler == "branch"
    if cfg.sampler == "metropolis":
        n_trials = 1
    tail = cfg.lj_tail_pref if (lj and cfg.lj_tail_pref) else 0.0
    c = zero_counters()

    u_main = (np.asarray(main, float).reshape(-1, dim) - origin) % side
    border = np.asarray(border, float).reshape(-1, dim)

    def to_global(u):
        return (np.atleast_2d(u) + origin) % side

    def system_for(rest_u):
        glob = to_global(rest_u) if len(rest_u) else np.empty((0, dim))
        if len(border):
            return np.vstack([glob, border]) if len(glob) else border
        return glob

    step = 0
    while step < n_steps:
        N = len(u_main)
        choice = (
            "add"
            if N == 0
            else rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])
        )

        # ----- translate (cell-torus wrap keeps the molecule owned) -----
        if choice == "move":
            c["move_step"] += 1
            j = int(rng.choice(N))
            u_new = (u_main[j] + mf * (rng.uniform(0, 1, dim) - 0.5)) % L_c
            rest_u = np.delete(u_main, j, axis=0)
            system = system_for(rest_u)
            moved = to_global(u_new)[0]
            if len(system) == 0:
                u_main[j] = u_new
                c["move_accept"] += 1
                continue
            tree = cKDTree(system, boxsize=box)
            if not lj:
                if len(tree.query_ball_point(moved, lower)) == 0:
                    u_main[j] = u_new
                    c["move_accept"] += 1
            else:
                old = to_global(u_main[j])[0]
                d_old, _ = tree.query(
                    old, k=min(nn, len(system)), distance_upper_bound=uc
                )
                d_new, _ = tree.query(
                    moved, k=min(nn, len(system)), distance_upper_bound=uc
                )
                d_old = np.atleast_1d(d_old)
                d_new = np.atleast_1d(d_new)
                d_old = d_old[np.isfinite(d_old)]
                d_new = d_new[np.isfinite(d_new)]
                if d_new.size and np.any(d_new < lower):
                    continue
                d_e = _lj_energy(d_new, uc) - _lj_energy(d_old, uc)
                if rng.uniform(0, 1) < np.exp(-beta * d_e):
                    u_main[j] = u_new
                    c["move_accept"] += 1
            continue

        # ----- insertion / deletion -----
        if choice == "add":
            c["add_step"] += 1
            n_add, n_remove = depth + 1, depth
        else:
            c["remove_step"] += 1
            n_add, n_remove = depth, depth + 1
        step += 1

        if depth == 0:
            # mirror scheme, as in the serial engine
            if choice == "add":
                u_center = rng.uniform(0.0, L_c, size=dim)
                anchor = -1
            else:
                anchor = int(rng.choice(N))
                u_center = (u_main[anchor] + rng.uniform(-cav, cav, size=dim)) % L_c
            if n_remove == 0:
                disk_in_u = np.empty((0, dim))
                rest_u = u_main
            else:
                disk_in_u = u_main[anchor][None, :]
                rest_u = np.delete(u_main, anchor, axis=0)
        else:
            # depth >= 1: the original BRANCH-MC proposal on the cell torus,
            # as the legacy parallelization ran it. This mode is APPROXIMATE:
            # the depth >= 1 scheme satisfies global balance but not detailed
            # balance (proof: masking a reversible chain with the
            # reversal-symmetric touched-set indicator provably preserves its
            # stationary distribution, yet measures +2.6 +/- 0.6 % on the
            # hard-sphere EOS -- so the chain is not reversible), and global
            # balance relies on full translation invariance, which a cell
            # cannot provide. The residual bias is controlled by the
            # cavity/cell ratio: +19% at cell_length = 2.7 cavity widths,
            # +1.3 +/- 0.8 % at 6.7 widths, +0.9 +/- 0.6 % at 10 widths.
            # sim_setup warns accordingly; serial runs are the exact
            # reference at depth >= 1.
            if choice == "add" or lj:
                u_center = rng.uniform(0.0, L_c, size=dim)
            else:
                u_center = u_main[int(rng.choice(N))].copy()
            d = (u_main - u_center + L_c / 2.0) % L_c - L_c / 2.0
            idx_cav = np.flatnonzero(np.all(np.abs(d) <= cav, axis=1))
            n_cav = len(idx_cav)
            if n_cav == 0:
                continue
            if n_cav < n_remove:
                if choice == "add":
                    n_add, n_remove = n_cav, n_cav - 1
                else:
                    n_add, n_remove = n_cav - 1, n_cav
            if n_remove == 0:
                disk_in_u = np.empty((0, dim))
                rest_u = u_main
            else:
                chosen_idx = rng.choice(idx_cav, n_remove, replace=False)
                disk_in_u = u_main[chosen_idx]
                keep = np.ones(N, dtype=bool)
                keep[chosen_idx] = False
                rest_u = u_main[keep]
        disk_in = to_global(disk_in_u) if len(disk_in_u) else np.empty((0, dim))
        system = system_for(rest_u)

        # trial cloud on the cell torus around the centre, mapped to global
        trials_u = (rng.uniform(-cav, cav, size=(n_trials, dim)) + u_center) % L_c
        trials = to_global(trials_u)

        keep_mask = np.ones(n_trials, dtype=bool)
        if n_remove > 0:
            keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False

        if len(system) == 0:
            # empty environment: every trial survives with unit pair weight
            trial_set = TrialSet(trials, np.ones(n_trials))
            km: BoolArray | None = keep_mask
        else:
            tree = cKDTree(system, boxsize=box)
            raw = geometry.find_neighbors(
                tree, trials, lower, uc, nn, keep_mask=keep_mask
            )
            if raw is None:
                if n_add != 0:
                    continue
                trial_set = TrialSet(
                    np.empty((0, dim)),
                    np.empty(0),
                    ref_neighbor_coord=np.empty((0, dim)),
                )
                km = np.empty(0, dtype=bool)
            else:
                trial_set, km = _weighted_trials(
                    raw, tree, cfg, potential, choice, N + n_offset
                )

        # removed molecule's own weight
        if n_remove > 0:
            if len(system) == 0:
                mol_in_cav = TrialSet(disk_in, np.ones(1))
            else:
                raw_in = geometry.find_neighbors(tree, disk_in, lower, uc, nn)
                if raw_in is None:
                    continue
                mol_in_cav, _ = _weighted_trials(
                    raw_in, tree, cfg, potential, choice, N + n_offset
                )
                if len(mol_in_cav) != n_remove:
                    continue
        else:
            mol_in_cav = None

        assert km is not None  # raw was handled above; km rides with it
        pool = trial_set.mask(km)
        if n_remove > 0:
            assert mol_in_cav is not None  # built in the n_remove branch above
            pool = pool.merge(mol_in_cav)
            if len(pool) > 1:
                pool = pool[rng.permutation(len(pool))]

        # forward (insertion) estimate
        if n_add > 0:
            if len(pool) < n_add:
                continue
            e_u_r_new = 0.0
            if branch:
                num_new, idx = count_and_sample(
                    pool.coords, n_add, lower, box_lengths=box
                )
                if num_new <= 0 or idx is None:
                    continue
                chosen_coords = pool.coords[list(idx)]
                if lj:
                    u_ins = _lj_set_energy(chosen_coords, system, box, uc)
                    sum_new = np.log(num_new) - beta * u_ins
                else:
                    sum_new = np.log(num_new)
            else:
                chosen, sum_new_arr = select_forward(
                    pool,
                    n_add,
                    weigher,
                    rand=lambda: rng.uniform(0, 1),
                    side=side,
                    r_squared=cfg.r_squared,
                    gamma_schedule=cfg.gamma_dict,
                )
                if chosen is None or sum_new_arr is None:
                    continue
                chosen_coords = chosen.coords
                sum_new = np.sum(np.log(sum_new_arr))
                e_u_r_new = np.sum(np.log(chosen.energy_weight))
        else:
            chosen_coords = np.empty((0, dim))
            sum_new = 0.0
            e_u_r_new = 0.0

        # reverse (deletion) estimate
        if n_remove > 0:
            e_u_r_current = 0.0
            if branch:
                num_cur, _ = count_and_sample(
                    pool.coords, n_remove, lower, box_lengths=box
                )
                if num_cur <= 0:
                    continue
                if lj:
                    u_rem = _lj_set_energy(disk_in, system, box, uc)
                    sum_current = np.log(num_cur) - beta * u_rem
                else:
                    sum_current = np.log(num_cur)
            else:
                _trimmed = trial_set.mask(km)
                trimmed = _trimmed if len(_trimmed) else None
                assert mol_in_cav is not None  # n_remove > 0 built it above
                s_arr, e_arr = select_reverse(
                    trimmed,
                    mol_in_cav,
                    n_remove,
                    weigher,
                    side=side,
                    r_squared=cfg.r_squared,
                    gamma_schedule=cfg.gamma_dict,
                )
                sum_current = np.sum(np.log(s_arr))
                e_u_r_current = np.sum(np.log(e_arr))
        else:
            sum_current = 0.0
            e_u_r_current = 0.0

        # LJ tail: system-level, applied once per move with the global count
        if tail:
            n_glob_old = N + n_offset
            n_glob_new = n_glob_old + n_add - n_remove
            delta_E = beta * tail * (n_glob_new**2 - n_glob_old**2)
        else:
            delta_E = 0.0
        if ljaux and not branch:
            # divide out the chosen/removed bias: e_u_r on combined weights
            # plus the explicit pair-energy term cancels the Boltzmann parts
            u_ins = (
                _lj_set_energy(np.atleast_2d(chosen_coords), system, box, uc)
                if len(chosen_coords)
                else 0.0
            )
            u_rem = _lj_set_energy(disk_in, system, box, uc) if len(disk_in) else 0.0
            delta_E += (e_u_r_new - e_u_r_current) + beta * (u_ins - u_rem)
        elif not (lj or branch):
            # pmf aux bias divides back out; plain hard core: both terms 0
            delta_E += e_u_r_new - e_u_r_current

        if choice == "add":
            c["add_attempt"] += 1
            mu_ideal = -np.log(v_cell / (N + 1))
            if branch:
                k_trials = -np.log((n_trials - n_remove) / n_add)
            else:
                k_trials = -np.log(n_trials - n_remove)
            mu_excess = -sum_new + sum_current - k_trials
            log_alpha = max(0.0, mu_ideal - beta * mu + mu_excess + delta_E)
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                new_u = (np.atleast_2d(chosen_coords) - origin) % side
                # rest_u, not u_main: at depth >= 1 the move also removed
                # n_remove molecules, which must not be resurrected
                u_main = np.vstack([rest_u, new_u]) if len(rest_u) else new_u
                c["add_accept"] += 1
        else:
            mu_ideal = -np.log(N / v_cell)
            if branch:
                k_trials = -np.log((n_trials - n_add) / n_remove)
            else:
                k_trials = -np.log(n_trials - n_add)
            mu_excess = -sum_new + sum_current + k_trials
            log_alpha = max(0.0, mu_ideal + beta * mu + mu_excess + delta_E)
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                if n_add > 0:
                    # at depth >= 1 a deletion re-inserts n_add molecules
                    re_u = (np.atleast_2d(chosen_coords) - origin) % side
                    u_main = np.vstack([rest_u, re_u]) if len(rest_u) else re_u
                else:
                    u_main = rest_u
                c["remove_accept"] += 1

    return to_global(u_main) if len(u_main) else np.empty((0, dim)), c


def _run_cell(args):
    """Worker entry point (one argument tuple, multiprocessing-friendly):
    build the per-cell rng and weigher, run `advance_cell` on this cell's
    interior particles with its frozen border, and return the updated
    interior plus counters."""
    main, border, origin, n_steps, cfg, potential, n_offset, seed = args
    rng = np.random.default_rng(seed)
    weigher = make_weigher(cfg, potential)
    return advance_cell(
        main, border, origin, n_steps, cfg, potential, weigher, rng, n_offset
    )


def advance_parallel(state, cfg, potential, weigher):
    """Advance one window by domain decomposition (2D or 3D)."""
    if cfg.sampler not in ("rosenbluth", "branch", "metropolis"):
        raise NotImplementedError(
            f"parallel supports metropolis, rosenbluth, and branch; got {cfg.sampler!r}"
        )
    if cfg.cell_length <= 2.0 * cfg.cavity_radius:
        raise ValueError(
            f"cell_length ({cfg.cell_length}) must exceed twice the cavity "
            f"radius ({2 * cfg.cavity_radius}) for the cell-torus geometry"
        )

    dim = 2 if cfg.two_dim else 3
    rng = state.rng
    pos = np.asarray(state.positions, float).reshape(-1, dim)
    n_global = len(pos)

    cells, rest = decomposition.grid_up_nd(
        pos, cfg.box, cfg.cell_length, cfg.buffer, rng, dim
    )
    n_steps = cfg.sampling_window
    seeds = rng.integers(0, 2**31 - 1, size=len(cells))
    tasks = [
        (
            cell[0],
            cell[1],
            cell[2],
            n_steps,
            cfg,
            potential,
            n_global - len(cell[0]),
            int(s),
        )
        for cell, s in zip(cells, seeds)
    ]

    total = zero_counters()
    new_mains = []
    if cfg.n_workers > 1 and len(tasks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            results = list(ex.map(_run_cell, tasks))
    else:
        results = [_run_cell(t) for t in tasks]
    for new_main, counters in results:
        new_mains.append(new_main)
        for k in total:
            total[k] += counters[k]

    parts = [m for m in new_mains if len(m)] + ([rest] if len(rest) else [])
    pos = np.vstack(parts) if parts else np.empty((0, dim))

    state.positions = pos
    state.step += cfg.sampling_window * max(1, len(cells))
    packing = (len(pos) * cfg.volume_of_molecule) / cfg.total_volume
    return {
        "N": len(pos),
        "packing_fraction": packing,
        "counters": total,
        "n_cells": len(cells),
    }
