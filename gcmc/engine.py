"""Serial Monte Carlo engine: advance the chain by one window.

A window is `sampling_window` insertion/deletion steps; translation moves are
interspersed (chosen half the time) but do not count toward the window, exactly
as in the original. This reproduces the original serial run_mc_in_a_cell:
translation with an LJ Metropolis test or a hard-core overlap test, and
add/remove via the nested-cavity Rosenbluth weights with the same acceptance
criterion (ideal term, trial-count term, Rosenbluth excess, and the energy
difference that is zero for hard cores and folded into the LJ tail on the bulk).

All randomness goes through state.rng, so the chain is reproducible and
checkpointable. BRANCH and reweighting are not wired here yet and raise.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

from .trials import TrialSet
from . import geometry
from .samplers.base import EnergyWeigher, UniformWeigher, ReweightWeigher
from . import reweighting
from .samplers.rosenbluth import select_forward, select_reverse
from .samplers.branch import count_and_sample
from .state import zero_counters


def make_weigher(cfg, potential):
    if cfg.potential_type == "hard_sphere":
        return UniformWeigher()
    if cfg.potential_type in ("lj", "aux"):
        return EnergyWeigher(potential)
    if cfg.potential_type == "reweight":
        if not cfg.two_dim:
            raise NotImplementedError("reweighting is the 2D hexatic (psi6) method")
        return ReweightWeigher()
    raise NotImplementedError(f"engine does not yet wire potential {cfg.potential_type!r}")


def _weighted_trials(raw, kdtree, cfg, potential, choice, n_full):
    """Turn the raw neighbour result into a weighted TrialSet, matching the
    original find_neighbors_fast weighting for the active mode."""
    coords, dists, idxs, km = raw
    g1 = cfg.gamma_dict[1]
    t = cfg.potential_type
    if t == "hard_sphere":
        return TrialSet(coords, np.ones(len(coords))), km
    if t == "lj":
        w = potential.weight_bulk(dists, n_particles=n_full, choice=choice)
        return TrialSet(coords, w), km
    if t == "aux":
        if cfg.two_dim:
            return TrialSet(coords, potential.weight_bulk(dists, gamma=g1)), km
        w, ref = potential.weight_bulk(dists, coords=coords, idxs=idxs,
                                       kdtree=kdtree, gamma=g1)
        return TrialSet(coords, w, ref_neighbor_coord=ref), km
    if t == "reweight":
        # 2D hexatic: unit bulk weight, carry the psi6 phasor accumulators
        cos_sum, sin_sum, counts = reweighting.calculate_order_parameter(
            coords, dists, idxs, kdtree, cfg.box, g1)
        return TrialSet(coords, np.ones(len(coords)),
                        cossum=cos_sum, sinsum=sin_sum, counts=counts), km
    raise NotImplementedError(t)


def _advance_metropolis(state, cfg, potential):
    """Standard single-particle grand-canonical Metropolis (textbook GCMC): one
    trial insertion at a uniform-random box point or one deletion of a random
    particle, accepted by the GCMC criterion. No cavity, no multiple trials, no
    depth. Supports hard_core (2D/3D) and Lennard-Jones (3D), using the same
    pair energy and tail as the Rosenbluth path so the two agree at equilibrium.
    """
    if cfg.potential_type == "aux":
        raise NotImplementedError(
            "metropolis with a pmf auxiliary is not yet implemented; "
            "use the rosenbluth sampler for pmf runs")

    rng = state.rng
    pos = np.asarray(state.positions, float)
    c = zero_counters()
    side = cfg.box
    box = cfg.box_dims
    dim = cfg.dimension
    diameter = cfg.diameter
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    beta = cfg.beta
    mu = cfg.chem_pot
    V = cfg.total_volume
    hs = cfg.potential_type == "hard_sphere"

    def neighbour_dists(point, tree, k):
        d, _ = tree.query([point], k=min(k, max(1, k)), distance_upper_bound=uc)
        d = d[0]
        return d[np.isfinite(d)]

    step = 0
    while step <= cfg.sampling_window:
        N = len(pos)
        choice = "add" if N == 0 else rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ----- translate -----
        if choice == "move" and N > 0:
            c["move_step"] += 1
            j = rng.choice(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            disp = mf * (rng.uniform(0, 1, dim) - 0.5)
            moved = (old + disp) % side
            if len(without) == 0:
                pos = moved.reshape(1, -1)
                c["move_accept"] += 1
                continue
            tree = cKDTree(without, boxsize=box)
            d_new = neighbour_dists(moved, tree, nn)
            if hs:
                if not (d_new.size and np.any(d_new < diameter)):
                    pos = np.append(without, [moved], axis=0)
                    c["move_accept"] += 1
            else:
                d_old = neighbour_dists(old, tree, nn)
                u_new = _lj_energy(d_new, uc)
                u_old = _lj_energy(d_old, uc)
                if np.log(rng.uniform(0, 1)) <= -beta * (u_new - u_old):
                    pos = np.append(without, [moved], axis=0)
                    c["move_accept"] += 1
            continue

        if choice == "remove" and N == 0:
            continue

        # ----- insertion -----
        if choice == "add":
            c["add_step"] += 1
            c["add_attempt"] += 1
            step += 1
            trial = rng.uniform(0, side, dim)
            if N > 0:
                tree = cKDTree(pos, boxsize=box)
                d = neighbour_dists(trial, tree, nn)
            else:
                d = np.empty(0)
            if hs:
                if d.size and np.any(d < diameter):
                    continue
                log_bf = 0.0
            else:
                u = _lj_energy(d, uc) + cfg.lj_tail_pref * (2.0 * N + 1.0)
                log_bf = -beta * u
            log_acc = math.log(V) - math.log(N + 1) + beta * mu + log_bf
            if math.log(rng.uniform(0, 1)) <= log_acc:
                pos = np.append(pos, [trial], axis=0) if N > 0 else trial.reshape(1, -1)
                c["add_accept"] += 1

        # ----- deletion -----
        else:
            c["remove_step"] += 1
            step += 1
            j = rng.choice(N)
            chosen = pos[j]
            without = np.delete(pos, j, axis=0)
            if hs:
                log_bf = 0.0
            else:
                tree = cKDTree(without, boxsize=box) if len(without) else None
                d = neighbour_dists(chosen, tree, nn) if tree is not None else np.empty(0)
                u = _lj_energy(d, uc) + cfg.lj_tail_pref * (2.0 * N - 1.0)
                log_bf = -beta * u  # removing reverses the sign below
            log_acc = math.log(N) - math.log(V) - beta * mu - log_bf
            if math.log(rng.uniform(0, 1)) <= log_acc:
                pos = without
                c["remove_accept"] += 1

    state.positions = pos
    state.step += cfg.sampling_window
    packing = (len(pos) * cfg.volume_of_molecule) / cfg.total_volume
    return {"N": len(pos), "packing_fraction": packing, "counters": c}


def _lj_energy(dists, rc):
    d = np.asarray(dists, float)
    d = d[(d < rc) & (d > 0)]
    if d.size == 0:
        return 0.0
    ir6 = (1.0 / d) ** 6
    return float(np.sum(4.0 * ir6 * (ir6 - 1.0)))


def advance(state, cfg, potential, weigher):
    if cfg.metropolis:
        return _advance_metropolis(state, cfg, potential)
    if cfg.potential_type == "reweight" and not cfg.two_dim:
        raise NotImplementedError(
            "3D reweighting (Q6) is not a coherent estimator in the original "
            "and is not wired; reweighting is the 2D hexatic (psi6) method")

    rng = state.rng
    pos = np.asarray(state.positions, float)
    c = zero_counters()

    side = cfg.box
    box = cfg.box_dims
    two_d = cfg.two_dim
    dim = cfg.dimension
    diameter = cfg.diameter
    r_sq = cfg.r_squared
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    n_trials = cfg.no_of_trial_molecules
    depth = cfg.depth
    gdict = cfg.gamma_dict
    beta = cfg.beta
    temp = cfg.temp
    mu = cfg.chem_pot
    total_vol = cfg.total_volume
    lj = cfg.potential_type == "lj"
    aux = cfg.potential_type == "aux"
    branch = cfg.branch
    branch_radius = cfg.diameter / 2.0
    reweight = cfg.potential_type == "reweight"
    if branch and reweight:
        raise NotImplementedError("branch and reweight are alternative biasing "
                                  "schemes; combining them is undefined")

    def rand():
        return rng.random()

    step = 0
    while step <= cfg.sampling_window:
        N = len(pos)
        choice = "add" if N == 0 else rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ---------------------------------------------------------- translate
        if choice == "move" and N > 0:
            c["move_step"] += 1
            j = rng.choice(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            ranf1, ranf2 = rng.uniform(0, 1, 2)
            mx = (old[0] + mf * (ranf1 - 0.5)) % side
            my = (old[1] + mf * (ranf2 - 0.5)) % side
            if two_d:
                moved = np.array([mx, my])
            else:
                ranf3 = (ranf1 + ranf2) % 1
                mz = (old[2] + mf * (ranf3 - 0.5)) % side
                moved = np.array([mx, my, mz])
            tree = cKDTree(without, boxsize=box)
            if lj:
                d_new, _ = tree.query([moved], k=nn, distance_upper_bound=uc)
                d_new = d_new[0][np.isfinite(d_new[0])]
                if d_new.size and np.any(d_new < diameter):
                    pass
                else:
                    d_old, _ = tree.query([old], k=nn, distance_upper_bound=uc)
                    d_old = d_old[0][np.isfinite(d_old[0])]

                    def _u(d):
                        if d.size == 0:
                            return 0.0
                        ir6 = (1.0 / d) ** 6
                        return np.sum(4.0 * ir6 * (ir6 - 1.0))
                    dU = _u(d_new) - _u(d_old)
                    if np.log(rng.uniform(0, 1)) <= -dU / temp:
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
        if step >= cfg.sampling_window:
            break

        # cavity centre
        if choice == "add" or lj:
            center = rng.uniform(0, side, size=3)[:dim]
        else:
            center = pos[rng.choice(N)]

        idx_cav = geometry.indices_in_cavity(pos, center, cfg.cavity_radius, side, two_d)
        n_cav = len(idx_cav)
        if n_cav == 0:
            continue

        if n_cav < n_remove:
            if choice == "add":
                n_add, n_remove = n_cav, n_cav - 1
            else:
                n_add, n_remove = n_cav - 1, n_cav

        # split system into in-cavity (to remove) and the rest
        if n_remove == 0:
            disk_in = np.empty((0, dim))
            disk_rest = pos
        else:
            chosen_idx = rng.choice(idx_cav, n_remove, replace=False)
            disk_in = pos[chosen_idx]
            keep = np.ones(N, dtype=bool)
            keep[chosen_idx] = False
            disk_rest = pos[keep]

        if len(disk_rest) == 0:
            # nothing left to weigh trials against; degenerate near N=1, never
            # reached at production density. skip rather than crash.
            continue
        # trial molecules in the cavity
        trial_nums = rng.uniform(-cfg.cavity_radius, cfg.cavity_radius, size=(dim, n_trials))
        tx = (trial_nums[0] + center[0]) % side
        ty = (trial_nums[1] + center[1]) % side
        if two_d:
            trials = np.column_stack((tx, ty))
        else:
            tz = (trial_nums[2] + center[2]) % side
            trials = np.column_stack((tx, ty, tz))

        keep_mask = np.ones(n_trials, dtype=bool)
        if n_remove != 0:
            keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False

        tree = cKDTree(disk_rest, boxsize=box)
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
            mol_in_cav = TrialSet(np.empty((0, dim)), np.empty(0))

        # assemble the pool the forward selection draws from
        if n_remove != 0:
            pool = trial_set.mask(km).merge(mol_in_cav)
            pool = pool[rng.permutation(len(pool))]
        else:
            pool = trial_set.copy()

        # forward (new configuration)
        if n_add != 0:
            if len(trial_set) < n_add:
                continue
            if branch:
                num_new, idx = count_and_sample(pool.coords, n_add,
                                                radius=branch_radius, box_lengths=box)
                if num_new == 0:
                    continue
                sum_new = np.log(num_new)
                e_u_r_new = 0.0
                chosen_coords = pool.coords[list(idx)]
            else:
                chosen, sum_new_arr = select_forward(pool, n_add, weigher, rand=rand,
                                                     side=side, r_squared=r_sq, gamma_schedule=gdict)
                if chosen is None:
                    continue
                sum_new = np.sum(np.log(sum_new_arr))
                if reweight:
                    e_u_r_new = np.sum(np.log(reweighting.psi6_magnitude(
                        chosen.cossum, chosen.sinsum, chosen.counts)))
                else:
                    e_u_r_new = np.sum(np.log(chosen.energy_weight))
                chosen_coords = chosen.coords
        else:
            sum_new = 0.0
            e_u_r_new = 0.0
            chosen_coords = np.empty((0, dim))

        # reverse (current configuration)
        if n_remove != 0:
            if branch:
                num_cur, _ = count_and_sample(pool.coords, n_remove,
                                              radius=branch_radius, box_lengths=box)
                sum_current = np.log(num_cur) if num_cur > 0 else -np.inf
                e_u_r_current = 0.0
            else:
                trimmed = trial_set.mask(km)
                trimmed = trimmed if len(trimmed) != 0 else None
                sum_cur_arr, e_cur_arr = select_reverse(trimmed, mol_in_cav, n_remove, weigher,
                                                        side=side, r_squared=r_sq, gamma_schedule=gdict)
                sum_current = np.sum(np.log(sum_cur_arr))
                e_u_r_current = np.sum(np.log(e_cur_arr))
        else:
            sum_current = 0.0
            e_u_r_current = 0.0

        if branch and not np.isfinite(sum_current):
            continue

        # acceptance
        if choice == "add":
            c["add_attempt"] += 1
            mu_ideal = -np.log(total_vol / (N + 1))
            if branch:
                k_trials = -np.log((n_trials - n_remove) / n_add)
            else:
                k_trials = -np.log(n_trials - n_remove)
            mu_excess = -sum_new + sum_current - k_trials
            delta_E = 0.0 if lj else (e_u_r_new - e_u_r_current)
            log_alpha = mu_ideal - beta * mu + mu_excess + delta_E
            log_alpha = -np.log(min(1, np.exp(-log_alpha)))
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                pos = np.append(disk_rest, chosen_coords, axis=0)
                c["add_accept"] += 1
        else:
            mu_ideal = -np.log(N / total_vol)
            if branch:
                k_trials = -np.log((n_trials - n_add) / n_remove)
            else:
                k_trials = -np.log(n_trials - n_add)
            mu_excess = -sum_new + sum_current + k_trials
            delta_E = 0.0 if lj else (e_u_r_new - e_u_r_current)
            log_alpha = mu_ideal + beta * mu + mu_excess + delta_E
            log_alpha = -np.log(min(1, np.exp(-log_alpha)))
            if log_alpha <= -np.log(rng.uniform(0, 1)):
                if len(chosen_coords) == 0:
                    pos = disk_rest
                else:
                    pos = np.append(disk_rest, chosen_coords, axis=0)
                c["remove_accept"] += 1

    state.positions = pos
    state.step += cfg.sampling_window
    packing = (len(pos) * cfg.volume_of_molecule) / total_vol
    return {"N": len(pos), "packing_fraction": packing, "counters": c}
