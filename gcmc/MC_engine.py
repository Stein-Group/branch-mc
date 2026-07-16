"""Serial Monte Carlo engine: advance the chain by one window.

A window is `sampling_window` insertion/deletion steps; translation moves are
interspersed (chosen half the time) but do not count toward the window, exactly
as in the original. This reproduces the original serial run_mc_in_a_cell:
translation with an LJ Metropolis test or a hard-core overlap test, and
add/remove via the nested-cavity Rosenbluth weights with the same acceptance
criterion (ideal term, trial-count term, Rosenbluth excess, and the energy
difference that is zero for hard cores and folded into the LJ tail on the bulk).

All randomness goes through state.rng, so the chain is reproducible and
checkpointable.

Detailed balance at depth 0 is validated against the hard-sphere equation of
state (Carnahan-Starling) and by metropolis/rosenbluth/branch cross-agreement
for hard spheres and Lennard-Jones; equilibrium density is invariant under the
trial count and the trial box side, as it must be. The deletion move mirrors
the insertion exactly: a uniformly chosen particle is removed and the reverse
trial cloud is generated around a centre drawn uniformly in the cavity box
around it. depth > 0 is gated in sim_setup pending a re-derivation of the
multi-molecule acceptance.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

from .trials import TrialSet
from . import geometry
from . import neighbors
from .samplers.base import EnergyWeigher, UniformWeigher, ReweightWeigher
from .samplers import reweighting
from .samplers.rosenbluth import select_forward, select_reverse
from .samplers.branch import count_and_sample
from .samplers.channel import (ChannelSweep, ChannelSweepLJ, local_from_box,
                               cuboid_members, random_rotation_matrix,
                               ChannelSweepFull, ChannelSweepLJFull,
                               ray_box_chord, axial_coord_full, rod_members_full,
                               cuboid_members_cl, rod_members_full_cl)
from .samplers import branch as branch_mod
from .state import zero_counters
from .potentials.lennard_jones import lj_pair_energy



def _pore_bounds(cfg):
    """Centre-accessible bounds for a hard pore inside the cubic cell of side
    cfg.box. Returns (kind, params). The slab/cylinder is centred in the cell.
    A sphere of radius cfg.radius must keep its centre off the wall by that
    radius, so the clearance is cfg.radius, not a hardcoded 0.5."""
    L = cfg.box
    cl = cfg.radius
    if cfg.pore_geometry == "slit":
        z0 = (L - cfg.pore_H) / 2.0
        return ("slit", z0 + cl, z0 + cfg.pore_H - cl)           # zmin, zmax for centres
    return ("cylinder", L / 2.0, cfg.pore_R - cl)                 # axis centre, rmax for centres


def _inside_pore(coords, cfg):
    """Boolean mask: which centres satisfy the hard wall. coords is (M, dim) or
    a single (dim,) point. A sphere of radius cfg.radius must keep its centre off
    the wall by that radius (the configurable clearance, not a hardcoded 0.5)."""
    if cfg.pore_geometry is None:
        return None
    a = np.atleast_2d(np.asarray(coords, float))
    kind = cfg.pore_geometry
    cl = cfg.radius
    if kind == "slit":
        L = cfg.box; z0 = (L - cfg.pore_H) / 2.0
        z = a[:, 2]
        ok = (z >= z0 + cl) & (z <= z0 + cfg.pore_H - cl)
    else:
        L = cfg.box; cx = cy = L / 2.0
        r = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
        ok = r <= (cfg.pore_R - cl)
    return ok if np.ndim(coords) > 1 else bool(ok[0])


def make_weigher(cfg, potential):
    if cfg.potential_type == "hard_sphere":
        return UniformWeigher()
    if cfg.potential_type in ("lj", "aux", "ljaux", "ljbranch"):
        return EnergyWeigher(potential)
    if cfg.potential_type == "reweight":
        if not cfg.two_dim:
            raise NotImplementedError("reweighting is the 2D hexatic (psi6) method")
        return ReweightWeigher()
    raise NotImplementedError(f"engine does not yet wire potential {cfg.potential_type!r}")


def _is_matrix_neighbor(container, idxs, n_fluid):
    """Which returned neighbours are frozen obstacles.

    A kd-tree over `[fluid | matrix]` answers this by index: anything at or past
    `n_fluid` is an obstacle (the pad index lands there too, and carries an
    infinite distance). A CellList hands back slot ids, which say nothing about
    ordering, so it answers from its own `is_matrix` flags. Same question, two
    containers."""
    if container is not None and hasattr(container, "matrix_at"):
        return container.matrix_at(idxs)
    return np.asarray(idxs) >= n_fluid


def _lj_singleparticle_weight(dists, idxs, potential, n_fluid=None,
                              container=None):
    """Per-trial Boltzmann factor exp(-beta * sum_j u(r_ij)) over neighbours
    within rc. With `n_fluid` given (QA), frozen-matrix neighbours are dropped
    from the LJ sum (they interact by overlap exclusion only); without it every
    neighbour counts. For a bulk run this equals potential.weight_bulk(dists)."""
    d = np.asarray(dists, float)
    if n_fluid is not None:
        d = np.where(_is_matrix_neighbor(container, idxs, n_fluid), np.inf, d)
    return potential.weight_pair(d)


def _weighted_trials(raw, kdtree, cfg, potential, choice, n_full, n_fluid=None):
    """Turn the raw neighbour result into a weighted TrialSet, matching the
    original find_neighbors_fast weighting for the active mode."""
    coords, dists, idxs, km = raw
    g1 = cfg.gamma_dict[1]
    t = cfg.potential_type
    if t == "hard_sphere":
        return TrialSet(coords, np.ones(len(coords))), km
    if t == "ljbranch":
        # Reuse: each trial's single-particle LJ weight against the fluid rest
        # (any frozen matrix excludes by overlap only) is the member-vs-system
        # factor every clique needs downstream. Computed once here from the
        # neighbour distances already in hand, never recomputed per clique.
        w = _lj_singleparticle_weight(dists, idxs, potential, n_fluid,
                                      container=kdtree)
        return TrialSet(coords, w), km
    if t == "ljaux":
        w, ref = potential.weight_bulk(dists, coords=coords, idxs=idxs,
                                       kdtree=kdtree, gamma=g1)
        return TrialSet(coords, w, ref_neighbor_coord=ref), km
    if t == "lj":
        # pair Boltzmann factors only. The long-range tail is a move-level
        # quantity applied once in the acceptance (delta_E_lj); folding it into
        # the per-trial weights put it inside W_old on deletions, where the
        # denominator position inverted its sign and over-accepted deletions
        # by exp(2*beta*|dU_tail|).
        w = potential.weight_bulk(dists)
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
    if cfg.potential_type == "ljaux":
        # with a single trial the selection bias cancels identically, so
        # metropolis + pmf IS metropolis on the plain LJ potential
        potential = potential.lj

    rng = state.rng
    pos = np.asarray(state.positions, float)
    c = zero_counters()
    side = cfg.box
    box = cfg.box_dims
    dim = cfg.dimension
    lower = cfg.lower_cutoff      # exclusion floor (= diameter for hard cores)
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    beta = cfg.beta
    mu = state.mu if getattr(state, "mu", None) is not None else cfg.chem_pot
    V = cfg.total_volume
    hs = cfg.potential_type == "hard_sphere"

    def neighbour_dists(point, tree, k):
        # saturating query: double k until the neighbour list is not truncated,
        # so an LJ energy summed over it never silently drops an in-range pair.
        d, _ = geometry.query_saturating(tree, [point], k, uc)
        d = d[0]
        return d[np.isfinite(d)]

    step = 0
    while step < cfg.sampling_window:
        N = len(pos)
        # One fixed move-type distribution regardless of N. The proposal ratio
        # q(0->1)/q(1->0) is then 1 and never has to enter the acceptance; forcing
        # "add" at N=0 (the old code) left that ratio out and biased P(N=0) by a
        # factor of 4. A move or remove proposed at N=0 is simply a null / rejected
        # attempt, counted so the window advances as it would from any state.
        choice = rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ----- translate -----
        if choice == "move":
            c["move_step"] += 1
            if N == 0:
                continue                      # null proposal: nothing to move
            j = rng.choice(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            disp = mf * (rng.uniform(0, 1, dim) - 0.5)
            moved = (old + disp) % side
            if cfg.pore_geometry is not None and not _inside_pore(moved, cfg):
                continue                      # move into the wall: rejected
            if len(without) == 0:
                pos = moved.reshape(1, -1)
                c["move_accept"] += 1
                continue
            tree = cKDTree(without, boxsize=box)
            d_new = neighbour_dists(moved, tree, nn)
            if hs:
                if not (d_new.size and geometry.overlaps(d_new, lower).any()):
                    pos = np.append(without, [moved], axis=0)
                    c["move_accept"] += 1
            else:
                # LJ hard core at the floor: a move onto an overlap is rejected
                # outright (the same floor the biased samplers apply), so every
                # sampler shares one "LJ plus hard core at lower_cutoff" model.
                if d_new.size and geometry.overlaps(d_new, lower).any():
                    continue
                d_old = neighbour_dists(old, tree, nn)
                u_new = _lj_energy(d_new, uc)
                u_old = _lj_energy(d_old, uc)
                if np.log(rng.uniform(0, 1)) <= -beta * (u_new - u_old):
                    pos = np.append(without, [moved], axis=0)
                    c["move_accept"] += 1
            continue

        # ----- insertion -----
        if choice == "add":
            c["add_step"] += 1
            c["add_attempt"] += 1
            step += 1
            trial = rng.uniform(0, side, dim)
            if cfg.pore_geometry is not None and not _inside_pore(trial, cfg):
                continue                      # trial hit the wall: failed insertion
            if N > 0:
                tree = cKDTree(pos, boxsize=box)
                d = neighbour_dists(trial, tree, nn)
            else:
                d = np.empty(0)
            if d.size and geometry.overlaps(d, lower).any():
                continue                      # hard core at the floor (both models)
            if hs:
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
            if N == 0:
                continue                      # rejected attempt: nothing to remove
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
    # metropolis has no cavity, so n_cav is not defined; report 0 for a uniform
    # stats schema.
    return {"N": len(pos), "packing_fraction": packing, "counters": c,
            "n_cav": 0.0}


def _lj_energy(dists, rc):
    d = np.asarray(dists, float)
    d = d[(d < rc) & (d > 0)]
    if d.size == 0:
        return 0.0
    return float(np.sum(lj_pair_energy(d)))


def _ljbranch_forward(pool, k, lower, box, rc, beta, rng):
    """Forward (insertion) weight over all non-overlapping k-cliques of the
    pool, one drawn proportional to its Boltzmann weight. The member-vs-system
    factor of each clique is reused from pool.energy_weight (computed in the
    neighbour build); only the intra-clique pairs are added inside the sum.
    Returns (chosen_coords, log W_new) or (None, None) if no clique exists."""
    logW, count, chosen = branch_mod.ljbranch_weight_sum(
        pool.coords, pool.energy_weight, k, lower, box, rc, beta, rng=rng)
    if count == 0 or chosen is None or not np.isfinite(logW):
        return None, None
    return chosen, logW


def _ljbranch_reverse(pool, k, lower, box, rc, beta):
    """Reverse (deletion) weight: the same non-overlapping k-clique sum on the
    current pool, where the removed group is guaranteed to be one member. Same
    enumerator and same weight definition as the forward, so detailed balance
    holds by construction. Returns log W_curr, or -inf if no k-clique exists."""
    logW, count, _ = branch_mod.ljbranch_weight_sum(
        pool.coords, pool.energy_weight, k, lower, box, rc, beta, rng=None)
    if count == 0 or not np.isfinite(logW):
        return -np.inf
    return logW


def _lj_set_energy(coords, rest, box, rc, rest_neighbors=None):
    """Total LJ pair energy contributed by `coords`: each point against `rest`
    within rc, plus pairs among `coords` (relevant only for depth>0, where more
    than one particle is inserted/removed at once). sigma = 1 (reduced units).

    `rest_neighbors` is the live `neighbors.CellList` whose ACTIVE fluid slots are
    exactly `rest`; passing it is what removes the O(N) index build this used to
    do once per clique. Frozen matrix slots are dropped: obstacles interact by
    overlap exclusion only, never through the LJ sum, which is what handing a
    fluid-only `rest` array used to express. Without it, a throwaway cell list is
    built here from `rest`, so every other caller (parallel.py) is unchanged.
    """
    coords = np.asarray(coords, float)
    if len(coords) == 0:
        return 0.0
    box_arr = np.asarray(box, float)
    total = 0.0
    nb = rest_neighbors
    if nb is None and rest is not None and len(rest) > 0:
        rest = np.asarray(rest, float)
        nb = neighbors.CellList.from_positions(rest, box_arr, rest.shape[1], rc)
    if nb is not None and nb.n_active:
        for pt in coords:
            d, slots = nb.query_point(pt, rc, sort=False)
            if d.size:
                d = d[~nb.is_matrix[slots]]
                total += _lj_energy(d, rc)
    if len(coords) > 1:
        ii, jj = np.triu_indices(len(coords), k=1)
        dvec = coords[ii] - coords[jj]
        dvec -= np.round(dvec / box_arr) * box_arr
        total += _lj_energy(np.sqrt((dvec ** 2).sum(axis=1)), rc)
    return total


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
    lower = cfg.lower_cutoff          # the single exclusion floor (diameter, or 0.85 for LJ)
    r_sq = cfg.r_squared
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    n_trials = cfg.no_of_trial_molecules
    depth = cfg.depth
    gdict = cfg.gamma_dict
    beta = cfg.beta
    temp = cfg.temp
    mu = state.mu if getattr(state, "mu", None) is not None else cfg.chem_pot
    nvt = cfg.ensemble_type == "NVT"
    # quenched-annealed frozen matrix: obstacles that enter the overlap test but
    # are never inserted/deleted/moved/counted. Empty => ordinary bulk run.
    matrix = getattr(cfg, "matrix_positions", None)
    has_matrix = matrix is not None and len(matrix) > 0
    total_vol = cfg.total_volume
    ljaux = cfg.potential_type == "ljaux"
    ljbranch = cfg.potential_type == "ljbranch"   # true LJ+branch, no auxiliary
    lj = cfg.potential_type in ("lj", "ljbranch") or ljaux  # LJ physics: floor, translation, tail
    # the hard-core weight is 1 for every admissible trial and reads no distance;
    # every other weigher reads the rows, so only this one may ask for less
    hs_needs_dists = cfg.potential_type != "hard_sphere"
    aux = cfg.potential_type == "aux"
    branch = cfg.branch
    channel = getattr(cfg, "channel", False)   # branch variant: cuboid + sweep
    channel_full = getattr(cfg, "channel_full_span", False)  # wall-to-wall rod
    n_trials_base = n_trials                    # full-span resets n_trials/move
    full_repacking = getattr(cfg, "full_repacking", False)  # whole-cavity repack
    reweight = cfg.potential_type == "reweight"
    if branch and reweight:
        raise NotImplementedError("branch and reweight are alternative biasing "
                                  "schemes; combining them is undefined")

    def rand():
        return rng.random()

    # The one neighbour structure for this window. Built once from the current
    # configuration (frozen obstacles first, so `is_matrix` is a prefix), then
    # updated in place by every accepted move.
    cl = neighbors.CellList.from_positions(
        pos, side, dim, uc, matrix=matrix if has_matrix else None)
    n_mat = len(matrix) if has_matrix else 0

    # Particle state inside this window is a list of SLOT IDS, not a coordinate
    # array. `alive[:N]` are the live fluid slots; coordinates live once, in the
    # cell list, and are read through `cl.coords[slot]`. Removal is a swap with
    # the tail (O(1)); insertion appends. `state.positions` is materialised once,
    # at the end, so io / logging / checkpoints see exactly what they saw before.
    #
    # This changes WHICH particle a given rng draw selects, because the order of
    # `alive` is not the order the old `pos` array happened to have after its
    # np.delete/np.append churn. The selection is uniform over live particles
    # either way, so the sampled distribution is unchanged and only the realised
    # chain differs. Stage-B validation is statistical, not bit-identical.
    N = len(pos)
    alive = np.empty(max(2 * N, N + 1024), dtype=np.int64)
    alive[:N] = np.arange(n_mat, n_mat + N, dtype=np.int64)
    # inverse map, so a membership test that speaks in slot ids can be turned
    # into the swap-remove positions `alive` needs, without a search
    where = np.full(cl.capacity, -1, dtype=np.int64)
    where[alive[:N]] = np.arange(N)

    def alive_push(slot):
        nonlocal N, alive, where
        if N == len(alive):
            alive = np.concatenate([alive, np.empty(len(alive), dtype=np.int64)])
        if slot >= len(where):
            grown = np.full(cl.capacity, -1, dtype=np.int64)
            grown[:len(where)] = where
            where = grown
        alive[N] = slot
        where[slot] = N
        N += 1

    def alive_drop_slots(slots):
        """Swap-remove the given slots. Descending alive-position order, so the
        tail element swapped into a hole is never itself one of the doomed."""
        nonlocal N
        for i in sorted((int(where[s]) for s in slots), reverse=True):
            N -= 1
            moved = alive[N]
            alive[i] = moved
            where[moved] = i

    def commit_exchange(chosen_coords, chosen_slots):
        """Mirror an accepted exchange into the cell list and `alive`. The chosen
        slots were already tombstoned when the proposal was set up and are never
        reactivated, so it is safe for an insert to rebuild and reclaim them;
        that ordering is the cell list's slot-lifetime contract."""
        if chosen_slots is not None:
            alive_drop_slots(chosen_slots)
        for p_ in chosen_coords:
            alive_push(cl.insert(p_))

    # running total of cavity occupancy (spheres found in the trial cuboid/cavity
    # at each exchange), reported as a per-window mean so a run can check the
    # local density n_cav/v_cav against the global N/V.
    _ncav_sum = 0.0
    _ncav_count = 0

    step = 0
    while step < cfg.sampling_window:
        if nvt:
            # NVT: only number-conserving repacking moves; no insertions or
            # deletions, so N stays at the target exactly. An empty system has
            # nothing to repack, so the window completes as a no-op rather than
            # spinning forever on null proposals.
            if N == 0:
                break
            choice = "repack"
        else:
            # One fixed move-type distribution regardless of N. Forcing "add" at
            # N=0 (the old code) dropped the proposal ratio q(0->1)/q(1->0)=4 out
            # of the acceptance and biased P(N=0) by that factor. A move or remove
            # proposed at N=0 is a null / rejected attempt, counted so the window
            # advances as from any state.
            choice = rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ---------------------------------------------------------- translate
        if choice == "move":
            c["move_step"] += 1
            if N == 0:
                continue                      # null proposal: nothing to move
            j = rng.choice(N)
            slot = int(alive[j])
            old = cl.coords[slot].copy()
            if two_d:
                ranf1, ranf2 = rng.uniform(0, 1, 2)
                mx = (old[0] + mf * (ranf1 - 0.5)) % side
                my = (old[1] + mf * (ranf2 - 0.5)) % side
                moved = np.array([mx, my])
            else:
                # three independent displacement components. The old code set the
                # z draw to (ranf1 + ranf2) % 1, pinning every displacement to a
                # fixed 2D surface and starving z mixing; drawing a third uniform
                # frees it. This consumes one extra RNG value per 3D translation,
                # so old checkpoints resume as a valid but different chain.
                ranf1, ranf2, ranf3 = rng.uniform(0, 1, 3)
                mx = (old[0] + mf * (ranf1 - 0.5)) % side
                my = (old[1] + mf * (ranf2 - 0.5)) % side
                mz = (old[2] + mf * (ranf3 - 0.5)) % side
                moved = np.array([mx, my, mz])
            if cfg.pore_geometry is not None and not _inside_pore(moved, cfg):
                continue                      # move into the wall: rejected
            if N == 1 and not has_matrix:
                # lone sphere in an empty box: no neighbours, no energy change,
                # so the standard Metropolis move is always accepted. Short-circuit
                # before any query, which is also what keeps the rng draw count
                # matching the untouched engine on this path.
                cl.move(slot, moved)
                c["move_accept"] += 1
                continue
            # The overlap test must also see the frozen matrix, so a fluid
            # particle cannot move onto an obstacle; the cell list holds both.
            # Deactivating the mover's slot is what `cKDTree(without)` used to
            # express, at O(1) instead of an O(N) rebuild. A radius query returns
            # every neighbour inside the cutoff, so `query_saturating`'s doubling
            # loop has nothing left to guarantee.
            cl.remove(slot)
            accepted = False
            if lj:
                d_new, _ = cl.query_point(moved, uc, inclusive=False)
                if d_new.size and geometry.overlaps(d_new, lower).any():
                    pass
                else:
                    d_old, _ = cl.query_point(old, uc, inclusive=False)

                    def _u(d):
                        if d.size == 0:
                            return 0.0
                        return float(np.sum(lj_pair_energy(d)))
                    dU = _u(d_new) - _u(d_old)
                    accepted = np.log(rng.uniform(0, 1)) <= -dU / temp
            else:
                accepted = cl.count_within(moved, geometry.overlap_radius(lower)) == 0
            if accepted:
                cl.move(slot, moved)          # in place; `alive` does not change
                c["move_accept"] += 1
            else:
                cl.reactivate(slot)
            continue

        # ---------------------------------------------------------- add/remove
        if choice == "repack":
            c["repack_step"] += 1
            n_add = n_remove = depth + 1     # remove k, insert k; N conserved
        elif choice == "add":
            c["add_step"] += 1
            n_add, n_remove = depth + 1, depth
        elif choice == "remove":
            c["remove_step"] += 1
            if N == 0:
                # rejected attempt: nothing to remove, but the exchange step still
                # counts toward the window, matching the metropolis path.
                step += 1
                continue
            n_add, n_remove = depth, depth + 1
        else:
            # "move" is fully handled above and every other value is a bug; fail
            # loudly rather than silently falling through into the remove branch.
            raise RuntimeError(f"unhandled move choice {choice!r}")

        step += 1

        # Proposal geometry differs by depth.
        #
        # depth == 0: the mirror scheme, EOS-validated in this package. The
        # original monolith's depth-0 cavity path fails the hard-sphere EOS
        # (+33% at mu=4.3, density depends on cavity size), so it is not used.
        #
        # depth >= 1: the original BRANCH-MC proposal, transcribed verbatim
        # from cav_n_d_3d_g3_LJ.py and EOS-validated there (depth 1: 290.3,
        # depth 2: 292.5 vs CS 292.9; invariant under cavity radius). Its
        # acceptance algebra is built for exactly this proposal, so no mirror
        # and no generation-probability corrections apply here.
        # Membership is decided from cell gathers, in SLOT space. `idx_cav` is a
        # sorted array of slot ids, not row indices; nothing downstream cared
        # which, and the sort keeps `rng.choice` over it reproducible.
        if depth == 0 and not full_repacking:
            if choice == "add":
                center = rng.uniform(0, side, size=3)[:dim]
                anchor = -1
            else:   # remove or repack: local, anchored on an existing particle
                anchor = int(alive[rng.choice(N)])
                center = (cl.coords[anchor]
                          + rng.uniform(-cfg.cavity_radius, cfg.cavity_radius,
                                        size=dim)) % side
            if n_remove > 0:
                idx_cav = geometry.indices_in_cavity_cl(cl, center,
                                                        cfg.cavity_radius, side,
                                                        two_d)
                n_cav = len(idx_cav)
            else:
                idx_cav = np.empty(0, dtype=int)
                n_cav = 0
            _ncav_sum += n_cav
            _ncav_count += 1
        else:
            if choice == "add":
                center = rng.uniform(0, side, size=3)[:dim]
            else:   # remove or repack: centre on an existing particle so the
                center = cl.coords[int(alive[rng.choice(N)])]  # cavity nonempty
            anchor = -1
            # channel: draw the orientation now so removal membership and the
            # trial cloud share the same oriented body (required for reversibility).
            chan_R = random_rotation_matrix(rng) if channel else None
            chan_chord = None         # full-span: (axis, e1, e2, s_min, s_max)
            if channel and channel_full:
                # wall-to-wall rod along a freely chosen orientation. The length
                # is set by where the line meets the box (box for an axis line,
                # up to box*sqrt(3) for the body diagonal). Membership and the
                # trial count both follow from that chord, so the two directions
                # act on one body and the linear trial density stays fixed.
                axis, e1, e2 = chan_R[:, 0], chan_R[:, 1], chan_R[:, 2]
                s_min, s_max = ray_box_chord(center, axis, box)
                chan_chord = (axis, e1, e2, s_min, s_max)
                chord_len = s_max - s_min
                w = cfg.channel_width
                n_trials = max(2, int(round(cfg.trial_density
                                            * chord_len * w * w)))
                idx_cav, _ = rod_members_full_cl(
                    cl, center, axis, e1, e2, s_min, s_max, 0.5 * w, box)
            elif channel:
                n_trials = n_trials_base
                idx_cav = cuboid_members_cl(cl, center, chan_R,
                                            0.5 * cfg.channel_length,
                                            0.5 * cfg.channel_width, side)
            else:
                idx_cav = geometry.indices_in_cavity_cl(cl, center,
                                                        cfg.cavity_radius, side,
                                                        two_d)
            n_cav = len(idx_cav)
            _ncav_sum += n_cav
            _ncav_count += 1
            if full_repacking:
                # whole-cavity repack: throw away depth and act on everything the
                # cavity holds. add removes all n_cav and inserts one more, remove
                # inserts one fewer, repack keeps the count. An empty cavity has
                # nothing to repack, so add falls to a plain insertion and the
                # rest are rejected. The insertion count can never exceed the
                # trial pool.
                if n_cav == 0:
                    if choice == "add":
                        n_add, n_remove = 1, 0
                    else:
                        continue
                elif choice == "repack":
                    n_add = n_remove = n_cav
                elif choice == "add":
                    n_add, n_remove = n_cav + 1, n_cav
                else:
                    n_add, n_remove = n_cav - 1, n_cav
                if n_add > n_trials:
                    continue     # more insertions than the trial cloud can supply
            elif n_cav == 0:
                # empty cavity. An ADD needs nothing to remove, so it kicks down
                # to a pure depth-0 insertion (n_add=1, n_remove=0) rather than
                # being abandoned -- abandoning it is what drained the fluid at
                # low density, where most global add-centres land in empty space.
                # REMOVE/REPACK genuinely have nothing in the cavity to act on.
                if choice == "add":
                    n_add, n_remove = 1, 0
                else:
                    continue
            elif n_cav < n_remove:
                if choice == "repack":
                    # repack must stay balanced: remove and add the same count,
                    # capped at what the cavity actually holds, so N is conserved
                    n_add = n_remove = n_cav
                elif choice == "add":
                    n_add, n_remove = n_cav, n_cav - 1
                else:
                    n_add, n_remove = n_cav - 1, n_cav

        # Split the system into in-cavity (to remove) and the rest. The rest is
        # never materialised: it is exactly the cell list's active fluid once the
        # chosen slots are tombstoned, and its only other use was a count.
        chosen_slots = None
        n_rest = N - n_remove
        if n_remove == 0:
            disk_in = np.empty((0, dim))
        else:
            if depth == 0 and not full_repacking:
                # the anchor is always removed; any further removals come
                # uniformly from the remaining cavity occupants
                others = idx_cav[idx_cav != anchor]
                if len(others) < n_remove - 1:
                    continue
                chosen_slots = np.empty(n_remove, dtype=np.int64)
                chosen_slots[0] = anchor
                if n_remove > 1:
                    chosen_slots[1:] = rng.choice(others, n_remove - 1,
                                                  replace=False)
            else:
                chosen_slots = rng.choice(idx_cav, n_remove, replace=False)
            disk_in = cl.coords[chosen_slots]

        if n_rest == 0 and has_matrix and n_add == 0:
            # pure deletion with no fluid left: nothing to remove.
            continue
        # trial molecules in the cavity
        if channel and channel_full:
            # trials uniform along the wall-to-wall chord: axial s in [s_min,
            # s_max], square cross-section w x w. Same chord as the membership
            # test, so insertion and removal act on one body.
            axis, e1, e2, s_min, s_max = chan_chord
            b_loc = 0.5 * cfg.channel_width
            s_t = rng.uniform(s_min, s_max, n_trials)
            u_t = rng.uniform(-b_loc, b_loc, n_trials)
            v_t = rng.uniform(-b_loc, b_loc, n_trials)
            trials = np.mod(center + s_t[:, None] * axis
                            + u_t[:, None] * e1 + v_t[:, None] * e2, side)
        elif channel:
            # oriented thin cuboid around the same centre, using the SAME R drawn
            # for the removal-membership test, so insertion and removal act on one
            # body. R is kept so the sweep can recover local coordinates.
            a_loc, b_loc = 0.5 * cfg.channel_length, 0.5 * cfg.channel_width
            U_loc = np.column_stack((
                rng.uniform(-a_loc, a_loc, n_trials),
                rng.uniform(-b_loc, b_loc, n_trials),
                rng.uniform(-b_loc, b_loc, n_trials)))
            trials = np.mod(center + U_loc @ chan_R.T, side)
        else:
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

        # The exclusion set is fluid-rest PLUS frozen matrix: trials and shadows
        # that overlap either are rejected. `cKDTree(disk_rest + matrix)` used to
        # say that with an O(N) rebuild; deactivating the chosen slots says the
        # same thing in O(n_remove). The slots stay tombstoned until the move
        # resolves: an accepted move never wants them back (`commit_exchange`
        # inserts the chosen coordinates instead), and every other exit path,
        # including the several `continue`s below, restores them.
        removed_slots = (chosen_slots if chosen_slots is not None
                         else np.empty(0, dtype=np.int64))
        for _s in removed_slots:
            cl.remove(int(_s))
        committed = False
        try:
            # An empty exclusion set (no fluid left and no matrix) keeps the old
            # `tree is None` path exactly: every trial is admissible with padded
            # neighbour rows, so the first insertion and the last removal go
            # through the ordinary weight and acceptance path.
            tree = cl if cl.n_active else None
            pore_valid = _inside_pore(trials, cfg) if cfg.pore_geometry is not None else None
            # one query for both bodies: the trials and the members share a
            # cuboid, and a query costs numpy dispatch, not arithmetic
            raw, raw_in = geometry.find_neighbors_pair(
                tree, trials, disk_in, lower, uc, nn,
                keep_mask=keep_mask, valid=pore_valid,
                dists_needed=hs_needs_dists)
            if raw is None:
                if n_add != 0:
                    continue       # nothing to insert from: equivalent to rejecting
                # pure deletion: a fully overlapping shadow cloud is a valid (and
                # common) outcome whose reverse weight is the removed molecules
                # alone. Abandoning these moves discards precisely the deletions
                # with the smallest W_old, i.e. the highest acceptance, which
                # suppressed the deletion flux and inflated the density.
                trial_set = TrialSet(np.empty((0, dim)), np.empty(0),
                                     ref_neighbor_coord=np.empty((0, dim)))
                km = np.empty(0, dtype=bool)
            else:
                trial_set, km = _weighted_trials(raw, tree, cfg, potential, choice, N, n_rest)

            if n_remove != 0:
                if raw_in is None:
                    continue
                mol_in_cav, _ = _weighted_trials(raw_in, tree, cfg, potential, choice, N, n_rest)
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

            # channel mode: one sweep over the pool serves both the forward count
            # (Z_{n_add}) and the reverse count (Z_{n_remove}); kmax covers both.
            chan_sweep = None
            if channel and channel_full and len(pool):
                # every pool atom (surviving trial or removed member) lies on the
                # rod, so its axial coordinate is recovered from the same chord. The
                # sweep uses minimum-image box distances and retention eviction, so
                # the wall-to-wall wrap (the seam) is scored exactly.
                axis, e1, e2, s_min, s_max = chan_chord
                ax_pool = axial_coord_full(pool.coords, center, axis,
                                           s_min, s_max, box)
                kmx = max(n_add, n_remove)
                if ljbranch:
                    chan_sweep = ChannelSweepLJFull(pool.coords, ax_pool,
                                                    pool.energy_weight, box,
                                                    lower, uc, beta, kmx)
                else:
                    chan_sweep = ChannelSweepFull(pool.coords, ax_pool, box,
                                                  lower, kmx)
            elif channel and len(pool):
                U_pool = local_from_box(pool.coords, center, chan_R, side)
                kmx = max(n_add, n_remove)
                if ljbranch:
                    chan_sweep = ChannelSweepLJ(U_pool, pool.energy_weight,
                                                lower, uc, beta, kmx)
                else:
                    chan_sweep = ChannelSweep(U_pool, lower, kmx)

            # forward (new configuration)
            if n_add != 0:
                if len(trial_set) < n_add:
                    continue
                if branch:
                    if ljbranch:
                        if channel:
                            sum_new = chan_sweep.log_weight(n_add) if chan_sweep else -np.inf
                            if not np.isfinite(sum_new):
                                continue
                            idx = chan_sweep.sample(n_add, rng)
                            if idx is None:
                                continue
                            chosen_coords = pool.coords[list(idx)]
                        else:
                            # true LJ+branch: Boltzmann-weighted sum over all
                            # non-overlapping k-cliques, one drawn proportional to its
                            # weight. Returns the chosen clique's coords and
                            # log W_new = log sum over cliques.
                            chosen_coords, sum_new = _ljbranch_forward(
                                pool, n_add, lower, box, uc, beta, rng)
                            if chosen_coords is None:
                                continue
                        e_u_r_new = 0.0
                    else:
                        if channel:
                            num_new = chan_sweep.count(n_add) if chan_sweep else 0
                            if num_new == 0:
                                continue
                            idx = chan_sweep.sample(n_add, rng)
                        else:
                            num_new, idx = count_and_sample(pool.coords, n_add,
                                                            lower, box_lengths=box)
                            if num_new == 0:
                                continue
                        chosen_coords = pool.coords[list(idx)]
                        sum_new = np.log(num_new)
                        e_u_r_new = 0.0
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
                    if ljbranch:
                        if channel:
                            sum_current = (chan_sweep.log_weight(n_remove)
                                           if chan_sweep else -np.inf)
                        else:
                            # reverse weight log W_curr: the same non-overlapping
                            # clique sum on the CURRENT pool, with the removed group
                            # guaranteed to be one member. Same enumerator and weight
                            # as the forward, so detailed balance holds.
                            sum_current = _ljbranch_reverse(
                                pool, n_remove, lower, box, uc, beta)
                        e_u_r_current = 0.0
                    else:
                        if channel:
                            num_cur = chan_sweep.count(n_remove) if chan_sweep else 0
                        else:
                            num_cur, _ = count_and_sample(pool.coords, n_remove,
                                                          lower, box_lengths=box)
                        if num_cur <= 0:
                            sum_current = -np.inf
                        else:
                            sum_current = np.log(num_cur)
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

            # LJ tail: a system-level energy change, applied exactly once per move
            # with the same sign in both directions. For branch the per-particle
            # pair energy is inside sum_new / sum_current; for rosenbluth it is
            # inside the Boltzmann trial weights. Either way only the mean-field
            # tail remains as an explicit term.
            if lj:
                n_new = n_rest + len(chosen_coords)
                n_old = n_rest + len(disk_in)
                delta_E_lj = beta * cfg.lj_tail_pref * (n_new ** 2 - n_old ** 2)
            else:
                delta_E_lj = None

            # The depth >= 1 acceptance is the original's, complete as written;
            # the depth-0 mirror scheme likewise needs no extra terms.
            corr = 0.0

            # acceptance
            if choice == "repack":
                c["repack_attempt"] += 1
                # Number-conserving move: remove k, insert k in one step. The
                # chemical-potential prefactor cancels identically (the e^{beta mu}
                # of the k insertions against the e^{-beta mu} of the k deletions,
                # and the V/N number factors), leaving only the configurational
                # Rosenbluth ratio and the energy change. n_add == n_remove makes
                # the k_trials factors of the forward and reverse equal, so they too
                # cancel and no count prefactor remains.
                mu_excess = -sum_new + sum_current
                if ljaux:
                    u_ins = _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl) if len(chosen_coords) else 0.0
                    u_rem = _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl) if len(disk_in) else 0.0
                    delta_E = ((delta_E_lj or 0.0) + (e_u_r_new - e_u_r_current)
                               + beta * (u_ins - u_rem))
                else:
                    delta_E = delta_E_lj if delta_E_lj is not None else (0.0 if lj else (e_u_r_new - e_u_r_current))
                # neg_log_acc is minus the log acceptance ratio, clamped at 0 so the
                # acceptance probability exp(-neg_log_acc) never exceeds 1. The move is
                # accepted when neg_log_acc <= -log(u), u ~ U(0,1).
                neg_log_acc = mu_excess + delta_E - corr
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["repack_accept"] += 1
            elif choice == "add":
                c["add_attempt"] += 1
                mu_ideal = -np.log(total_vol / (N + 1))
                if branch:
                    k_trials = -np.log((n_trials - n_remove) / n_add)
                else:
                    k_trials = -np.log(n_trials - n_remove)
                mu_excess = -sum_new + sum_current - k_trials
                if ljaux:
                    u_ins = _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl) if len(chosen_coords) else 0.0
                    u_rem = _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl) if len(disk_in) else 0.0
                    delta_E = ((delta_E_lj or 0.0) + (e_u_r_new - e_u_r_current)
                               + beta * (u_ins - u_rem))
                else:
                    delta_E = delta_E_lj if delta_E_lj is not None else (0.0 if lj else (e_u_r_new - e_u_r_current))
                neg_log_acc = mu_ideal - beta * mu + mu_excess + delta_E - corr
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["add_accept"] += 1
            else:
                mu_ideal = -np.log(N / total_vol)
                if branch:
                    k_trials = -np.log((n_trials - n_add) / n_remove)
                else:
                    k_trials = -np.log(n_trials - n_add)
                mu_excess = -sum_new + sum_current + k_trials
                if ljaux:
                    u_ins = _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl) if len(chosen_coords) else 0.0
                    u_rem = _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl) if len(disk_in) else 0.0
                    delta_E = ((delta_E_lj or 0.0) + (e_u_r_new - e_u_r_current)
                               + beta * (u_ins - u_rem))
                else:
                    delta_E = delta_E_lj if delta_E_lj is not None else (0.0 if lj else (e_u_r_new - e_u_r_current))
                neg_log_acc = mu_ideal + beta * mu + mu_excess + delta_E - corr
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["remove_accept"] += 1
        finally:
            if not committed:
                for _s in removed_slots:
                    cl.reactivate(int(_s))


    # The window's only materialisation of the configuration. Order follows
    # `alive`, which is not the order the old array had; nothing downstream
    # depends on particle order.
    state.positions = np.array(cl.coords[alive[:N]], dtype=float)
    state.step += cfg.sampling_window
    packing = (N * cfg.volume_of_molecule) / total_vol
    n_cav_mean = (_ncav_sum / _ncav_count) if _ncav_count else 0.0
    return {"N": N, "packing_fraction": packing, "counters": c,
            "n_cav": n_cav_mean}
