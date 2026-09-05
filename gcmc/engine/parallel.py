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

from .. import config_specs
from .._types import BoolArray
from ..proposals.trials import TrialSet
from ..sampling.branch import count_and_sample
from ..sampling.rosenbluth import select_forward, select_reverse
from ..space import decomposition, geometry, neighbors
from ..system.state import zero_counters
from .MC_engine import _lj_energy, _lj_set_energy, _weighted_trials, make_weigher

# Test hook (U4 of cell_unify_plan.md): when true, `advance_cell` verifies after
# every proposal that its cell list still describes exactly the live
# configuration. O(N) per move, so it is off in production and tests turn it on.
DEBUG_CELL_LIST = False

# Test/debug hook: run every cell in this process instead of a pool. One cell is
# one core in production (parallel_input_plan.md D4/D6), so `cfg.n_workers` is
# the cell count and a 27-cell config would otherwise spawn 27 processes inside
# a unit test. Set by tests only; never read from a params file.
IN_PROCESS = False


def _check_cell_list_consistency(cl, main_slots, u_main, border, origin, side, dim):
    """Assert the cell list still describes exactly the live configuration.

    Off by default: `advance_cell` calls this only when DEBUG_CELL_LIST is set,
    because it is O(N) and the hot path is not. Tests flip the flag. Mirrors the
    invariants test_neighbors_parity asserts for the serial engine, plus the two
    that are specific to a cell: the border must still be there, and it must
    still be frozen.
    """
    n_border = cl.n_active - len(main_slots)
    assert len(main_slots) == len(u_main), (
        f"slot bookkeeping desynced from coordinates: "
        f"{len(main_slots)} slots vs {len(u_main)} molecules"
    )
    assert len(set(main_slots.tolist())) == len(main_slots), "duplicate live slot"
    live = set(cl.active_slots().tolist())
    assert set(main_slots.tolist()) <= live, "a live molecule's slot is tombstoned"
    assert n_border >= 0, "border slots went missing from the cell list"
    # every live molecule's stored coordinate is its global position
    if len(main_slots):
        want = (np.atleast_2d(u_main) + origin) % side
        got = np.stack([cl._coordsT[:, int(s)] for s in main_slots])
        assert np.allclose(got, want % side, atol=1e-12), (
            "cell list coordinates drifted from u_main"
        )
    # border molecules are never flagged frozen (they are fluid that sits still,
    # and is_matrix would drop them from the LJ sum), so nothing may be flagged
    assert not cl.is_matrix[: cl._n_slots].any(), (
        "a parallel cell has no frozen matrix; is_matrix would delete the "
        "border's LJ energy"
    )


def advance_cell(
    main,
    border,
    origin,
    n_steps,
    kern,
    samp,
    pot_spec,
    ens,
    potential,
    weigher,
    rng,
    n_offset,
    main_species=None,
    border_species=None,
):
    """Advance one cell by n_steps insertion/deletion proposals.

    main    : (N, dim) global coordinates of the cell's live molecules
    border  : (M, dim) frozen global coordinates within `buffer` of the cell
    origin  : (dim,)  global lower corner of the cell cube
    kern    : KernelSpec   -- the precomputed forms (beta, box_dims, r_squared)
    samp    : SamplerSpec  -- depth, cavity, trial count, move factor
    pot_spec: PotentialSpec -- the overlap floor, the tail flag, the pmf schedule
    ens     : EnsembleSpec -- the state point (mu)
    n_offset: global molecule count outside this cell at window start (for the
              LJ tail, which is a system-level quantity)
    Returns (new_main_global, counters).

    FOUR SUB-CONFIGS RATHER THAN `Config`. `advance_parallel` still takes a
    `Config` -- it must, because `runner` assigns it and `MC_engine.advance` to
    the same name and calls whichever it got. It projects once, at the top, and
    what crosses into the workers from there is these four objects.

    That boundary is where the narrowing pays for itself here. Every task tuple
    is PICKLED to a separate process, so `cfg` was not merely a wide parameter:
    it was a whole `Config` -- species tables, matrix coordinates, IO paths --
    serialised once per cell per window, to be read for about twenty numbers.
    The four specs carry those numbers and nothing else, and `IOSpec` and
    `ExecutionSpec` never cross at all, which is the structural statement that a
    cell kernel cannot depend on where the run writes or how many cores it has.

    Engineering: ONE persistent `neighbors.CellList` answers every neighbour
    question for the whole cell-window, built once here and updated in place by
    every accepted move. It replaces the per-move `cKDTree` this used to build
    from scratch (5.4 ms at N = 18,000 against a ~0.14 ms cell-list
    query+update), and it is the same structure the serial engine runs on:
    exactly one neighbour structure in the hot path.

    The list holds the border ring as well as the live molecules, in GLOBAL
    coordinates over the GLOBAL box, because the physics (distances, overlaps,
    energies) is always evaluated globally; the cell torus applies to proposal
    geometry only and is untouched here.

    Border slots are deliberately NOT flagged `is_matrix`, unlike the frozen
    obstacles of a QA run. A border molecule is ordinary fluid that merely sits
    still for the window: its LJ energy must count, and `is_matrix` is exactly
    the flag `_lj_set_energy` and `_lj_singleparticle_weight` use to DROP a
    neighbour from the energy sum (obstacles interact by exclusion only). Their
    frozenness is enforced structurally instead: `main_slots` holds the live
    molecules and nothing else ever selects, moves, removes, or counts a
    border slot.

    Particle identity inside the window lives in `main_slots`, kept row-aligned
    with `u_main` through every mutation, so the rng still selects the same
    molecule it always did and no draw is added, removed, or reordered.
    Proposals follow the serial tombstone protocol: tombstone the would-be
    removed slots, query, then commit or reactivate in a `finally`, which is
    what keeps the list consistent on every one of the many early `continue`
    exits.
    """
    dim = 2 if kern.two_dim else 3
    box = np.asarray(kern.box_dims, dtype=np.float64)
    # per-axis box and cell length: cubic gives [L, L, L] / [Lc, Lc, Lc], so
    # every `% side`, min-image and insertion draw stays bit-identical to the old
    # scalar path; anisotropic wraps and tiles per axis.
    side = box
    L_c = np.broadcast_to(np.asarray(kern.cell_dims, dtype=np.float64), (dim,))
    cav = samp.cavity_radius
    depth = samp.depth
    lower, uc, nn = pot_spec.lower_cutoff, kern.upper_cutoff, kern.no_neigh
    beta, mu, mf = kern.beta, ens.chem_pot, samp.move_factor
    n_trials = samp.no_of_trial_molecules
    v_cell = float(np.prod(L_c))
    ljaux = kern.potential_type == "ljaux"
    lj = kern.potential_type == "lj" or ljaux
    branch = samp.sampler == "branch"
    if samp.sampler == "metropolis":
        n_trials = 1
    tail = pot_spec.lj_tail_pref if (lj and pot_spec.lj_tail_pref) else 0.0
    # the hard-core weight is 1 for every admissible trial and reads no
    # distance, so that path may ask the query for the nearest neighbour only
    hs_needs_dists = kern.potential_type != "hard_sphere"
    c = zero_counters()

    u_main = (np.asarray(main, float).reshape(-1, dim) - origin) % side
    border = np.asarray(border, float).reshape(-1, dim)

    def to_global(u):
        return (np.atleast_2d(u) + origin) % side

    # ---- the one neighbour structure for this cell-window ----
    # cutoff = uc, the largest radius any query below uses (the overlap floor
    # `lower` is never larger), matching the serial engine's choice.
    n_total = len(u_main) + len(border)
    cl = neighbors.CellList(side, dim, uc, capacity=max(2 * n_total, n_total + 256, 16))
    # Species travel with the coordinates into the list, exactly as in the serial
    # engine, so the worker's reorderings cannot scramble them. None means single
    # species and the list reads type 0 everywhere, which is what it read before
    # the column existed.
    if len(border):
        cl.bulk_insert(border, species=border_species)
    main_slots = (
        cl.bulk_insert(to_global(u_main), species=main_species)
        if len(u_main)
        else np.empty(0, dtype=np.int64)
    )
    cl.rebuild()

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
            moved = to_global(u_new)[0]
            slot = int(main_slots[j])
            # tombstone self: the queries below must see the environment, not
            # the molecule being moved
            cl.remove(slot)
            accepted = False
            try:
                if cl.n_active == 0:
                    accepted = True  # empty environment: nothing to overlap
                elif not lj:
                    if cl.count_within(moved, lower) == 0:
                        accepted = True
                else:
                    old = to_global(u_main[j])[0]
                    # sorted, so these are the same ascending distance arrays
                    # the kd-tree returned (its rows came back distance-sorted
                    # and inf-padded, and the pads were filtered out); a radius
                    # query is complete where the fixed-k one could truncate
                    d_old = cl.query_point(old, uc)[0]
                    d_new = cl.query_point(moved, uc)[0]
                    if d_new.size and np.any(d_new < lower):
                        pass  # overlap: reject WITHOUT drawing, exactly as before
                    else:
                        d_e = _lj_energy(d_new, uc) - _lj_energy(d_old, uc)
                        if rng.uniform(0, 1) < np.exp(-beta * d_e):
                            accepted = True
            finally:
                # `move` reactivates the slot and reindexes it; `reactivate`
                # puts it back untouched. Either way the list is consistent.
                if accepted:
                    cl.move(slot, moved)
                else:
                    cl.reactivate(slot)
            if accepted:
                u_main[j] = u_new
                c["move_accept"] += 1
            if DEBUG_CELL_LIST:
                _check_cell_list_consistency(
                    cl, main_slots, u_main, border, origin, side, dim
                )
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
                rest_slots = main_slots
                doomed = np.empty(0, dtype=np.int64)
            else:
                disk_in_u = u_main[anchor][None, :]
                rest_u = np.delete(u_main, anchor, axis=0)
                rest_slots = np.delete(main_slots, anchor)
                doomed = main_slots[[anchor]]
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
            # mirror geometry, as depth 0 and as the serial engine: a remove must
            # be able to reproduce the cavity an add proposed, which requires an
            # anchor plus a uniform offset over the cavity body. Pinning the
            # centre on the particle makes the two directions' cavity measures
            # mutually singular and biases the density (branch far worse than
            # rosenbluth, because log Z is a global pool property).
            anchor_g = -1
            if choice == "add" or lj:
                u_center = rng.uniform(0.0, L_c, size=dim)
            else:
                anchor_g = int(rng.choice(N))
                u_center = (u_main[anchor_g] + rng.uniform(-cav, cav, size=dim)) % L_c
            d = (u_main - u_center + L_c / 2.0) % L_c - L_c / 2.0
            idx_cav = np.flatnonzero(np.all(np.abs(d) <= cav, axis=1))
            n_cav = len(idx_cav)
            if n_cav == 0:
                # empty cavity: an add bootstraps with a single insertion (as the
                # serial engine does), a remove has nothing to do. Without this an
                # empty cell can never gain its first particle -> N stuck at 0.
                #
                # Restored: this was added in d3165a7 and lost again in 3965558
                # ("Made box and cell(parallel) anisotropic"). Symptom of its
                # absence is unmistakable and is what the regression test below
                # pins: add_step climbs to the full step count while add_attempt
                # stays at exactly 0, because every proposal is `continue`d before
                # it can reach the acceptance test, so N never leaves 0.
                if choice == "add":
                    n_add, n_remove = 1, 0
                else:
                    continue
            elif n_cav < n_remove:
                # capped group size; the two directions must cap to each other's
                # inverse or detailed balance fails. See the long note at the
                # matching branch in MC_engine.advance -- the add used to remove
                # n_cav - 1 and leave one occupant behind, which no remove
                # proposal undoes.
                if choice == "add":
                    n_add, n_remove = n_cav + 1, n_cav
                else:
                    n_add, n_remove = n_cav - 1, n_cav
                if n_add > n_trials:
                    continue
            if n_remove == 0:
                disk_in_u = np.empty((0, dim))
                rest_u = u_main
                rest_slots = main_slots
                doomed = np.empty(0, dtype=np.int64)
            else:
                if anchor_g >= 0:
                    # the anchor is always removed; the rest come uniformly from
                    # the remaining occupants
                    others = idx_cav[idx_cav != anchor_g]
                    if len(others) < n_remove - 1:
                        continue
                    chosen_idx = np.empty(n_remove, dtype=np.int64)
                    chosen_idx[0] = anchor_g
                    if n_remove > 1:
                        chosen_idx[1:] = rng.choice(others, n_remove - 1, replace=False)
                else:
                    chosen_idx = rng.choice(idx_cav, n_remove, replace=False)
                disk_in_u = u_main[chosen_idx]
                keep = np.ones(N, dtype=bool)
                keep[chosen_idx] = False
                rest_u = u_main[keep]
                rest_slots = main_slots[keep]
                doomed = main_slots[chosen_idx]
        disk_in = to_global(disk_in_u) if len(disk_in_u) else np.empty((0, dim))

        # Tombstone the removal set for the whole proposal: every query below
        # must see the system WITHOUT it. Committed moves leave the slots
        # tombstoned for good (an insert may rebuild and reclaim them, which is
        # the cell list's slot-lifetime contract); every other exit path
        # reactivates them in the `finally`.
        for _s in doomed:
            cl.remove(int(_s))
        committed = False
        try:
            # trial cloud on the cell torus around the centre, mapped to global
            trials_u = (rng.uniform(-cav, cav, size=(n_trials, dim)) + u_center) % L_c
            trials = to_global(trials_u)

            keep_mask = np.ones(n_trials, dtype=bool)
            if n_remove > 0:
                keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False

            if cl.n_active == 0:
                # empty environment: every trial survives with unit pair weight
                trial_set = TrialSet(trials, np.ones(n_trials))
                km: BoolArray | None = keep_mask
                mol_in_cav = (
                    TrialSet(disk_in, np.ones(len(disk_in))) if n_remove > 0 else None
                )
            else:
                # one query for the trial cloud and the removal shadow together:
                # both bodies sit in the same cuboid and a cell-list query costs
                # far more in numpy dispatch than in arithmetic
                raw, raw_in = geometry.find_neighbors_pair(
                    cl,
                    trials,
                    disk_in,
                    lower,
                    uc,
                    nn,
                    keep_mask=keep_mask,
                    dists_needed=hs_needs_dists,
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
                        raw, cl, kern, pot_spec, ens, potential, choice, N + n_offset
                    )

                # removed molecule's own weight
                if n_remove > 0:
                    if raw_in is None:
                        continue
                    mol_in_cav, _ = _weighted_trials(
                        raw_in,
                        cl,
                        kern,
                        pot_spec,
                        ens,
                        potential,
                        choice,
                        N + n_offset,
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
                        pool.coords, n_add, lower, box_lengths=box, rng=rng
                    )
                    if num_new <= 0 or idx is None:
                        continue
                    chosen_coords = pool.coords[list(idx)]
                    if lj:
                        u_ins = _lj_set_energy(
                            chosen_coords, None, box, uc, rest_neighbors=cl
                        )
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
                        r_squared=kern.r_squared,
                        gamma_schedule=pot_spec.gamma_dict,
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
                        u_rem = _lj_set_energy(
                            disk_in, None, box, uc, rest_neighbors=cl
                        )
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
                        r_squared=kern.r_squared,
                        gamma_schedule=pot_spec.gamma_dict,
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
                    _lj_set_energy(
                        np.atleast_2d(chosen_coords), None, box, uc, rest_neighbors=cl
                    )
                    if len(chosen_coords)
                    else 0.0
                )
                u_rem = (
                    _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl)
                    if len(disk_in)
                    else 0.0
                )
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
                    committed = True  # set BEFORE any insert: a rebuild may
                    # reclaim the tombstoned slots and they must never be
                    # reactivated afterwards
                    new_u = (np.atleast_2d(chosen_coords) - origin) % side
                    # rest_u, not u_main: at depth >= 1 the move also removed
                    # n_remove molecules, which must not be resurrected
                    u_main = np.vstack([rest_u, new_u]) if len(rest_u) else new_u
                    fresh = [int(cl.insert(g)) for g in to_global(new_u)]
                    main_slots = np.concatenate(
                        [rest_slots, np.asarray(fresh, dtype=np.int64)]
                    )
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
                    committed = True  # see the note above
                    if n_add > 0:
                        # at depth >= 1 a deletion re-inserts n_add molecules
                        re_u = (np.atleast_2d(chosen_coords) - origin) % side
                        u_main = np.vstack([rest_u, re_u]) if len(rest_u) else re_u
                        fresh = [int(cl.insert(g)) for g in to_global(re_u)]
                        main_slots = np.concatenate(
                            [rest_slots, np.asarray(fresh, dtype=np.int64)]
                        )
                    else:
                        u_main = rest_u
                        main_slots = rest_slots
                    c["remove_accept"] += 1
        finally:
            if not committed:
                for _s in doomed:
                    cl.reactivate(int(_s))
            if DEBUG_CELL_LIST:
                _check_cell_list_consistency(
                    cl, main_slots, u_main, border, origin, side, dim
                )

    # The surviving interior, and its species read from the SAME slots, so the two
    # cannot come apart on the way back to the parent. None in, None out: a
    # single-species run's return value is what it always was.
    out_main = to_global(u_main) if len(u_main) else np.empty((0, dim))
    if main_species is None and border_species is None:
        out_species = None
    else:
        out_species = np.asarray(cl.species[main_slots], dtype=np.int64)
    return out_main, c, out_species


def _run_cell(args):
    """Worker entry point (one argument tuple, multiprocessing-friendly):
    build the per-cell rng and weigher, run `advance_cell` on this cell's
    interior particles with its frozen border, and return the updated
    interior plus counters."""
    (
        main,
        border,
        origin,
        n_steps,
        kern,
        samp,
        pot_spec,
        ens,
        potential,
        n_offset,
        seed,
        main_species,
        border_species,
    ) = args
    rng = np.random.default_rng(seed)
    weigher = make_weigher(kern, potential)
    return advance_cell(
        main,
        border,
        origin,
        n_steps,
        kern,
        samp,
        pot_spec,
        ens,
        potential,
        weigher,
        rng,
        n_offset,
        main_species,
        border_species,
    )


def advance_parallel(state, cfg, potential, weigher):
    """Advance one window by domain decomposition (2D or 3D).

    Keeps `Config` in its signature, deliberately: `runner` binds this or
    `MC_engine.advance` to the same name and calls whichever it bound, so the
    two must agree. This is the projection boundary -- `cfg` is read here and
    nowhere below, and the cells receive sub-configs.
    """
    kern = config_specs.kernel_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    pot_spec = config_specs.potential_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    exe = config_specs.execution_spec(cfg)
    io_ = config_specs.io_spec(cfg)

    if samp.sampler not in ("rosenbluth", "branch", "metropolis"):
        raise NotImplementedError(
            "parallel supports metropolis, rosenbluth, and branch; "
            f"got {samp.sampler!r}"
        )
    dim = 2 if kern.two_dim else 3
    rng = state.rng
    pos = np.asarray(state.positions, float).reshape(-1, dim)
    n_global = len(pos)
    # THIS KERNEL IS SINGLE-SPECIES, AND IT SAYS SO RATHER THAN PRETENDING.
    #
    # Nothing in this module draws a species, proposes an identity swap or reads
    # a per-pair sigma -- grep it. So a typed run through here inserts everything
    # as the first species and produces a perfectly valid single-species chain at
    # that species' activity, which is why it reported PASS in the feature matrix
    # while answering a different question from the one asked. Measured on a
    # three-species request at box 16: N grew 453 -> 788 over four windows with
    # composition (N, 0, 0) throughout.
    #
    # `sim_setup` refuses the combination (search: "not multi-species yet "),
    # so this is unreachable through a config. It is stated here anyway because a
    # test or a script can call `advance_parallel` directly, and the failure it
    # would get otherwise is a plausible number rather than an error.
    #
    # What P7 adds is the CELL KERNEL. The plumbing is already here: `grid_up_nd`
    # splits a species column, the reassembly below puts it back, and it fixes
    # the writeback that used to drop it whenever the caller had not supplied one.
    if kern.n_species > 1:
        raise NotImplementedError(
            f"advance_parallel is single-species; this config has "
            f"{kern.n_species}. The cell kernel draws no species and proposes no "
            "identity swap, so the run would silently sample species 0 only. "
            "P7 is the stage that wires it; sim_setup refuses the combination."
        )

    cells, rest, rest_species = decomposition.grid_up_nd(
        pos,
        kern.box_dims,
        kern.cell_dims,
        exe.buffer,
        rng,
        dim,
        n_ax=exe.cell_counts,
        species=state.species,
    )
    n_steps = io_.sampling_window
    seeds = rng.integers(0, 2**31 - 1, size=len(cells))
    tasks = [
        (
            cell.main,
            cell.border,
            cell.origin,
            n_steps,
            kern,
            samp,
            pot_spec,
            ens,
            potential,
            n_global - len(cell.main),
            int(s),
            cell.main_species,
            cell.border_species,
        )
        for cell, s in zip(cells, seeds)
    ]

    total = zero_counters()
    new_mains = []
    new_species = []
    if exe.n_workers > 1 and len(tasks) > 1 and not IN_PROCESS:
        with concurrent.futures.ProcessPoolExecutor(max_workers=exe.n_workers) as ex:
            results = list(ex.map(_run_cell, tasks))
    else:
        results = [_run_cell(t) for t in tasks]
    for new_main, counters, cell_species in results:
        new_mains.append(new_main)
        new_species.append(cell_species)
        for k in total:
            total[k] += counters[k]

    # Reassembly: the species column is concatenated in the SAME cell order as the
    # coordinates, with `rest` last, so a molecule's type follows its coordinates
    # through the round trip. Cells contributing no molecules are skipped on both
    # sides together, which is why the two comprehensions share the `len(m)` test
    # rather than each filtering independently.
    kept = [(m, sp) for m, sp in zip(new_mains, new_species) if len(m)]
    parts = [m for m, _ in kept] + ([rest] if len(rest) else [])
    pos = np.vstack(parts) if parts else np.empty((0, dim))

    if state.species is None:
        spc = None
    else:
        spc_parts = [
            sp if sp is not None else np.zeros(len(m), dtype=np.int64) for m, sp in kept
        ] + ([rest_species] if len(rest) else [])
        spc = np.concatenate(spc_parts) if spc_parts else np.empty(0, dtype=np.int64)
        if len(spc) != len(pos):
            raise AssertionError(
                f"decomposition lost species: {len(spc)} types for {len(pos)} "
                "molecules after reassembly"
            )

    state.positions = pos
    state.species = spc
    state.step += io_.sampling_window * max(1, len(cells))
    packing = (len(pos) * pot_spec.volume_of_molecule) / kern.total_volume
    return {
        "N": len(pos),
        "packing_fraction": packing,
        "counters": total,
        "n_cells": len(cells),
    }
