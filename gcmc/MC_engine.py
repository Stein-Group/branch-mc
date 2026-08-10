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
import os as _os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._types import RNG, BoolArray, FloatArray, IntArray

if TYPE_CHECKING:
    from .potentials.base import Potential
    from .samplers.channel import ChannelSweep, ChannelSweepLJ
    from .sim_setup import Config
    from .state import ChainState
    from .trials import TrialSet
from scipy.spatial import cKDTree

from . import geometry, neighbors
from .potentials.lennard_jones import lj_pair_energy
from .samplers import branch as branch_mod
from .samplers import reweighting
from .samplers.base import EnergyWeigher, ReweightWeigher, UniformWeigher
from .samplers.branch import count_and_sample
from .samplers.channel import (
    ChannelSweep,
    ChannelSweepCyclic,
    ChannelSweepLJ,
    ChannelSweepLJCyclic,
    axial_coord_full,
    cuboid_members_cl,
    local_from_box,
    random_rotation_matrix,
    ray_box_chord,
    rod_members_full_cl,
)
from .samplers.rosenbluth import select_forward, select_reverse
from .state import zero_counters
from .trials import TrialSet

# ---------------------------------------------------------------------------
# VALIDATION RECORD for the short-range (dual-cutoff) auxiliary.
#
# A release gate lived here and refused aux_cutoff < upper_cutoff. It is gone
# because the evidence it was waiting for is in. What that evidence is, so the
# claim can be audited rather than taken on trust:
#
# EXACT BY CONSTRUCTION (no sampling involved)
#   * bit-identical to the no-auxiliary engine at aux_cutoff == upper_cutoff, over
#     2D/3D x fixed/cyclic rod x depth-2/full_repacking x 3 seeds
#   * clique weight sums == brute-force enumeration over all C(n,k) subsets at the
#     SHORT radius, in 2D and 3D, with non-uniform member weights, including
#     minimum-image pairs across the cyclic seam
#   * the same sums == ljbranch_weight_sum, an independent implementation
#   * sample() draws proportional to clique weight, 40k draws against exact
#     per-clique weights
#   * the correction's two energies == a from-scratch minimum-image double loop
#     sharing no code with the engine
#
# SAMPLED DISTRIBUTION UNCHANGED (aux-on == aux-off, seed-wise, 2 sigma, one
# number per seed, each with the stationarity precondition satisfied)
#   NVT  rho* 0.145   z = 0.31   resolution 0.93% of |U|
#   NVT  rho* 0.347   z = 0.44   resolution 0.59%
#   NVT  rho* 0.573   z = 0.06   resolution 0.60%
#   muVT mu = -4      z = 0.58   resolution 8.1% of <N>
#   cutoff sweep at rho* 0.347: 1.5 / 1.8 / 2.0 give z = 0.39 / 0.74 / 0.69 with
#   no ordering among them, so there is no radius-dependent bias in the usable
#   range -- which is the shape a failure to divide out the proposal bias would
#   have had.
#
# KNOWN LIMIT, and it is an efficiency limit rather than a correctness one:
# below about 1.4 sigma the correction's spread explodes and acceptance collapses
# (sd 1.53 kT and acceptance 0.069 at cutoff 1.3, against 0.451 and 0.851 at 1.5).
# The chain then samples nothing while still completing and still looking
# plausible, so sim_setup warns. The default of 1.5 is nearer that edge than
# expected.


def _pore_bounds(cfg: Config) -> tuple[str, float, float]:
    """Centre-accessible bounds for a hard pore inside the cubic cell of side
    cfg.box. Returns (kind, params). The slab/cylinder is centred in the cell.
    A sphere of radius cfg.radius must keep its centre off the wall by that
    radius, so the clearance is cfg.radius, not a hardcoded 0.5."""
    L = cfg.box
    cl = cfg.radius
    if cfg.pore_geometry == "slit":
        assert cfg.pore_H is not None  # config validation pairs geometry+size
        z0 = (L - cfg.pore_H) / 2.0
        return ("slit", z0 + cl, z0 + cfg.pore_H - cl)  # zmin, zmax for centres
    assert cfg.pore_R is not None  # config validation pairs geometry+size
    return ("cylinder", L / 2.0, cfg.pore_R - cl)  # axis centre, rmax for centres


def _inside_pore(coords: FloatArray, cfg: Config) -> BoolArray | bool | None:
    """Boolean mask: which centres satisfy the hard wall. coords is (M, dim) or
    a single (dim,) point. A sphere of radius cfg.radius must keep its centre off
    the wall by that radius (the configurable clearance, not a hardcoded 0.5)."""
    if cfg.pore_geometry is None:
        return None
    a = np.atleast_2d(np.asarray(coords, float))
    kind = cfg.pore_geometry
    cl = cfg.radius
    if kind == "slit":
        assert cfg.pore_H is not None
        L = cfg.box
        z0 = (L - cfg.pore_H) / 2.0
        z = a[:, 2]
        ok = (z >= z0 + cl) & (z <= z0 + cfg.pore_H - cl)
    else:
        assert cfg.pore_R is not None
        L = cfg.box
        cx = cy = L / 2.0
        r = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
        ok = r <= (cfg.pore_R - cl)
    return ok if np.ndim(coords) > 1 else bool(ok[0])


def make_weigher(cfg: Config, potential: Potential | None) -> Any:
    """Pick the selection-weight policy for the configured potential: uniform
    for hard cores (all admissible trials equal), Boltzmann/bias-weighted for
    every soft or auxiliary mode, psi6 for 2D hexatic reweighting. This is the
    only place potential_type maps to a weigher, so a new mode wires in here."""
    match cfg.potential_type:
        case "hard_sphere":
            return UniformWeigher()
        case "lj" | "aux" | "ljaux" | "ljbranch":
            assert potential is not None  # only reweight runs without one
            return EnergyWeigher(potential)
        case "reweight":
            if not cfg.two_dim:
                raise NotImplementedError("reweighting is the 2D hexatic (psi6) method")
            return ReweightWeigher()
        case other:
            raise NotImplementedError(f"engine does not yet wire potential {other!r}")


def _is_matrix_neighbor(container: Any, idxs: IntArray, n_fluid: int) -> BoolArray:
    """Which returned neighbours are frozen obstacles.

    A kd-tree over `[fluid | matrix]` answers this by index: anything at or past
    `n_fluid` is an obstacle (the pad index lands there too, and carries an
    infinite distance). A CellList hands back slot ids, which say nothing about
    ordering, so it answers from its own `is_matrix` flags. Same question, two
    containers."""
    if container is not None and hasattr(container, "matrix_at"):
        return container.matrix_at(idxs)
    return np.asarray(idxs) >= n_fluid


def _lj_singleparticle_weight(
    dists: FloatArray,
    idxs: IntArray,
    potential: Potential,
    n_fluid: int | None = None,
    container: Any = None,
) -> FloatArray:
    """Per-trial Boltzmann factor exp(-beta * sum_j u(r_ij)) over neighbours
    within rc. With `n_fluid` given (QA), frozen-matrix neighbours are dropped
    from the LJ sum (they interact by overlap exclusion only); without it every
    neighbour counts. For a bulk run this equals potential.weight_bulk(dists)."""
    d = np.asarray(dists, float)
    if n_fluid is not None:
        d = np.where(_is_matrix_neighbor(container, idxs, n_fluid), np.inf, d)
    return potential.weight_pair(d)


def _weighted_trials(
    raw: tuple[FloatArray, FloatArray, IntArray, BoolArray | None],
    kdtree: Any,
    cfg: Config,
    potential: Potential | None,
    choice: str,
    n_full: int,
    n_fluid: int | None = None,
) -> tuple[TrialSet, BoolArray | None]:
    """Turn the raw neighbour result into a weighted TrialSet, matching the
    original find_neighbors_fast weighting for the active mode."""
    coords, dists, idxs, km = raw
    g1 = cfg.gamma_dict[1]
    match cfg.potential_type:
        case "hard_sphere":
            return TrialSet(coords, np.ones(len(coords))), km
        case "ljbranch":
            assert potential is not None
            # Reuse: each trial's single-particle LJ weight against the fluid
            # rest (any frozen matrix excludes by overlap only) is the
            # member-vs-system factor every clique needs downstream. Computed
            # once here from the neighbour distances already in hand, never
            # recomputed per clique.
            w = _lj_singleparticle_weight(
                dists, idxs, potential, n_fluid, container=kdtree
            )
            return TrialSet(coords, w), km
        case "ljaux":
            assert potential is not None
            w, ref = potential.weight_bulk(
                dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=g1
            )
            return TrialSet(coords, w, ref_neighbor_coord=ref), km
        case "lj":
            assert potential is not None
            # pair Boltzmann factors only. The long-range tail is a move-level
            # quantity applied once in the acceptance (delta_E_lj); folding it
            # into the per-trial weights put it inside W_old on deletions,
            # where the denominator position inverted its sign and
            # over-accepted deletions by exp(2*beta*|dU_tail|).
            w = cast(FloatArray, potential.weight_bulk(dists))
            return TrialSet(coords, w), km
        case "aux":
            assert potential is not None
            if cfg.two_dim:
                w2d = cast(FloatArray, potential.weight_bulk(dists, gamma=g1))
                return TrialSet(coords, w2d), km
            w, ref = potential.weight_bulk(
                dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=g1
            )
            return TrialSet(coords, w, ref_neighbor_coord=ref), km
        case "reweight":
            # 2D hexatic: unit bulk weight, carry the psi6 phasor accumulators
            cos_sum, sin_sum, counts = reweighting.calculate_order_parameter(
                coords, dists, idxs, kdtree, cfg.box, g1
            )
            return TrialSet(
                coords,
                np.ones(len(coords)),
                cossum=cos_sum,
                sinsum=sin_sum,
                counts=counts,
            ), km
        case other:
            raise NotImplementedError(other)


def _advance_metropolis(
    state: ChainState, cfg: Config, potential: Potential
) -> dict[str, Any]:
    """Standard single-particle grand-canonical Metropolis (textbook GCMC): one
    trial insertion at a uniform-random box point or one deletion of a random
    particle, accepted by the GCMC criterion. No cavity, no multiple trials, no
    depth. Supports hard_core and Lennard-Jones, both in 2D and 3D, using the
    same pair energy and tail as the Rosenbluth path so the two agree at
    equilibrium. (This said "Lennard-Jones (3D)" until 2D LJ was checked and
    found to run: the 2D tail has its own closed form in sim_setup and the pair
    energy is dimension-independent. Worth knowing because 2D LJ metropolis is
    the only available reference for validating the 2D LJ channel, which has
    none -- tests/unit/test_channel_2d.py validates the 2D engine against the
    exact P(N=0)/P(N=1) ratio for HARD CORES only.)
    """
    if cfg.potential_type == "aux":
        raise NotImplementedError(
            "metropolis with a pmf auxiliary is not yet implemented; "
            "use the rosenbluth sampler for pmf runs"
        )
    if cfg.potential_type == "ljaux":
        # with a single trial the selection bias cancels identically, so
        # metropolis + pmf IS metropolis on the plain LJ potential
        from .potentials.aux import LJAux3D

        assert isinstance(potential, LJAux3D)
        potential = potential.lj

    rng = state.rng
    pos = np.asarray(state.positions, float)
    c = zero_counters()
    box = np.asarray(cfg.box_dims, dtype=np.float64)
    # per-axis side (A2.2): for a cubic box this is [L, L, L], so `% side` and
    # rng.uniform(0, side, dim) are bit-identical to the old scalar side (numpy
    # broadcasts the standard-uniform draws the same way); for an orthorhombic
    # box it is the correct per-axis wrap and insertion range.
    side = box
    dim = cfg.dimension
    lower = cfg.lower_cutoff  # exclusion floor (= diameter for hard cores)
    uc = cfg.upper_cutoff
    nn = cfg.no_neigh
    mf = cfg.move_factor
    beta = cfg.beta
    mu = state.mu if state.mu is not None else cfg.chem_pot
    V = cfg.total_volume
    hs = cfg.potential_type == "hard_sphere"

    def neighbour_dists(point: FloatArray, tree: Any, k: int) -> FloatArray:
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
                continue  # null proposal: nothing to move
            j = rng.choice(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            disp = mf * (rng.uniform(0, 1, dim) - 0.5)
            moved = (old + disp) % side
            if cfg.pore_geometry is not None and not _inside_pore(moved, cfg):
                continue  # move into the wall: rejected
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
                continue  # trial hit the wall: failed insertion
            if N > 0:
                tree = cKDTree(pos, boxsize=box)
                d = neighbour_dists(trial, tree, nn)
            else:
                d = np.empty(0)
            if d.size and geometry.overlaps(d, lower).any():
                continue  # hard core at the floor (both models)
            if hs:
                log_bf = 0.0
            else:
                tail = cfg.lj_tail_pref
                assert tail is not None  # set for every LJ config
                u = _lj_energy(d, uc) + tail * (2.0 * N + 1.0)
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
                continue  # rejected attempt: nothing to remove
            j = rng.choice(N)
            chosen = pos[j]
            without = np.delete(pos, j, axis=0)
            if hs:
                log_bf = 0.0
            else:
                tree = cKDTree(without, boxsize=box) if len(without) else None
                d = (
                    neighbour_dists(chosen, tree, nn)
                    if tree is not None
                    else np.empty(0)
                )
                assert cfg.lj_tail_pref is not None  # set for every LJ config
                tail = cfg.lj_tail_pref
                assert tail is not None  # set for every LJ config
                u = _lj_energy(d, uc) + tail * (2.0 * N - 1.0)
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
    return {"N": len(pos), "packing_fraction": packing, "counters": c, "n_cav": 0.0}


def _lj_energy(dists: FloatArray, rc: float) -> float:
    """Total truncated-LJ energy of one particle against a distance list:
    sum of u(r) over 0 < r < rc; zero-distance entries (self-pairs from a
    query that saw the particle itself) and beyond-cutoff entries drop out."""
    d = np.asarray(dists, float)
    d = d[(d < rc) & (d > 0)]
    if d.size == 0:
        return 0.0
    return float(np.sum(lj_pair_energy(d)))


def _ljbranch_forward(
    pool: TrialSet,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
    rng: RNG,
) -> tuple[FloatArray, float] | tuple[None, None]:
    """Forward (insertion) weight over all non-overlapping k-cliques of the
    pool, one drawn proportional to its Boltzmann weight. The member-vs-system
    factor of each clique is reused from pool.energy_weight (computed in the
    neighbour build); only the intra-clique pairs are added inside the sum.
    Returns (chosen_coords, log W_new) or (None, None) if no clique exists."""
    logW, count, chosen = branch_mod.ljbranch_weight_sum(
        pool.coords, pool.energy_weight, k, lower, box, rc, beta, rng=rng
    )
    if count == 0 or chosen is None or not np.isfinite(logW):
        return None, None
    return chosen, logW


def _ljbranch_reverse(
    pool: TrialSet,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
) -> float:
    """Reverse (deletion) weight: the same non-overlapping k-clique sum on the
    current pool, where the removed group is guaranteed to be one member. Same
    enumerator and same weight definition as the forward, so detailed balance
    holds by construction. Returns log W_curr, or -inf if no k-clique exists."""
    logW, count, _ = branch_mod.ljbranch_weight_sum(
        pool.coords, pool.energy_weight, k, lower, box, rc, beta, rng=None
    )
    if count == 0 or not np.isfinite(logW):
        return -np.inf
    return logW


def _lj_set_energy(
    coords: FloatArray,
    rest: FloatArray | None,
    box: FloatArray,
    rc: float,
    rest_neighbors: Any = None,
) -> float:
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
        total += _lj_energy(np.sqrt((dvec**2).sum(axis=1)), rc)
    return total


# ---------------------------------------------------------------- diagnostics
# Drop-in instrumentation, added to find out why the channel sampler grinds at
# high density. Everything here is OFF unless the matching environment variable
# is set. With all of them unset this module is behaviourally identical to the
# original: same RNG stream, same counters, same on-disk schema.
#
#   BRANCH_DIAG=1      collect rejection-reason counters in MC_engine.DIAG
#   BRANCH_DIAG_REC=1  also record per-attempt guard inputs in DIAGV["rec"]
#                      as (choice, n_add, n_remove, len(trial_set), len(pool))
#   POOLGUARD=1        test the forward-selection guard against len(pool)
#                      rather than len(trial_set); see the note at the guard
#   NVT_SKIP_NULL=1    in NVT, a repack that reproduces the incoming
#                      configuration exactly does not count as a step
#
# DIAG/DIAGV/DIAGN are module globals and are NOT reset between windows; a
# driver that wants per-window numbers should clear them itself.
DIAG: dict[str, int] = {}
DIAGV: dict[str, list] = {}
DIAGN: dict[str, list] = {}

_DIAG_ON = bool(_os.environ.get("BRANCH_DIAG"))
_DIAG_REC = bool(_os.environ.get("BRANCH_DIAG_REC"))
_POOLGUARD = bool(_os.environ.get("POOLGUARD"))
_SKIP_NULL = bool(_os.environ.get("NVT_SKIP_NULL"))
_REC_CAP = int(_os.environ.get("BRANCH_DIAG_REC_CAP", "2000000"))

# A jammed rod can have exactly one feasible clique -- the one already there --
# so every repack would be null and an uncounted null would spin the window
# forever. Once this many nulls have gone uncounted in one window, the rest are
# counted normally and DIAG["repack_null_capped"] records that it happened.
_NULL_BUDGET_PER_STEP = int(_os.environ.get("NVT_NULL_BUDGET", "50"))

_CURN = [0]  # system size at the current attempt, for binning events by density


def _d(k: str, n: int = 1) -> None:
    """Count one event, and remember the N it happened at."""
    if not _DIAG_ON:
        return
    DIAG[k] = DIAG.get(k, 0) + n
    DIAGN.setdefault(k, []).append(_CURN[0])


def _dv(k: str, v: float) -> None:
    """Record one scalar sample."""
    if not _DIAG_ON:
        return
    DIAGV.setdefault(k, []).append(float(v))


def _drec(t: tuple) -> None:
    """Record one per-attempt tuple (capped, these lists get long)."""
    if not _DIAG_REC:
        return
    lst = DIAGV.setdefault("rec", [])
    if len(lst) < _REC_CAP:
        lst.append(t)


def _count_reused(chosen: FloatArray, old: FloatArray) -> int:
    """How many of the chosen positions are removed members put back verbatim.

    The pool the clique is drawn from is exactly `surviving trials + removed
    members`, so a chosen position is either a removed member put back or a
    genuinely new trial position -- there is no third case. Counting the first
    therefore splits the clique exactly.

    Exact float comparison is the right test here for the same reason it is in
    `_same_positions`: a reused position is `pool.coords[...]`, which entered
    the pool verbatim as `cl.coords[chosen_slots]`. It is the same float64
    triple, bit for bit, never a recomputed one. Equality means "this is that
    particle", not "this is near that particle".

    Cost is n_add * n_remove float compares (<= 400 even at full repacking),
    which is noise beside the transfer-matrix sweep that produced the clique.
    """
    if len(chosen) == 0 or len(old) == 0:
        return 0
    a = np.asarray(chosen)
    b = np.asarray(old)
    return int(np.count_nonzero(np.all(a[:, None, :] == b[None, :, :], axis=2).any(axis=1)))


def _same_positions(a: FloatArray, b: FloatArray) -> bool:
    """True when two coordinate sets are identical as SETS, ignoring order.

    Exact float comparison is the right test here and not a floating-point sin:
    the chosen coordinates are drawn out of `pool.coords`, and the removed
    members entered that pool verbatim as `cl.coords[chosen_slots]`. A member
    that is put back is the same float64 triple, bit for bit, never a recomputed
    one. So equality means "nothing moved", not "nothing moved much".
    """
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)]))


def _lj_set_energy_split(
    coords: FloatArray,
    box: FloatArray,
    rc_short: float,
    rc_full: float,
    rest_neighbors: Any = None,
) -> tuple[float, float]:
    """(U_short, U_full) for the same set of `coords`, from ONE neighbour query.

    Both are exactly what `_lj_set_energy` returns at the respective radius --
    each point against the live fluid rest, plus pairs among `coords` -- so this
    is a two-for-one, not a different quantity. Frozen matrix slots are dropped
    from both, as there.

    WHY ONE QUERY. The dual-cutoff correction needs the difference between two
    radii, and the wider one already contains every pair the narrower one sees:
    query once at rc_full, then sum twice with two masks. Querying separately
    would double the only expensive part and would also open the door to the two
    sums disagreeing about which neighbours exist, which is precisely the error
    the correction cannot tolerate -- it is a difference of two nearly equal
    numbers, so a single spurious pair does not cancel.

    Note rc_short <= rc_full is required, not checked: the caller is the engine,
    where the validator has already enforced it.
    """
    coords = np.asarray(coords, float)
    if len(coords) == 0:
        return 0.0, 0.0
    box_arr = np.asarray(box, float)
    u_short = u_full = 0.0
    nb = rest_neighbors
    if nb is not None and nb.n_active:
        # ONE BATCHED QUERY for all k points, not k separate query_point calls.
        #
        # This is the auxiliary's whole per-move overhead and it was the reason a
        # real run came out SLOWER than the same run without an auxiliary at high
        # density. Profiling a box-20 fixed-rod full_repacking job at rho* 0.614
        # found ~14 _neighbor_pairs calls per move, of which ~12 came from this
        # loop -- two calls to this function per move (inserted group, removed
        # group), each looping over k points. Meanwhile the DP that the auxiliary
        # makes cheaper was only ~27% of the window: at high density most trials
        # overlap the fluid and are dropped, so the pool is small and the sweep is
        # already cheap. Shrinking a cheap thing while adding k queries per move
        # is a net loss, and that is exactly what the timing showed.
        #
        # `query` pads with inf / data.shape[0], and that pad index is one past the
        # end of `is_matrix`, so the finite mask has to come first.
        d, slots = nb.query(coords, rc_full)
        live = np.isfinite(d)
        if live.any():
            dv = d[live]
            sv = slots[live]
            dv = dv[~nb.is_matrix[sv]]
            u_full += _lj_energy(dv, rc_full)
            u_short += _lj_energy(dv, rc_short)
    if len(coords) > 1:
        ii, jj = np.triu_indices(len(coords), k=1)
        dvec = coords[ii] - coords[jj]
        dvec -= np.round(dvec / box_arr) * box_arr
        dr = np.sqrt((dvec**2).sum(axis=1))
        u_full += _lj_energy(dr, rc_full)
        u_short += _lj_energy(dr, rc_short)
    return u_short, u_full


def advance(
    state: ChainState, cfg: Config, potential: Potential | None, weigher: Any
) -> dict[str, Any]:
    """Run one sampling window of the multi-trial (Rosenbluth / branch /
    channel) engine and return the window summary.

    Physics: each iteration proposes one Monte Carlo move -- a single-particle
    translation, or a grand-canonical exchange that removes `n_remove`
    particles from a local body (cube cavity, oriented cuboid channel, or
    wall-to-wall rod) and inserts `n_add` fresh ones selected from a weighted
    trial cloud -- and accepts it with the detailed-balance ratio assembled at
    the bottom of the loop (ideal-gas factor, chemical potential, Rosenbluth /
    partition-function ratio, energy terms). NVT mode restricts to number-
    conserving moves. `_advance_metropolis` is the independent single-trial
    reference implementation and is deliberately NOT this function.

    Engineering: one persistent CellList answers every neighbour question for
    the window; particle identity lives in slot ids, the live set in the
    `alive` array with O(1) swap-remove, and `state.positions` is materialised
    exactly once, at the end. The proposal/rollback protocol (tombstone the
    would-be-removed slots, query, then commit or reactivate in a finally) is
    what keeps the cell list consistent on EVERY exit path, including the many
    early `continue`s.
    """
    if cfg.metropolis:
        assert potential is not None  # metropolis never runs in reweight mode
        return _advance_metropolis(state, cfg, potential)
    if cfg.potential_type == "reweight" and not cfg.two_dim:
        raise NotImplementedError(
            "3D reweighting (Q6) is not a coherent estimator in the original "
            "and is not wired; reweighting is the 2D hexatic (psi6) method"
        )

    rng = state.rng
    pos = np.asarray(state.positions, float)
    c = zero_counters()

    box = np.asarray(cfg.box_dims, dtype=np.float64)
    # per-axis side: for a cubic box this is [L, L, L], so every `% side`,
    # minimum-image and insertion draw below is bit-identical to the old scalar
    # `cfg.box` (numpy broadcasts the standard-uniform draws the same way); for
    # an orthorhombic box it is the correct per-axis wrap and insertion range.
    side = box
    two_d = cfg.two_dim
    dim = cfg.dimension
    lower = cfg.lower_cutoff  # the single exclusion floor (diameter, or 0.85 for LJ)
    r_sq = cfg.r_squared
    uc = cfg.upper_cutoff
    # SELECTION radius, as opposed to the PHYSICS radius `uc`.
    #
    # With no short-range auxiliary these are the same number and every
    # expression below is byte-for-byte what it was. With one, `rc_sel` is the
    # shorter radius at which trials are proposed and the clique sum is
    # evaluated (dual-cutoff CBMC: Vlugt, Martin, Smit, Siepmann, Krishna,
    # Mol. Phys. 94, 727 (1998)); the mismatch against `uc` is divided back out
    # in the acceptance, so the sampled ensemble is unchanged.
    #
    # WHICH USES OF uc BECOME rc_sel, and why the split falls exactly here:
    #   ->  the member-weight neighbour query, the clique weight sums and the
    #       sweep interaction windows. These are PROPOSAL machinery: any strictly
    #       positive weight function is a legal bias, so a shorter radius costs
    #       no accuracy and is corrected for exactly.
    #   --  the LJ tail, the Metropolis translation acceptance and the cell-list
    #       cell width stay on `uc`. The first two ARE the sampled physics. The
    #       third is a capacity rather than a physics choice: cell width is keyed
    #       to the largest radius any query will use, so a list built at rc_sel
    #       would push the full-potential query onto a 5^dim stencil instead of
    #       3^dim -- slower, for a radius that is not the physics anyway.
    rc_sel = uc if getattr(cfg, "aux_cutoff", None) is None else float(cfg.aux_cutoff)
    nn = cfg.no_neigh
    mf = cfg.move_factor
    n_trials = cfg.no_of_trial_molecules
    depth = cfg.depth
    gdict = cfg.gamma_dict
    beta = cfg.beta
    temp = cfg.temp
    mu = state.mu if state.mu is not None else cfg.chem_pot
    nvt = cfg.ensemble_type == "NVT"
    # quenched-annealed frozen matrix: obstacles that enter the overlap test but
    # are never inserted/deleted/moved/counted. Empty => ordinary bulk run.
    matrix: FloatArray | None = getattr(cfg, "matrix_positions", None)
    has_matrix = matrix is not None and len(matrix) > 0
    total_vol = cfg.total_volume
    ljaux = cfg.potential_type == "ljaux"
    ljbranch = cfg.potential_type == "ljbranch"  # true LJ+branch, no auxiliary
    # The auxiliary is ACTIVE only when a shorter radius is actually configured.
    # At rc_sel == uc the correction below is identically zero, so the flag is
    # kept as `rc_sel < uc` rather than "an auxiliary block exists": that way the
    # no-op case does no arithmetic at all and bit-identity is structural.
    aux_on = ljbranch and rc_sel < uc
    lj = (
        cfg.potential_type in ("lj", "ljbranch") or ljaux
    )  # LJ physics: floor, translation, tail
    # the hard-core weight is 1 for every admissible trial and reads no distance;
    # every other weigher reads the rows, so only this one may ask for less
    hs_needs_dists = cfg.potential_type != "hard_sphere"
    branch = cfg.branch
    channel = getattr(cfg, "channel", False)  # branch variant: cuboid + sweep
    channel_cyclic = getattr(cfg, "channel_cyclic", False)  # wall-to-wall rod
    n_trials_base = n_trials  # full-span resets n_trials/move
    full_repacking = getattr(cfg, "full_repacking", False)  # whole-cavity repack
    reweight = cfg.potential_type == "reweight"
    if branch and reweight:
        raise NotImplementedError(
            "branch and reweight are alternative biasing "
            "schemes; combining them is undefined"
        )

    def rand() -> float:
        return rng.random()

    # The one neighbour structure for this window. Built once from the current
    # configuration (frozen obstacles first, so `is_matrix` is a prefix), then
    # updated in place by every accepted move.
    # GRID PITCH. `CellList`'s `cutoff` argument sets the cell WIDTH and nothing
    # else -- `self.cutoff` is read on exactly one line, to compute `nc` -- and
    # every query passes its own radius explicitly. So this is a pure
    # performance knob with no semantic content.
    #
    # It used to be `uc`, which sizes the grid for the LARGEST radius in the run
    # while the HOTTEST query is the smallest one. The trial cloud is screened at
    # rc_sel, and at 192 trials per exchange that single query is ~60% of the
    # exchange path (cProfile, rho* 0.797). With cw = uc = 2.5 the 27-cell
    # stencil sweeps a volume of 422 to answer a question about a sphere of
    # volume 14 -- 336 candidate distances per trial where ~11 are in range, and
    # ~98% of those trials are about to be discarded as overlapping anyway.
    #
    # Sizing the grid to rc_sel puts reach(rc_sel) back at 1, so the hot query
    # sweeps 98 instead of 422. The uc queries (translations, the dual-cutoff
    # correction) go to reach 2 and sweep 455 instead of 422 -- marginally worse,
    # but there are far fewer of them and they are not the bottleneck.
    #
    # MEASURED at rho* = 0.797, box 20, T 1.5, channel L6 w2.0 depth 2,
    # trial_density 8, auxiliary 1.5: 1.95x per window (median of repeats).
    # With no auxiliary rc_sel == uc and this is the old value exactly, so a
    # non-auxiliary run is untouched.
    #
    # NOT BIT-REPRODUCIBLE against existing runs. A different cell width changes
    # the padded row width k of the neighbour arrays, which changes numpy's
    # pairwise-summation tree inside the weight products, which lands one ulp
    # away -- enough to flip an acceptance and send the chain down a different
    # (equally valid) path. Same sampled distribution, different random walk.
    # Set GRID_PITCH to reproduce an old run exactly.
    _grid = float(_os.environ.get("GRID_PITCH", "0") or 0) or rc_sel
    cl = neighbors.CellList.from_positions(
        pos, side, dim, _grid, matrix=matrix if has_matrix else None
    )
    n_mat = len(matrix) if matrix is not None and has_matrix else 0

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

    def alive_push(slot: int) -> None:
        nonlocal N, alive, where
        if len(alive) == N:
            alive = np.concatenate([alive, np.empty(len(alive), dtype=np.int64)])
        if slot >= len(where):
            grown = np.full(cl.capacity, -1, dtype=np.int64)
            grown[: len(where)] = where
            where = grown
        alive[N] = slot
        where[slot] = N
        N += 1

    def alive_drop_slots(slots: IntArray | list[int]) -> None:
        """Swap-remove the given slots. Descending alive-position order, so the
        tail element swapped into a hole is never itself one of the doomed."""
        nonlocal N
        for i in sorted((int(where[s]) for s in slots), reverse=True):
            N -= 1
            moved = alive[N]
            alive[i] = moved
            where[moved] = i

    def commit_exchange(
        chosen_coords: FloatArray, chosen_slots: IntArray | list[int] | None
    ) -> None:
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
    # dual-cutoff correction statistics for this window. The MEAN should sit near
    # zero (the shell energy cancels between the inserted and removed groups) and
    # the SD is what costs acceptance, so both are reported: a drifting mean means
    # the two groups are not being drawn from comparable surroundings, which is a
    # different bug from a merely wide distribution.
    _corr_sum = 0.0
    _corr_sq = 0.0
    _corr_count = 0

    step = 0
    # see the NULL REPACK note in the repack acceptance branch
    null_budget = _NULL_BUDGET_PER_STEP * cfg.sampling_window if _SKIP_NULL else 0

    # CLIQUE COMPOSITION -- how much of each drawn clique is genuinely new.
    #
    # RAW SUMS, NEVER MEANS. These are three integers per direction and the log
    # writes the integers, not a ratio. That is the whole point: a per-window
    # mean would make any coarser reporting interval a mean of means, weighted
    # by nothing meaningful (windows differ in how many cliques they draw, and
    # under full repacking they differ in clique SIZE too). With sums, the
    # statistic over any span of rows -- one window, a thousand, the whole run,
    # a resumed run stitched across a walltime kill -- is
    #
    #     fraction new  = sum(clq*_new) / sum(clq*_pos)
    #     new per clique= sum(clq*_new) / sum(clq*_n)
    #
    # summing numerator and denominator separately, which is exact. `avg_n_cav`
    # below is the counter-example: it is a per-window mean and the runner then
    # takes a running mean OF those, so it is already a mean of means and must
    # not be used as a template.
    #
    # Counted where the acceptance is computed, so clq{A,R}_n match add_attempt
    # / the remove equivalent exactly -- a free cross-check on the log.
    _clq = {
        "A_n": 0, "A_pos": 0, "A_new": 0,
        "R_n": 0, "R_pos": 0, "R_new": 0,
        "P_n": 0, "P_pos": 0, "P_new": 0,
    }
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
                continue  # null proposal: nothing to move
            j = rng.choice(N)
            slot = int(alive[j])
            old = cl.coords[slot].copy()
            if two_d:
                ranf1, ranf2 = rng.uniform(0, 1, 2)
                mx = (old[0] + mf * (ranf1 - 0.5)) % side[0]
                my = (old[1] + mf * (ranf2 - 0.5)) % side[1]
                moved = np.array([mx, my])
            else:
                # three independent displacement components. The old code set the
                # z draw to (ranf1 + ranf2) % 1, pinning every displacement to a
                # fixed 2D surface and starving z mixing; drawing a third uniform
                # frees it. This consumes one extra RNG value per 3D translation,
                # so old checkpoints resume as a valid but different chain.
                ranf1, ranf2, ranf3 = rng.uniform(0, 1, 3)
                mx = (old[0] + mf * (ranf1 - 0.5)) % side[0]
                my = (old[1] + mf * (ranf2 - 0.5)) % side[1]
                mz = (old[2] + mf * (ranf3 - 0.5)) % side[2]
                moved = np.array([mx, my, mz])
            if cfg.pore_geometry is not None and not _inside_pore(moved, cfg):
                continue  # move into the wall: rejected
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

                    def _u(d: FloatArray) -> float:
                        if d.size == 0:
                            return 0.0
                        return float(np.sum(lj_pair_energy(d)))

                    dU = _u(d_new) - _u(d_old)
                    accepted = np.log(rng.uniform(0, 1)) <= -dU / temp
            else:
                accepted = cl.count_within(moved, geometry.overlap_radius(lower)) == 0
            if accepted:
                cl.move(slot, moved)  # in place; `alive` does not change
                c["move_accept"] += 1
            else:
                cl.reactivate(slot)
            continue

        # ---------------------------------------------------------- add/remove
        if choice == "repack":
            # NOTE on `repack` vs `full_repacking`, which are easy to confuse and
            # are unrelated. `repack` is a MOVE TYPE and is proposed only in NVT
            # (see the choice dispatch above): it removes k and inserts k, so N is
            # conserved, which is what NVT requires. `full_repacking` is a SIZE
            # setting, valid in either ensemble, which throws away `depth` and acts
            # on the whole cavity occupancy instead. So in muVT the repack_*
            # counters are legitimately zero even with full_repacking on -- muVT
            # draws only move/add/remove, and full_repacking there means "exchange
            # as many particles as the cavity holds", not "repack".
            c["repack_step"] += 1
            n_add = n_remove = depth + 1  # remove k, insert k; N conserved
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
                center = rng.uniform(0.0, 1.0, size=3)[:dim] * side
                anchor = -1
            else:  # remove or repack: local, anchored on an existing particle
                anchor = int(alive[rng.choice(N)])
                center = (
                    cl.coords[anchor]
                    + rng.uniform(-cfg.cavity_radius, cfg.cavity_radius, size=dim)
                ) % side
            if n_remove > 0:
                idx_cav = geometry.indices_in_cavity_cl(
                    cl, center, cfg.cavity_radius, side, two_d
                )
                n_cav = len(idx_cav)
            else:
                idx_cav = np.empty(0, dtype=int)
                n_cav = 0
            _ncav_sum += n_cav
            _ncav_count += 1
        else:
            # depth >= 1 uses the SAME mirror geometry as depth 0, for the same
            # reason. An `add` centres the cavity uniformly; a `remove` must
            # therefore be able to produce that identical cavity, which it can
            # only do by drawing an anchor and offsetting it uniformly over the
            # cavity body. Pinning the centre exactly ON a particle (what the
            # original did here) makes the two directions' cavity measures
            # mutually singular: no reverse path reproduces the forward cavity.
            # The bias that follows is small for rosenbluth, whose weights are
            # path-local, and large for branch, whose log Z is a global property
            # of the whole pool -- measured +0.0069 in eta at 2D mu = 1, L = 2,
            # against +0.0012 for rosenbluth. See the derivation note below.
            #
            # The orientation is drawn BEFORE the centre because an oriented body
            # must be offset in its own frame: for the channel the mirror offset
            # runs along the cuboid's local axes, not the lab axes.
            chan_R = random_rotation_matrix(rng, dim) if channel else None
            anchor_d1 = -1
            if choice == "add":
                center = rng.uniform(0.0, 1.0, size=3)[:dim] * side
            else:
                anchor_d1 = int(alive[rng.choice(N)])
                _base = cl.coords[anchor_d1]
                if channel and channel_cyclic:
                    # wall-to-wall rod: the axial direction is DEGENERATE (any
                    # axial shift of the centre names the same rod, the same
                    # members and the same trial region), so only the
                    # cross-section is offset. The resulting measure ratio is
                    # A_perp / w^(dim-1) = V / v, identical to the plain cavity.
                    assert chan_R is not None
                    _b = 0.5 * cfg.channel_width
                    _off = rng.uniform(-_b, _b) * chan_R[:, 1]
                    if dim == 3:
                        _off = _off + rng.uniform(-_b, _b) * chan_R[:, 2]
                    center = (_base + _off) % side
                elif channel:
                    assert chan_R is not None
                    _a, _b = 0.5 * cfg.channel_length, 0.5 * cfg.channel_width
                    _u = np.empty(dim)
                    _u[0] = rng.uniform(-_a, _a)
                    _u[1:] = rng.uniform(-_b, _b, dim - 1)
                    center = (_base + _u @ chan_R.T) % side
                else:
                    center = (
                        _base
                        + rng.uniform(-cfg.cavity_radius, cfg.cavity_radius, size=dim)
                    ) % side
            anchor = anchor_d1 if choice != "add" else -1
            chan_chord: (
                tuple[FloatArray, FloatArray, FloatArray | None, float, float] | None
            )
            chan_chord = None  # full-span: (axis, e1, e2, s_min, s_max)
            if channel and channel_cyclic:
                assert chan_R is not None  # channel implies a drawn rotation
                # wall-to-wall rod along a freely chosen orientation. The length
                # is set by where the line meets the box (box for an axis line,
                # up to box*sqrt(dim) for the body diagonal). Membership and the
                # trial count both follow from that chord, so the two directions
                # act on one body and the linear trial density stays fixed. In 2D
                # the cross-section has one perpendicular axis e1 (e2 is None).
                axis, e1 = chan_R[:, 0], chan_R[:, 1]
                e2 = chan_R[:, 2] if dim == 3 else None
                s_min, s_max = ray_box_chord(center, axis, box)
                chan_chord = (axis, e1, e2, s_min, s_max)
                chord_len = s_max - s_min
                w = cfg.channel_width
                n_trials = max(
                    2, int(round(cfg.trial_density * chord_len * w ** (dim - 1)))
                )
                idx_cav, _ = rod_members_full_cl(
                    cl, center, axis, e1, e2, s_min, s_max, 0.5 * w, box
                )
            elif channel:
                assert chan_R is not None  # channel implies a drawn rotation
                n_trials = n_trials_base
                idx_cav = cuboid_members_cl(
                    cl,
                    center,
                    chan_R,
                    0.5 * cfg.channel_length,
                    0.5 * cfg.channel_width,
                    side,
                )
            else:
                idx_cav = geometry.indices_in_cavity_cl(
                    cl, center, cfg.cavity_radius, side, two_d
                )
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
                    continue  # more insertions than the trial cloud can supply
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
                # The cavity holds fewer than `depth`, so the group size is
                # capped at what is actually there. The two directions MUST cap
                # to each other's inverse: an add that removes n_cav and inserts
                # n_cav + 1 leaves occupancy n_cav + 1, and the remove that
                # undoes it removes all n_cav + 1 and inserts n_cav -- which is
                # exactly the `else` branch evaluated at the new occupancy, and
                # the same condition (n_cav < depth) holds on both sides.
                #
                # This used to read `n_add, n_remove = n_cav, n_cav - 1` for the
                # add: it removed n_cav - 1 and left ONE occupant behind, so its
                # reverse would have to remove n_cav and insert n_cav - 1, a move
                # the remove branch never proposes. The add and remove fallbacks
                # were therefore not inverses and detailed balance failed. It
                # fires whenever n_cav < depth, so it is invisible at depth 1
                # (where n_cav == 0 is handled above and nothing else can
                # trigger it) and dominates at large depth: measured -0.0059 in
                # eta at 2D mu = 1, depth 3, L = 2. Note the n_cav == 0 add rule
                # above, (1, 0), and full_repacking both already use this
                # convention -- the old add fallback was the odd one out.
                if choice == "repack":
                    # repack must stay balanced: remove and add the same count,
                    # capped at what the cavity actually holds, so N is conserved
                    n_add = n_remove = n_cav
                elif choice == "add":
                    n_add, n_remove = n_cav + 1, n_cav
                else:
                    n_add, n_remove = n_cav - 1, n_cav
                if n_add > n_trials:
                    _d("bail_nadd_gt_ntrials")
                    continue  # more insertions than the trial cloud can supply

        # Split the system into in-cavity (to remove) and the rest. The rest is
        # never materialised: it is exactly the cell list's active fluid once the
        # chosen slots are tombstoned, and its only other use was a count.
        _CURN[0] = N
        _d("attempt")
        chosen_slots = None
        n_rest = N - n_remove
        if n_remove == 0:
            disk_in = np.empty((0, dim))
        else:
            if anchor >= 0 and not full_repacking:
                # the anchor is always removed; any further removals come
                # uniformly from the remaining cavity occupants
                others = idx_cav[idx_cav != anchor]
                if len(others) < n_remove - 1:
                    _d("bail_anchor_others")
                    continue
                chosen_slots = np.empty(n_remove, dtype=np.int64)
                chosen_slots[0] = anchor
                if n_remove > 1:
                    chosen_slots[1:] = rng.choice(others, n_remove - 1, replace=False)
            else:
                chosen_slots = rng.choice(idx_cav, n_remove, replace=False)
            disk_in = cl.coords[chosen_slots]

        if n_rest == 0 and has_matrix and n_add == 0:
            # pure deletion with no fluid left: nothing to remove.
            _d("bail_pure_del_empty")
            continue
        # trial molecules in the cavity
        if channel and channel_cyclic:
            # trials uniform along the wall-to-wall chord: axial s in [s_min,
            # s_max], cross-section w per perpendicular axis. Same chord as the
            # membership test, so insertion and removal act on one body. In 2D
            # there is a single perpendicular axis e1 (e2 is None).
            assert chan_chord is not None  # full-span channel drew the chord
            axis, e1, e2, s_min, s_max = chan_chord
            b_loc = 0.5 * cfg.channel_width
            s_t = rng.uniform(s_min, s_max, n_trials)
            u_t = rng.uniform(-b_loc, b_loc, n_trials)
            pts = center + s_t[:, None] * axis + u_t[:, None] * e1
            if e2 is not None:
                v_t = rng.uniform(-b_loc, b_loc, n_trials)
                pts = pts + v_t[:, None] * e2
            trials = np.mod(pts, side)
        elif channel:
            # oriented thin cuboid around the same centre, using the SAME R drawn
            # for the removal-membership test, so insertion and removal act on one
            # body. R is kept so the sweep can recover local coordinates.
            a_loc, b_loc = 0.5 * cfg.channel_length, 0.5 * cfg.channel_width
            # long axis plus (dim-1) cross-section widths: a cuboid w x w in 3D,
            # a rectangle with a single width w in 2D.
            cols = [rng.uniform(-a_loc, a_loc, n_trials)]
            cols += [rng.uniform(-b_loc, b_loc, n_trials) for _ in range(dim - 1)]
            U_loc = np.column_stack(cols)
            assert chan_R is not None  # channel implies a drawn rotation
            trials = np.mod(center + U_loc @ chan_R.T, side)
        else:
            trial_nums = rng.uniform(
                -cfg.cavity_radius, cfg.cavity_radius, size=(dim, n_trials)
            )
            tx = (trial_nums[0] + center[0]) % side[0]
            ty = (trial_nums[1] + center[1]) % side[1]
            if two_d:
                trials = np.column_stack((tx, ty))
            else:
                tz = (trial_nums[2] + center[2]) % side[2]
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
        removed_slots = (
            chosen_slots if chosen_slots is not None else np.empty(0, dtype=np.int64)
        )
        for _s in removed_slots:
            cl.remove(int(_s))
        committed = False
        try:
            # An empty exclusion set (no fluid left and no matrix) keeps the old
            # `tree is None` path exactly: every trial is admissible with padded
            # neighbour rows, so the first insertion and the last removal go
            # through the ordinary weight and acceptance path.
            tree = cl if cl.n_active else None
            pore_valid = cast(  # trials is (M, dim): the batch arm of _inside_pore
                "BoolArray | None",
                _inside_pore(trials, cfg) if cfg.pore_geometry is not None else None,
            )
            # one query for both bodies: the trials and the members share a
            # cuboid, and a query costs numpy dispatch, not arithmetic
            raw, raw_in = geometry.find_neighbors_pair(
                tree,
                trials,
                disk_in,
                lower,
                rc_sel,  # member weights are a proposal quantity: aux radius
                nn,
                keep_mask=keep_mask,
                valid=pore_valid,
                dists_needed=hs_needs_dists,
            )
            if raw is None:
                if n_add != 0:
                    _d("bail_raw_none_all_trials_overlap")
                    continue  # nothing to insert from: equivalent to rejecting
                # pure deletion: a fully overlapping shadow cloud is a valid (and
                # common) outcome whose reverse weight is the removed molecules
                # alone. Abandoning these moves discards precisely the deletions
                # with the smallest W_old, i.e. the highest acceptance, which
                # suppressed the deletion flux and inflated the density.
                trial_set = TrialSet(
                    np.empty((0, dim)),
                    np.empty(0),
                    ref_neighbor_coord=np.empty((0, dim)),
                )
                km: BoolArray | None = np.empty(0, dtype=bool)
            else:
                trial_set, km = _weighted_trials(
                    raw, tree, cfg, potential, choice, N, n_rest
                )

            if n_remove != 0:
                if raw_in is None:
                    _d("bail_rawin_none")
                    continue
                mol_in_cav, _ = _weighted_trials(
                    raw_in, tree, cfg, potential, choice, N, n_rest
                )
                if len(mol_in_cav) != n_remove:
                    _d("bail_molincav_mismatch")
                    continue
            else:
                mol_in_cav = TrialSet(np.empty((0, dim)), np.empty(0))

            # assemble the pool the forward selection draws from
            if n_remove != 0:
                assert km is not None  # raw was handled above; km rides with it
                pool = trial_set.mask(km).merge(mol_in_cav)
                pool = pool[rng.permutation(len(pool))]
            else:
                pool = trial_set.copy()

            # channel mode: one sweep over the pool serves both the forward count
            # (Z_{n_add}) and the reverse count (Z_{n_remove}); kmax covers both.
            chan_sweep: ChannelSweep | ChannelSweepLJ | None = None
            if channel and channel_cyclic and len(pool):
                # every pool atom (surviving trial or removed member) lies on the
                # rod, so its axial coordinate is recovered from the same chord. The
                # sweep uses minimum-image box distances and retention eviction, so
                # the wall-to-wall wrap (the seam) is scored exactly.
                assert chan_chord is not None  # full-span channel drew the chord
                axis, e1, e2, s_min, s_max = chan_chord
                ax_pool = axial_coord_full(pool.coords, center, axis, s_min, s_max, box)
                kmx = max(n_add, n_remove)
                if ljbranch:
                    chan_sweep = ChannelSweepLJCyclic(
                        pool.coords,
                        ax_pool,
                        pool.energy_weight,
                        box,
                        lower,
                        rc_sel,  # DP interaction window: aux radius
                        beta,
                        kmx,
                    )
                else:
                    chan_sweep = ChannelSweepCyclic(
                        pool.coords, ax_pool, box, lower, kmx
                    )
            elif channel and len(pool):
                assert chan_R is not None  # channel implies a drawn rotation
                U_pool = local_from_box(pool.coords, center, chan_R, side)
                kmx = max(n_add, n_remove)
                _dv("kmax", kmx)
                _dv("pool_size", len(pool))
                _dv("trials_admissible", len(trial_set))
                _dv("n_remove", n_remove)
                if ljbranch:
                    chan_sweep = ChannelSweepLJ(
                        U_pool, pool.energy_weight, lower, rc_sel, beta, kmx
                    )
                else:
                    chan_sweep = ChannelSweep(U_pool, lower, kmx)

            # forward (new configuration)
            if n_add != 0:
                # THE GUARD.
                #
                # This tested `len(trial_set)`, the surviving random trials. But
                # the DP does not draw from trial_set, it draws from `pool` =
                # trial_set.mask(km) + mol_in_cav, i.e. the surviving trials PLUS
                # the members just tombstoned. Those members are real particle
                # positions and so are guaranteed-valid placements; the guard did
                # not count them. Measured at rho* 0.756: median len(trial_set) 6,
                # median len(pool) 13, median n_add 10.
                #
                # It is also direction-dependent. n_add is n_cav+1 for an
                # insertion and n_cav-1 for a deletion, so at the same rod
                # occupancy the insertion faces a threshold two higher. Measured
                # firing rates over 3990 attempts: 88.1% of insertions against
                # 68.0% of deletions, a gap of 13-48 points at every occupancy.
                # A rejection rule that fires more often in one direction than
                # the other is a detailed-balance defect, and this one fires on
                # most attempts.
                #
                # `len(pool) < n_add` is not a new rule: it is exactly the DP's
                # own feasibility condition, made cheap. When it holds,
                # log_weight(n_add) is -inf anyway, and THAT test is symmetric
                # because both directions read one shared sweep object.
                _drec(
                    (choice, int(n_add), int(n_remove), int(len(trial_set)), int(len(pool)))
                )
                if (len(pool) if _POOLGUARD else len(trial_set)) < n_add:
                    _d("bail_guard_lt_nadd")
                    continue
                if branch:
                    if ljbranch:
                        if channel:
                            if chan_sweep is None:
                                continue  # empty pool: no clique, reject
                            assert isinstance(chan_sweep, ChannelSweepLJ)
                            sum_new = chan_sweep.log_weight(n_add)
                            if not np.isfinite(sum_new):
                                _d("bail_logZ_infeasible_LJ")
                                continue
                            idx: Sequence[int] | None = chan_sweep.sample(n_add, rng)
                            if idx is None:
                                _d("bail_sample_none_LJ")
                                continue
                            chosen_coords = pool.coords[list(idx)]
                        else:
                            # true LJ+branch: Boltzmann-weighted sum over all
                            # non-overlapping k-cliques, one drawn proportional to its
                            # weight. Returns the chosen clique's coords and
                            # log W_new = log sum over cliques.
                            _cc, _lw = _ljbranch_forward(
                                pool, n_add, lower, box, rc_sel, beta, rng
                            )
                            if _cc is None or _lw is None:
                                continue
                            chosen_coords, sum_new = _cc, _lw
                        e_u_r_new = 0.0
                    else:
                        if channel:
                            if chan_sweep is None:
                                continue  # empty pool: no clique, reject
                            assert isinstance(chan_sweep, ChannelSweep)
                            num_new = chan_sweep.count(n_add)
                            if num_new == 0:
                                _d("bail_count_zero_HS")
                                continue
                            idx = chan_sweep.sample(n_add, rng)
                            assert idx is not None  # num_new > 0
                        else:
                            num_new, idx = count_and_sample(
                                pool.coords, n_add, lower, box_lengths=box, rng=rng
                            )
                            if num_new == 0 or idx is None:
                                continue
                        chosen_coords = pool.coords[list(idx)]
                        sum_new = np.log(num_new)
                        e_u_r_new = 0.0
                else:
                    chosen, sum_new_arr = select_forward(
                        pool,
                        n_add,
                        weigher,
                        rand=rand,
                        side=side,
                        r_squared=r_sq,
                        gamma_schedule=gdict,
                    )
                    if chosen is None or sum_new_arr is None:
                        continue
                    sum_new = np.sum(np.log(sum_new_arr))
                    if reweight:
                        e_u_r_new = np.sum(
                            np.log(
                                reweighting.psi6_magnitude(
                                    chosen.cossum, chosen.sinsum, chosen.counts
                                )
                            )
                        )
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
                            if chan_sweep is not None:
                                assert isinstance(chan_sweep, ChannelSweepLJ)
                                sum_current = chan_sweep.log_weight(n_remove)
                            else:
                                sum_current = -np.inf
                        else:
                            # reverse weight log W_curr: the same non-overlapping
                            # clique sum on the CURRENT pool, with the removed group
                            # guaranteed to be one member. Same enumerator and weight
                            # as the forward, so detailed balance holds.
                            sum_current = _ljbranch_reverse(
                                pool, n_remove, lower, box, rc_sel, beta
                            )
                        e_u_r_current = 0.0
                    else:
                        if channel:
                            if chan_sweep is not None:
                                assert isinstance(chan_sweep, ChannelSweep)
                                num_cur = chan_sweep.count(n_remove)
                            else:
                                num_cur = 0
                        else:
                            num_cur, _ = count_and_sample(
                                pool.coords, n_remove, lower, box_lengths=box
                            )
                        if num_cur <= 0:
                            sum_current = -np.inf
                        else:
                            sum_current = np.log(num_cur)
                        e_u_r_current = 0.0
                else:
                    assert km is not None  # raw was handled above; km rides with it
                    _trimmed = trial_set.mask(km)
                    trimmed: TrialSet | None = _trimmed if len(_trimmed) != 0 else None
                    sum_cur_arr, e_cur_arr = select_reverse(
                        trimmed,
                        mol_in_cav,
                        n_remove,
                        weigher,
                        side=side,
                        r_squared=r_sq,
                        gamma_schedule=gdict,
                    )
                    sum_current = np.sum(np.log(sum_cur_arr))
                    e_u_r_current = np.sum(np.log(e_cur_arr))
            else:
                sum_current = 0.0
                e_u_r_current = 0.0

            if branch and not np.isfinite(sum_current):
                continue

            # clique composition, recorded once per scored move (see _clq above)
            _ck = {"add": "A", "remove": "R"}.get(choice, "P")
            _clq[_ck + "_n"] += 1
            _clq[_ck + "_pos"] += len(chosen_coords)
            _clq[_ck + "_new"] += len(chosen_coords) - _count_reused(
                chosen_coords, disk_in
            )

            # LJ tail: a system-level energy change, applied exactly once per move
            # with the same sign in both directions. For branch the per-particle
            # pair energy is inside sum_new / sum_current; for rosenbluth it is
            # inside the Boltzmann trial weights. Either way only the mean-field
            # tail remains as an explicit term.
            if lj:
                tail = cfg.lj_tail_pref
                assert tail is not None  # set for every LJ config
                n_new = n_rest + len(chosen_coords)
                n_old = n_rest + len(disk_in)
                delta_E_lj = beta * tail * (n_new**2 - n_old**2)
            else:
                delta_E_lj = None

            # DUAL-CUTOFF CORRECTION (Vlugt, Martin, Smit, Siepmann, Krishna,
            # Mol. Phys. 94, 727 (1998)).
            #
            # The clique sums above were built at rc_sel, so the chosen group was
            # drawn with probability w~_C / W~_new where w~_C = exp(-beta U_C^aux).
            # Writing out detailed balance for the exchange, the group's Boltzmann
            # factor no longer cancels against its own selection probability; what
            # survives is exactly the mismatch between the two radii:
            #
            #   Delta_corr = beta * [ (U_ins^uc - U_ins^aux)
            #                       - (U_rem^uc - U_rem^aux) ]
            #
            # It enters neg_log_acc with the SAME sign and in the same place as
            # the tail, and it is the only new term: the combinatorial k_trials
            # factors, the mu terms and the Rosenbluth ratio are untouched,
            # because a shorter radius changes what gets PROPOSED, not the space
            # of moves or their multiplicities.
            #
            # U here means "group against the live fluid rest, plus pairs within
            # the group" -- the same quantity the member weights and the intra-
            # clique factor together represent, which is what makes the
            # cancellation exact rather than approximate. The removed group's
            # slots are tombstoned at this point, so `rest` is the same set on
            # both sides and the inserted/removed cross term correctly does not
            # appear.
            #
            # MEAN vs FLUCTUATION: the absolute shell energy between rc_sel and
            # uc is large (order -3 eps per particle at rho* = 0.8), but this is a
            # DIFFERENCE between two groups of the same size in statistically
            # identical surroundings, so the mean cancels and only the fluctuation
            # reaches the acceptance. That is the whole reason the scheme is
            # affordable, and it is why the variance is accumulated and reported
            # below rather than left to be inferred from the acceptance ratio.
            if aux_on:
                u_ins_s, u_ins_f = _lj_set_energy_split(
                    chosen_coords, box, rc_sel, uc, rest_neighbors=cl
                )
                u_rem_s, u_rem_f = _lj_set_energy_split(
                    disk_in, box, rc_sel, uc, rest_neighbors=cl
                )
                delta_corr = beta * ((u_ins_f - u_ins_s) - (u_rem_f - u_rem_s))
                _corr_sum += delta_corr
                _corr_sq += delta_corr * delta_corr
                _corr_count += 1
                # aux_on implies ljbranch implies lj, so the tail term exists
                delta_E_lj = (delta_E_lj or 0.0) + delta_corr

            # The depth >= 1 acceptance is the original's, complete as written;
            # the depth-0 mirror scheme likewise needs no extra terms.
            corr = 0.0

            # acceptance
            if choice == "repack":
                # NULL REPACK.
                #
                # The pool the DP samples from contains the members that were
                # just tombstoned, so "put every one of them back exactly where
                # it was" is a legal clique and, at high density, often the
                # highest-weight one. That move is accepted (its acceptance
                # ratio is identically 1: same positions, same energy, same
                # Rosenbluth weight) and changes nothing. Counting it inflates
                # repack_accept and makes the sampler look like it is working.
                #
                # Not counting a self-transition does not touch the Markov
                # chain: X -> X leaves the state at X whether or not the
                # bookkeeping notices. What changes is the meaning of a cycle,
                # which now costs more real work and reports an honest
                # acceptance ratio.
                if (
                    _SKIP_NULL
                    and null_budget > 0
                    and _same_positions(chosen_coords, disk_in)
                ):
                    _d("repack_null")
                    c["repack_step"] -= 1
                    step -= 1
                    null_budget -= 1
                    continue
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
                    u_ins = (
                        _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl)
                        if len(chosen_coords)
                        else 0.0
                    )
                    u_rem = (
                        _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl)
                        if len(disk_in)
                        else 0.0
                    )
                    delta_E = (
                        (delta_E_lj or 0.0)
                        + (e_u_r_new - e_u_r_current)
                        + beta * (u_ins - u_rem)
                    )
                else:
                    delta_E = (
                        delta_E_lj
                        if delta_E_lj is not None
                        else (0.0 if lj else (e_u_r_new - e_u_r_current))
                    )
                # neg_log_acc is minus the log acceptance ratio, clamped at 0 so the
                # acceptance probability exp(-neg_log_acc) never exceeds 1. The move is
                # accepted when neg_log_acc <= -log(u), u ~ U(0,1).
                neg_log_acc = mu_excess + delta_E - corr
                _d("reach_metropolis_repack")
                _dv("repack_rosen", mu_excess)
                _dv("repack_dE", delta_E)
                _dv("repack_neg_log_acc", neg_log_acc)
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["repack_accept"] += 1
                    _d("accept_repack")
            elif choice == "add":
                c["add_attempt"] += 1
                mu_ideal = -np.log(total_vol / (N + 1))
                if branch:
                    k_trials = -np.log((n_trials - n_remove) / n_add)
                else:
                    k_trials = -np.log(n_trials - n_remove)
                mu_excess = -sum_new + sum_current - k_trials
                if ljaux:
                    u_ins = (
                        _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl)
                        if len(chosen_coords)
                        else 0.0
                    )
                    u_rem = (
                        _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl)
                        if len(disk_in)
                        else 0.0
                    )
                    delta_E = (
                        (delta_E_lj or 0.0)
                        + (e_u_r_new - e_u_r_current)
                        + beta * (u_ins - u_rem)
                    )
                else:
                    delta_E = (
                        delta_E_lj
                        if delta_E_lj is not None
                        else (0.0 if lj else (e_u_r_new - e_u_r_current))
                    )
                neg_log_acc = mu_ideal - beta * mu + mu_excess + delta_E - corr
                _d("reach_metropolis_add")
                _dv("add_rosen", mu_excess)
                _dv("add_dE", delta_E)
                _dv("add_ideal", mu_ideal - beta * mu)
                _dv("add_neg_log_acc", neg_log_acc)
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["add_accept"] += 1
                    _d("accept_add")
            else:
                mu_ideal = -np.log(N / total_vol)
                if branch:
                    k_trials = -np.log((n_trials - n_add) / n_remove)
                else:
                    k_trials = -np.log(n_trials - n_add)
                mu_excess = -sum_new + sum_current + k_trials
                if ljaux:
                    u_ins = (
                        _lj_set_energy(chosen_coords, None, box, uc, rest_neighbors=cl)
                        if len(chosen_coords)
                        else 0.0
                    )
                    u_rem = (
                        _lj_set_energy(disk_in, None, box, uc, rest_neighbors=cl)
                        if len(disk_in)
                        else 0.0
                    )
                    delta_E = (
                        (delta_E_lj or 0.0)
                        + (e_u_r_new - e_u_r_current)
                        + beta * (u_ins - u_rem)
                    )
                else:
                    delta_E = (
                        delta_E_lj
                        if delta_E_lj is not None
                        else (0.0 if lj else (e_u_r_new - e_u_r_current))
                    )
                neg_log_acc = mu_ideal + beta * mu + mu_excess + delta_E - corr
                _d("reach_metropolis_rem")
                _dv("rem_rosen", mu_excess)
                _dv("rem_dE", delta_E)
                _dv("rem_ideal", mu_ideal + beta * mu)
                _dv("rem_neg_log_acc", neg_log_acc)
                neg_log_acc = max(0.0, neg_log_acc)
                if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                    commit_exchange(chosen_coords, chosen_slots)
                    committed = True
                    c["remove_accept"] += 1
                    _d("accept_remove")
        finally:
            if not committed:
                for _s in removed_slots:
                    cl.reactivate(int(_s))

    # The window's only materialisation of the configuration. Order follows
    # `alive`, which is not the order the old array had; nothing downstream
    # depends on particle order.
    if _SKIP_NULL and null_budget == 0:
        # every uncounted null was spent: the rod is jammed enough that the
        # incoming packing is the only one the DP can find. The remaining nulls
        # in this window were counted as ordinary repacks.
        _d("repack_null_capped")
    state.positions = np.array(cl.coords[alive[:N]], dtype=float)
    state.step += cfg.sampling_window
    packing = (N * cfg.volume_of_molecule) / total_vol
    n_cav_mean = (_ncav_sum / _ncav_count) if _ncav_count else 0.0
    out = {"N": N, "packing_fraction": packing, "counters": c, "n_cav": n_cav_mean}
    # raw per-window sums; the runner logs them as integers (see _clq above)
    out["clique"] = dict(_clq)
    if _corr_count:
        mean = _corr_sum / _corr_count
        # population variance of the correction over the window's exchanges;
        # clamped at 0 because the sum-of-squares form can go slightly negative
        # by rounding when the true variance is near zero
        var = max(0.0, _corr_sq / _corr_count - mean * mean)
        out["aux_corr_mean"] = mean
        out["aux_corr_sd"] = float(np.sqrt(var))
        out["aux_corr_n"] = _corr_count
    return out
