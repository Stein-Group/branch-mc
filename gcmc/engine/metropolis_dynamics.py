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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._types import RNG, BoolArray, FloatArray, IntArray

if TYPE_CHECKING:
    from ..forcefield.base import Potential
    from ..proposals.trials import TrialSet
    from ..run.sim_setup import Config
    from ..sampling.channel import ChannelSweepHardCore, ChannelSweepLJ
    from ..system.state import ChainState

from scipy.spatial import cKDTree

from .. import config_specs
from ..config_specs import EnsembleSpec, KernelSpec, PotentialSpec
from ..forcefield import overlap
from ..forcefield.lennard_jones import lj_pair_energy
from ..forcefield.lj_table import LennardJonesTable
from ..forcefield.pair_table import table_pair_energy_vec
from ..proposals import channel_move, moves
from ..proposals.trials import TrialSet
from ..sampling import branch as branch_mod
from ..sampling import reweighting
from ..sampling.base import EnergyWeigher, ReweightWeigher, UniformWeigher
from ..sampling.branch import count_and_sample
from ..sampling.channel import (
    ChannelSweepHardCore,
    ChannelSweepLJ,
)
from ..sampling.rosenbluth import select_forward, select_reverse
from ..space import geometry, neighbors

#: log-acceptance of a proposal that failed geometrically. Named rather than
#: written as `-math.inf` at four call sites so the collection matrix and the
#: acceptance test cannot come to disagree about what "impossible" is.
_NINF_ACC = -math.inf


# ---------------------------------------------------------------------------
# VALIDATION RECORD for the short-range (dual-cutoff) auxiliary.
#
# A release gate lived here and refused aux_cutoff < upper_cutoff. It is gone
# because the evidence it was waiting for is in. What that evidence is, so the
# claim can be audited rather than taken on trust:
#
# EXACT BY CONSTRUCTION (no sampling involved)
#   * bit-identical to the no-auxiliary engine at aux_cutoff == upper_cutoff, over
#     2D/3D x fixed/full-span rod x depth-2/full_repacking x 3 seeds
#   * clique weight sums == brute-force enumeration over all C(n,k) subsets at the
#     SHORT radius, in 2D and 3D, with non-uniform member weights, including
#     minimum-image pairs across the periodic boundary
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


def _overlaps_any(
    dists: FloatArray,
    slots: IntArray,
    lower: float,
    trial_type: int,
    typed: LennardJonesTable | None,
    container: Any,
) -> bool:
    """Does any neighbour sit inside the overlap floor for its pair?

    Untyped, this is `overlap.overlaps(dists, lower).any()` -- the same call in
    the same place, so the scalar path is unchanged. Typed, the floor is
    `core_floor[trial_type, neighbour_type]` per neighbour, with the
    neighbour types read from the container by SLOT id.

    A single helper for both because the alternative is the same two-line
    conditional at every overlap site, and a hard floor that disagrees between
    the translate move and the exchange move is a detailed-balance defect that
    presents as a density offset.
    """
    if typed is None:
        return bool(overlap.overlaps(dists, lower).any())
    nt = container.species_at(slots)
    return bool(
        overlap.pair_overlaps(dists, trial_type, nt, typed.table.core_floor).any()
    )


def make_weigher(kern: KernelSpec, potential: Potential | None) -> Any:
    """Pick the selection-weight policy for the configured potential: uniform
    for hard cores (all admissible trials equal), Boltzmann/bias-weighted for
    every soft or auxiliary mode, psi6 for 2D hexatic reweighting. This is the
    only place potential_type maps to a weigher, so a new mode wires in here.

    Takes a `KernelSpec` rather than a `Config`: it reads `potential_type`
    and `two_dim` and nothing else, and both are derived fields. Narrowing it
    matters beyond tidiness -- `cell_parallel._run_cell` calls this inside the worker,
    so a `Config` parameter here is a `Config` shipped to every process.
    """
    match kern.potential_type:
        case "hard_sphere":
            return UniformWeigher()
        case "lj" | "aux" | "ljaux" | "ljbranch":
            assert potential is not None  # only reweight runs without one
            return EnergyWeigher(potential)
        case "reweight":
            if not kern.two_dim:
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
    raw: geometry.RawNeighbors,
    kdtree: Any,
    kern: KernelSpec,
    pot_spec: PotentialSpec,
    ens: EnsembleSpec,
    potential: Potential | None,
    choice: str,
    n_full: int,
    n_fluid: int | None = None,
) -> tuple[TrialSet, BoolArray | None]:
    """Turn the raw neighbour result into a weighted TrialSet, matching the
    original find_neighbors_fast weighting for the active mode.

    The species column rides through untouched: `geometry` filtered it by the
    same survivor mask as the coordinates, and this function only weights. It is
    put on the `TrialSet` here because that is where the cloud becomes a struct,
    and from there it survives the pool merge, the permutation and the Rosenbluth
    deletes by row alignment.

    Three sub-configs in place of `Config`, because the four values this
    reads genuinely come from three groups: `potential_type` and `two_dim` are
    derived, `gamma_dict` is a potential knob and `box` is an ensemble one.
    Listing them is the point -- eleven scattered `cfg.` reads left the real
    dependency to be inferred, and it turned out to be four fields out of 72.
    """
    coords, dists, idxs, km, spc = raw
    g1 = pot_spec.gamma_dict[1]
    match kern.potential_type:
        case "hard_sphere":
            return TrialSet(coords, np.ones(len(coords)), species=spc), km
        case "ljbranch":
            assert potential is not None
            # Reuse: each trial's single-particle LJ weight against the fluid
            # rest (any frozen matrix excludes by overlap only) is the
            # member-vs-system factor every clique needs downstream. Computed
            # once here from the neighbour distances already in hand, never
            # recomputed per clique.
            if (
                spc is not None
                and kdtree is not None
                and isinstance(potential, LennardJonesTable)
            ):
                # TYPED. The member-vs-system factor is per (trial type,
                # neighbour type), exactly as in the `lj` case. The sweep then
                # carries this as its one-body weight and adds only the
                # intra-clique pairs itself, so the two halves of a clique's
                # weight use the same table.
                nt = kdtree.species_at(idxs)
                if n_fluid is not None:
                    # QA: frozen obstacles exclude by overlap only and must not
                    # enter the energy sum. Same rule as the untyped path, which
                    # sets their distance to inf.
                    d_eff = np.where(
                        _is_matrix_neighbor(kdtree, idxs, n_fluid), np.inf, dists
                    )
                else:
                    d_eff = dists
                w = potential.weight_cloud(d_eff, spc, nt)
            else:
                w = _lj_singleparticle_weight(
                    dists, idxs, potential, n_fluid, container=kdtree
                )
            return TrialSet(coords, w, species=spc), km
        case "ljaux":
            assert potential is not None
            w, ref = potential.weight_bulk(
                dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=g1
            )
            return TrialSet(coords, w, ref_neighbor_coord=ref, species=spc), km
        case "lj":
            assert potential is not None
            # pair Boltzmann factors only. The long-range tail is a move-level
            # quantity applied once in the acceptance (delta_E_lj); folding it
            # into the per-trial weights put it inside W_old on deletions,
            # where the denominator position inverted its sign and
            # over-accepted deletions by exp(2*beta*|dU_tail|).
            if (
                spc is not None
                and kdtree is not None
                and isinstance(potential, LennardJonesTable)
            ):
                # TYPED: each trial carries its own drawn species and each
                # neighbour its own, so the pair energy is read from the table per
                # (trial type, neighbour type). Neighbour types come from the
                # container's species column indexed by the SLOT ids in `idxs` --
                # the same route `_is_matrix_neighbor` takes for its own flag, and
                # never from a coordinate.
                # `species_at`, not `species[idxs]`: the query pads rows with the
                # pad index, which is one past the slot table. It clamps, and the
                # clamp is neutral because a pad carries an infinite distance.
                nt = kdtree.species_at(idxs)
                w = potential.weight_cloud(dists, spc, nt)
            else:
                # `kdtree is None` is the EMPTY ENVIRONMENT (see the same guard in
                # `reweighting.calculate_order_parameter` and `HardCoreAux3D.weight_bulk`):
                # fully padded neighbour rows, so every pair energy is zero and
                # every weight is 1 whatever the types are. The table path is
                # skipped because there are no neighbour types to read, not
                # because the answer would differ.
                w = cast(FloatArray, potential.weight_bulk(dists))
            return TrialSet(coords, w, species=spc), km
        case "aux":
            assert potential is not None
            if kern.two_dim:
                # 2D is the analytic hard-disk g(r)
                w2d = cast(FloatArray, potential.weight_bulk(dists, gamma=g1))
                return TrialSet(coords, w2d, species=spc), km
            w, ref = potential.weight_bulk(
                dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=g1
            )
            return TrialSet(coords, w, ref_neighbor_coord=ref, species=spc), km
        case "reweight":
            # 2D hexatic: unit bulk weight, carry the psi6 phasor accumulators
            cos_sum, sin_sum, counts = reweighting.calculate_order_parameter(
                coords, dists, idxs, kdtree, ens.box, g1
            )
            return TrialSet(
                coords,
                np.ones(len(coords)),
                cossum=cos_sum,
                sinsum=sin_sum,
                counts=counts,
                species=spc,
            ), km
        case other:
            raise NotImplementedError(other)


@dataclass(frozen=True)
class MetropolisModel:
    """Everything a single-particle Metropolis proposal has to ask about a point.

    THIS EXISTS BECAUSE THERE WERE TWO LOOPS. `_advance_single_particle` and the
    framework loop that used to live in `mof_metropolis` said, in their own
    docstrings, that they mirrored each other "move for move", and they did:
    the same three-way move choice, the same textbook grand-canonical
    acceptance, the same null-proposal handling, written twice about 500 lines
    apart. Two copies of an acceptance criterion is how a fix lands in one of
    them, and the whole reason the swap move was executed as a deletion for a
    stage (`moves.draw_move` emitted a string one copy handled and the other
    fell through on) is that a move set had two homes.

    What actually differed between them was never the algebra. It was three
    questions about a point, and this object answers them:

        blocked(point, s)     is this position forbidden before any energy is
                              computed? (a pore wall; a framework atom)
        pair_energy(...)      the guest-guest energy and whether it overlaps
        one_body(point, s)    the static host's energy, zero without one

    plus a state point (`log_z`) that is per species, and reduces to `beta*mu`
    when there is one. Everything else -- the tail, the hard core, the species
    column -- is a field here rather than a fork in the loop.

    ONE ACCEPTANCE, IN ONE TERM ORDER. The two loops summed the same acceptance
    in different orders (`log V - log(N+1) + beta mu + log_bf` against
    `log z + log V - log(n+1) - beta u`), which is the same number only up to
    the last bits. The bulk order is kept verbatim, because the bulk chains are
    the ones with 52 pinned counter digests behind them and the generalised form
    is textually identical to the old one when there is one species; the
    framework chain was re-measured against its golden pin instead of argued
    about.

    Attribute lookups in a hot loop are a real cost in this codebase (see
    `_place_logfactor`), but this loop is single-trial: one energy call per
    proposal dominates by orders of magnitude, and the sweep samplers -- where
    that note was measured -- do not come through here.
    """

    beta: float
    box: FloatArray
    volume: float
    upper_cutoff: float
    lower_cutoff: float
    n_neigh: int
    dim: int
    move_factor: float
    #: log activity BY SPECIES ID. At one species this is `{0: beta * mu}` and
    #: every per-species expression below collapses to the single-species one it
    #: was before P4. Keyed rather than positional because a framework's guest
    #: type ids are whatever the force-field file numbered them.
    log_z: dict[int, float]
    #: fixed insertion probability per species; a property of the MOVE SET and
    #: never of the current occupancy, which is what detailed balance needs.
    species_probs: tuple[float, ...]
    species_ids: tuple[int, ...]
    hard_core: bool = False
    #: bulk LJ mean-field tail prefactor. None wherever the integral does not
    #: describe the system: a pore, a frozen matrix, a framework, or a mixture.
    tail_pref: float | None = None
    #: the static host, if there is one. Its presence is what makes an empty
    #: environment a real energy difference rather than a certainty.
    field: Any | None = None
    table: Any | None = None
    pots: Any | None = None
    #: pore admissibility, bulk only: a callable answering "is this point inside
    #: the accessible region?"
    inside_pore: Any | None = None
    #: Does this loop propose single-particle insertions and deletions? True is
    #: every path that existed before the window-exchange benchmark arm, and
    #: takes `moves.draw_move` exactly as before. False hands N-changing work to
    #: another move (the framework driver's window exchange) and draws from
    #: `moves.draw_fixed_n_move`: translation, plus identity swap for a mixture.
    exchange: bool = True
    #: (translation, exchange, identity swap) probabilities of one proposal, as
    #: a framework params file states them (`engine.move_probabilities`). None
    #: is every pre-existing path and draws the module defaults unchanged.
    move_probs: tuple[float, float, float] | None = None
    #: Count every proposal toward `window`, translations included. False keeps
    #: the historical rule (an exchange=True window counts exchange and swap
    #: attempts only), which every pinned chain depends on.
    count_every_move: bool = False

    @property
    def n_species(self) -> int:
        """How many species the move set draws from."""
        return len(self.log_z)

    def blocked(self, point: FloatArray, s: int) -> bool:
        """Is this position forbidden before any energy is computed?

        A pore wall and a framework atom are the same kind of answer -- the
        proposal never had a chance -- and they are mutually exclusive in
        practice, so one predicate covers both.
        """
        if self.inside_pore is not None and not self.inside_pore(point):
            return True
        return bool(self.field is not None and self.field.overlaps(point, s))

    def pair_energy(
        self,
        point: FloatArray,
        s: int,
        others: FloatArray,
        others_s: IntArray | None,
        tree: Any,
    ) -> tuple[float, bool, int]:
        """(guest-guest energy, does it overlap, pairs evaluated).

        The neighbour query SATURATES on both paths now. It always did on the
        bulk one; the framework one used a fixed `k = n_neigh` and could
        therefore have dropped in-range pairs silently at high loading. Changing
        it is free rather than a re-baseline, and that is measured, not assumed:
        over 126829 calls across the framework suite the two forms returned the
        same energy and the same overlap verdict every time, to the bit, because
        the query never actually saturated (17 neighbours found against a cap of
        30). The guarantee is now structural instead of a property of the
        densities that happened to be tested.
        """
        if len(others) == 0:
            return 0.0, False, 0
        d, idx = geometry.query_saturating(
            tree, [point], self.n_neigh, self.upper_cutoff
        )
        d, idx = d[0], idx[0]
        finite = np.isfinite(d)
        d, idx = d[finite], idx[finite]
        if d.size == 0:
            return 0.0, False, 0
        if self.table is None:
            # one world, sigma = epsilon = 1: a scalar floor and the frozen
            # pair energy, exactly as `_advance_single_particle` always did
            if overlap.overlaps(d, self.lower_cutoff).any():
                return 0.0, True, int(d.size)
            if self.hard_core:
                return 0.0, False, int(d.size)
            return _lj_energy(d, self.upper_cutoff), False, int(d.size)
        neigh = np.asarray(others_s, int)[idx]
        if overlap.pair_overlaps(d, s, neigh, self.table.core_floor).any():
            return 0.0, True, int(d.size)
        assert self.pots is not None
        return (
            self.pots[s].total_energy(d, neighbour_types=neigh),
            False,
            int(d.size),
        )

    def one_body(self, point: FloatArray, s: int) -> float:
        """The static host's energy at this point. Zero without a host, and the
        `is None` test is what makes an empty environment a CERTAIN acceptance
        in bulk and an open question in a framework."""
        if self.field is None:
            return 0.0
        return float(self.field.energy(point, s))


def metropolis_window(
    pos: FloatArray,
    spc: IntArray | None,
    rng: RNG,
    model: MetropolisModel,
    window: int,
    counters: dict[str, Any],
    energy: dict[str, float] | None = None,
) -> tuple[FloatArray, IntArray | None, int]:
    """One sampling window of single-particle Metropolis GCMC.

    Bulk, pore, quenched-annealed matrix and rigid framework all come through
    here, single species or a mixture; `model` answers what differs between
    them. Returns `(positions, species, pairs_evaluated)`. `counters` is
    mutated in place (keys from `state.window_counters`), and so is `energy`
    when given: `guest` and `host` accumulate the energy change of every
    accepted move, guest-guest (tail included) and host-guest respectively.
    `spc` None means one species, id `model.species_ids[0]`.

    The RNG order is fixed: a certain acceptance (hard core without overlap, or
    an empty environment without a host) draws no uniform, and the species draw
    precedes the position draw and consumes nothing at one species. Changing
    either changes every chain through this loop.
    """
    box = model.box
    side = box
    dim = model.dim
    beta = model.beta
    mf = model.move_factor
    log_v = math.log(model.volume)
    n_species = model.n_species
    ids = np.array(model.species_ids, dtype=np.int64)
    probs = np.array(model.species_probs, dtype=float)
    log_z = model.log_z
    n_evals = 0
    track = energy is not None and not model.hard_core
    du_guest = 0.0
    du_host = 0.0

    def pick_species() -> int:
        """Insertion/deletion species draw; consumes no randomness at one species."""
        if n_species == 1:
            return int(ids[0])
        return int(rng.choice(ids, p=probs))

    def species_of(j: int) -> int:
        return int(ids[0]) if spc is None else int(spc[j])

    step = 0
    while step < window:
        N = len(pos)
        if model.exchange:
            choice = moves.draw_move(rng, n_species, model.move_probs)
        else:
            choice = moves.draw_fixed_n_move(rng, n_species, model.move_probs)

        if choice == "move":
            counters["translate_tried"] += 1
            if not model.exchange or model.count_every_move:
                # without exchange a window of translations alone must end, so
                # every proposal counts
                step += 1
            if N == 0:
                continue
            j = int(rng.integers(N))
            s = species_of(j)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            without_s = None if spc is None else np.delete(spc, j)
            disp = mf * (rng.uniform(0, 1, dim) - 0.5)
            moved = (old + disp) % side
            if model.blocked(moved, s):
                continue
            if len(without) == 0 and model.field is None:
                # nothing to interact with: dU = 0, accepted without a draw
                pos = moved.reshape(1, -1)
                counters["translate_accepted"] += 1
                counters["translate_sum_dr2"] += float(disp @ disp)
                continue
            tree = cKDTree(without, boxsize=box) if len(without) else None
            p_new, ov_new, ne = model.pair_energy(moved, s, without, without_s, tree)
            n_evals += ne
            if ov_new:
                continue
            if model.hard_core:
                pos = np.append(without, [moved], axis=0)
                if spc is not None:
                    assert without_s is not None
                    spc = np.append(without_s, s)
                counters["translate_accepted"] += 1
                counters["translate_sum_dr2"] += float(disp @ disp)
                continue
            h_new = model.one_body(moved, s)
            p_old, _ov, ne = model.pair_energy(old, s, without, without_s, tree)
            n_evals += ne
            h_old = model.one_body(old, s)
            u_new = p_new + h_new
            u_old = p_old + h_old
            if np.log(rng.uniform(0, 1)) <= -beta * (u_new - u_old):
                pos = np.append(without, [moved], axis=0)
                if spc is not None:
                    assert without_s is not None
                    spc = np.append(without_s, s)
                counters["translate_accepted"] += 1
                counters["translate_sum_dr2"] += float(disp @ disp)
                if track:
                    du_guest += p_new - p_old
                    du_host += h_new - h_old
            continue

        if choice == "swap":
            counters["swap_tried"] += 1
            step += 1
            if N == 0:
                continue
            j = int(rng.integers(N))
            s = species_of(j)
            s_new = moves.draw_swap_target(rng, s, list(model.species_ids))
            point = pos[j]
            without = np.delete(pos, j, axis=0)
            without_s = None if spc is None else np.delete(spc, j)
            if model.blocked(point, s_new):
                continue
            tree = cKDTree(without, boxsize=box) if len(without) else None
            p_new, ov, ne = model.pair_energy(point, s_new, without, without_s, tree)
            n_evals += ne
            if ov:
                continue
            h_new = model.one_body(point, s_new)
            p_old, _ov, ne = model.pair_energy(point, s, without, without_s, tree)
            n_evals += ne
            h_old = model.one_body(point, s)
            u_new = p_new + h_new
            u_old = p_old + h_old
            # a labelled particle changes identity: activity ratio times the
            # Boltzmann factor, no n_s/(n_s'+1) term
            log_acc = log_z[s_new] - log_z[s] - beta * (u_new - u_old)
            if math.log(rng.uniform(0, 1)) <= log_acc:
                assert spc is not None
                spc[j] = s_new
                counters["swap_accepted"] += 1
                if track:
                    du_guest += p_new - p_old
                    du_host += h_new - h_old
            continue

        if choice == "add":
            counters["insert_tried"] += 1
            step += 1
            s = pick_species()
            trial = rng.uniform(0, side, dim)
            if model.blocked(trial, s):
                continue
            tree = cKDTree(pos, boxsize=box) if N > 0 else None
            p_u, ov, ne = model.pair_energy(trial, s, pos, spc, tree)
            n_evals += ne
            if ov:
                continue
            n_s = N if spc is None else int(np.count_nonzero(spc == s))
            if model.hard_core:
                log_bf = 0.0
            else:
                h_u = model.one_body(trial, s)
                u = p_u + h_u
                t_u = 0.0
                if model.tail_pref is not None:
                    t_u = model.tail_pref * (2.0 * N + 1.0)
                    u += t_u
                log_bf = -beta * u
            log_acc = log_v - math.log(n_s + 1) + log_z[s] + log_bf
            if math.log(rng.uniform(0, 1)) <= log_acc:
                pos = np.append(pos, [trial], axis=0) if N > 0 else trial.reshape(1, -1)
                if spc is not None:
                    spc = np.append(spc, s)
                counters["insert_accepted"] += 1
                if track:
                    du_guest += p_u + t_u
                    du_host += h_u
            continue

        if choice == "remove":
            counters["delete_tried"] += 1
            step += 1
            # the species draw comes before the emptiness test
            s = pick_species()
            if spc is not None:
                idx_s = np.nonzero(spc == s)[0]
                n_s = len(idx_s)
                if n_s == 0:
                    continue
                j = int(idx_s[int(rng.integers(n_s))])
            else:
                if N == 0:
                    continue
                n_s = N
                j = int(rng.integers(N))
            chosen = pos[j]
            without = np.delete(pos, j, axis=0)
            without_s = None if spc is None else np.delete(spc, j)
            if model.hard_core:
                log_bf = 0.0
            else:
                tree = cKDTree(without, boxsize=box) if len(without) else None
                p_u, _ov, ne = model.pair_energy(chosen, s, without, without_s, tree)
                n_evals += ne
                h_u = model.one_body(chosen, s)
                u = p_u + h_u
                t_u = 0.0
                if model.tail_pref is not None:
                    t_u = model.tail_pref * (2.0 * N - 1.0)
                    u += t_u
                log_bf = -beta * u
            log_acc = math.log(n_s) - log_v - log_z[s] - log_bf
            if math.log(rng.uniform(0, 1)) <= log_acc:
                pos = without
                if spc is not None:
                    assert without_s is not None
                    spc = without_s
                counters["delete_accepted"] += 1
                if track:
                    du_guest -= p_u + t_u
                    du_host -= h_u
            continue

        raise RuntimeError(f"unhandled move choice {choice!r}")

    if energy is not None:
        energy["guest"] = energy.get("guest", 0.0) + du_guest
        energy["host"] = energy.get("host", 0.0) + du_host
    return pos, spc, n_evals


def _advance_single_particle(
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

    This is now a `Config` ADAPTER over `metropolis_window`, which is the one
    loop. It projects the five sub-configs it needs into a
    `MetropolisModel` and reads the result back onto `state`; the move set, the
    acceptance and the RNG order all live in one place shared with the framework
    path, which used to be a second copy of them.
    """
    # The sub-configs this loop reads. Projected once; `cfg` is not read
    # again below. Five of the six -- a single-trial grand-canonical loop needs
    # the state point, the proposal knobs, the potential, the precomputed forms
    # and the window length, and nothing about where the run writes or how many
    # cores it has, so `ExecutionSpec` is the one that never appears.
    kern = config_specs.kernel_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    pot_spec = config_specs.potential_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    io_ = config_specs.io_spec(cfg)

    # The hard-core pmf auxiliary (`aux`) is refused by `sim_setup.from_params`
    # (M15: fail at the earliest layer that can see the problem), so it cannot
    # reach here from a params file. Kept as an assertion rather than deleted:
    # `_advance_single_particle` is also called directly by characterisation scripts
    # that build a Config by other means, and a silent wrong answer is worse
    # than a loud one.
    assert kern.potential_type != "aux", (
        "metropolis with a hard-core pmf auxiliary reached the engine; "
        "sim_setup.from_params refuses this combination, so this Config did "
        "not come from a validated params file"
    )
    if kern.potential_type == "ljaux":
        # with a single trial the selection bias cancels identically, so
        # metropolis + pmf IS metropolis on the plain LJ potential
        from ..forcefield.auxiliary import LJAux3D

        assert isinstance(potential, LJAux3D)
        potential = potential.lj

    hs = kern.potential_type == "hard_sphere"
    mu = state.mu if state.mu is not None else ens.chem_pot
    model = MetropolisModel(
        beta=kern.beta,
        # per-axis side: for a cubic box this is [L, L, L], so `% side`
        # and rng.uniform(0, side, dim) are bit-identical to the old scalar side
        # (numpy broadcasts the standard-uniform draws the same way); for an
        # orthorhombic box it is the correct per-axis wrap and insertion range.
        box=np.asarray(kern.box_dims, dtype=np.float64),
        volume=kern.total_volume,
        upper_cutoff=kern.upper_cutoff,
        lower_cutoff=pot_spec.lower_cutoff,
        n_neigh=kern.no_neigh,
        dim=ens.dimension,
        move_factor=samp.move_factor,
        # ONE species here, always: `sim_setup` refuses typed + metropolis
        # outside the canonical ensemble, which does not reach this function.
        log_z={0: kern.beta * mu},
        species_probs=(1.0,),
        species_ids=(0,),
        hard_core=hs,
        tail_pref=None if hs else pot_spec.lj_tail_pref,
        inside_pore=(
            None
            if ens.pore_geometry is None
            else (lambda pt: geometry.inside_pore(pt, ens, samp))
        ),
    )
    if not hs:
        assert model.tail_pref is not None  # set for every LJ config

    c: dict[str, Any] = {
        "translate_tried": 0,
        "translate_accepted": 0,
        "translate_sum_dr2": 0.0,
        "insert_tried": 0,
        "insert_accepted": 0,
        "delete_tried": 0,
        "delete_accepted": 0,
    }
    energy: dict[str, float] = {}
    pos, _spc, _n_evals = metropolis_window(
        np.asarray(state.positions, float),
        None,
        state.rng,
        model,
        io_.sampling_window,
        c,
        energy,
    )
    state.positions = pos
    # single species only: sim_setup refuses typed + metropolis outside NVT
    assert state.species is None or len(state.species) == len(pos)
    state.step += io_.sampling_window
    return {
        "N": len(pos),
        "columns": c,
        "dU": None if hs else energy["guest"],
    }


def _lj_energy(dists: FloatArray, rc: float) -> float:
    """Total truncated-LJ energy of one particle against a distance list:
    sum of u(r) over 0 < r < rc; zero-distance entries (self-pairs from a
    query that saw the particle itself) and beyond-cutoff entries drop out."""
    d = np.asarray(dists, float)
    d = d[(d < rc) & (d > 0)]
    if d.size == 0:
        return 0.0
    return float(np.sum(lj_pair_energy(d)))


def _lj_energy_typed(
    dists: FloatArray,
    si: IntArray,
    sj: IntArray,
    rc: float,
    table: Any,
) -> float:
    """`_lj_energy`'s per-pair sibling: sum of u_ij(r) over 0 < r < rc.

    Same truncation rule and same exclusions as the single-sigma form -- zero
    distances (a query that saw the particle itself) and beyond-cutoff entries
    drop out -- with `sigma_ij` and `eps_ij` gathered per pair from the table.
    `si` and `sj` are the two type ids of each distance, element-aligned with
    `dists`.

    `table_pair_energy_vec` rather than a loop over type combinations: this is
    called twice per exchange by the dual-cutoff correction, and the grouped form
    cost 53 s on one feature-matrix point when it sat inside clique enumeration.
    """
    d = np.asarray(dists, float)
    keep = (d < rc) & (d > 0)
    if not keep.any():
        return 0.0
    return float(
        np.sum(
            table_pair_energy_vec(
                d[keep], np.asarray(si)[keep], np.asarray(sj)[keep], table
            )
        )
    )


def _ljbranch_forward(
    pool: TrialSet,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
    rng: RNG,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
    table: Any = None,
) -> tuple[FloatArray, float, list[int]] | tuple[None, None, None]:
    """Forward (insertion) weight over all non-overlapping k-cliques of the
    pool, one drawn proportional to its Boltzmann weight. The member-vs-system
    factor of each clique is reused from pool.energy_weight (computed in the
    neighbour build); only the intra-clique pairs are added inside the sum.

    Returns (chosen_coords, log W_new, chosen_pool_rows) or (None, None, None)
    if no clique exists. The rows are what the typed insert path reads the
    inserted species off, and what M21's provenance is decided by."""
    # logW, count, chosen, rows = branch_mod.ljbranch_weight_sum(
    logW, count, chosen, rows = branch_mod.lj_clique_weight_sum(
        pool.coords,
        pool.energy_weight,
        k,
        lower,
        box,
        rc,
        beta,
        rng=rng,
        species=species,
        floor=floor,
        table=table,
    )
    if count == 0 or chosen is None or rows is None or not np.isfinite(logW):
        return None, None, None
    return chosen, logW, rows


def _ljbranch_reverse(
    pool: TrialSet,
    k: int,
    lower: float,
    box: float | FloatArray,
    rc: float,
    beta: float,
    species: IntArray | None = None,
    floor: FloatArray | None = None,
    table: Any = None,
) -> float:
    """Reverse (deletion) weight: the same non-overlapping k-clique sum on the
    current pool, where the removed group is guaranteed to be one member. Same
    enumerator and same weight definition as the forward, so detailed balance
    holds by construction. Returns log W_curr, or -inf if no k-clique exists."""
    logW, count, _, _ = branch_mod.ljbranch_weight_sum(
        pool.coords,
        pool.energy_weight,
        k,
        lower,
        box,
        rc,
        beta,
        rng=None,
        species=species,
        floor=floor,
        table=table,
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
    exactly `rest`; passing it saves an O(N) index build per clique. Frozen
    matrix slots are dropped: obstacles interact by overlap exclusion only, never
    through the LJ sum. Without it, a throwaway cell list is built here from
    `rest` (the path cell_parallel.py takes).
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
    return int(
        np.count_nonzero(np.all(a[:, None, :] == b[None, :, :], axis=2).any(axis=1))
    )


def _lj_set_energy_split(
    coords: FloatArray,
    box: FloatArray,
    rc_short: float,
    rc_full: float,
    rest_neighbors: Any = None,
    species: IntArray | None = None,
    table: Any = None,
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

    TYPED (`table` given, and `species` with it: the type id of each of `coords`).
    Both sums become per-pair, through `_lj_energy_typed`. Needed because the
    typed short-range auxiliary needs it: this function is the whole of
    the correction's arithmetic: run untyped on a mixture and the correction would
    be the sigma=1 shell difference of a system that has no sigma=1 pair in it --
    a wrong number in the acceptance rather than a missing speed-up. On the
    untyped path (`table is None`) every expression below is the one that was
    there, which the counter digests are the evidence for.
    """
    typed = table is not None
    if typed and species is None:
        raise ValueError(
            "a pair table was given without the species of `coords`; the two "
            "radii would be applied to the right pairs with the wrong types"
        )
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
            fluid = ~nb.is_matrix[sv]
            dv = dv[fluid]
            if typed:
                # which of `coords` each surviving hit belongs to. `query` returns
                # one row per point, so the row index IS the coord index; it has
                # to be carried through the same two masks as the distances or the
                # gathered types stop being element-aligned with them.
                rows = np.broadcast_to(np.arange(len(coords))[:, None], d.shape)
                own = np.asarray(species)[rows[live][fluid]]
                other = nb.species_at(sv[fluid])
                u_full += _lj_energy_typed(dv, own, other, rc_full, table)
                u_short += _lj_energy_typed(dv, own, other, rc_short, table)
            else:
                u_full += _lj_energy(dv, rc_full)
                u_short += _lj_energy(dv, rc_short)
    if len(coords) > 1:
        ii, jj = np.triu_indices(len(coords), k=1)
        dvec = coords[ii] - coords[jj]
        dvec -= np.round(dvec / box_arr) * box_arr
        dr = np.sqrt((dvec**2).sum(axis=1))
        if typed:
            spc = np.asarray(species)
            u_full += _lj_energy_typed(dr, spc[ii], spc[jj], rc_full, table)
            u_short += _lj_energy_typed(dr, spc[ii], spc[jj], rc_short, table)
        else:
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
    conserving moves. `_advance_single_particle` is the independent single-trial
    reference implementation and is deliberately NOT this function.

    Engineering: one persistent CellList answers every neighbour question for
    the window; particle identity lives in slot ids, the live set in the
    `alive` array with O(1) swap-remove, and `state.positions` is materialised
    exactly once, at the end. The proposal/rollback protocol (tombstone the
    would-be-removed slots, query, then commit or reactivate in a finally) is
    what keeps the cell list consistent on EVERY exit path, including the many
    early `continue`s.
    """
    # The sub-configs this window reads. Projected once, here; `cfg` is
    # not read again below. All six appear: the multi-trial engine is the widest
    # consumer of configuration in the package, and saying so with six named
    # objects beats saying it with one flat object of seventy-two fields.
    #
    # Five `getattr(cfg, name, default)` reads disappeared with the migration.
    # Those were defensive reads against a field that might not exist; on a typed
    # dataclass it always does, so the default was dead code that would have
    # masked a genuine rename rather than protecting against one.
    kern = config_specs.kernel_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    pot_spec = config_specs.potential_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    io_ = config_specs.io_spec(cfg)

    # Both fixed-N ensembles take the number-conserving loop below; they differ
    # in whether COMPOSITION is conserved with the total. `semigrand` lets the
    # repack redraw species; `NVT` with more than one species does not, and its
    # move set becomes translation + pair identity exchange instead.
    #
    # DERIVED HERE, ahead of the dispatch, because the dispatch needs it. The
    # sampler alone does not decide which loop runs: a canonical run has no trial
    # cloud, so `sampler.type` is 'metropolis' by validation, and `metropolis`
    # names `_advance_single_particle` -- which is single-species, has no branch for
    # the pair exchange, and raises `unhandled move choice 'swap'` on the first
    # one. That raise is how this was found. The canonical move set is the NVT
    # branch's, not the metropolis loop's, and the ensemble is what says so.
    nvt = ens.ensemble_type in ("NVT", "semigrand")
    canonical = ens.ensemble_type == "NVT" and kern.n_species > 1

    if kern.metropolis and not canonical:
        assert potential is not None  # metropolis never runs in reweight mode
        return _advance_single_particle(state, cfg, potential)
    if kern.potential_type == "reweight" and not kern.two_dim:
        raise NotImplementedError(
            "3D reweighting (Q6) is not a coherent estimator in the original "
            "and is not wired; reweighting is the 2D hexatic (psi6) method"
        )

    rng = state.rng
    pos = np.asarray(state.positions, float)

    box = np.asarray(kern.box_dims, dtype=np.float64)
    # per-axis side: for a cubic box this is [L, L, L], so every `% side`,
    # minimum-image and insertion draw below is bit-identical to the old scalar
    # `ens.box` (numpy broadcasts the standard-uniform draws the same way); for
    # an orthorhombic box it is the correct per-axis wrap and insertion range.
    side = box
    two_d = kern.two_dim
    dim = ens.dimension
    lower = (
        pot_spec.lower_cutoff
    )  # the single exclusion floor (diameter, or 0.85 for LJ)
    r_sq = kern.r_squared
    uc = kern.upper_cutoff
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
    # Bound to a local first: `getattr(cfg, ...) is None` does not narrow
    # `pot_spec.aux_cutoff` for the checker, and a bare cast would hide a real None
    # if the attribute ever stops being set.
    _aux_rc = pot_spec.aux_cutoff
    rc_sel = uc if _aux_rc is None else float(_aux_rc)
    nn = kern.no_neigh
    mf = samp.move_factor
    n_trials = samp.n_trial_molecules
    depth = samp.depth
    gdict = pot_spec.gamma_dict
    beta = kern.beta
    temp = pot_spec.temp
    mu = state.mu if state.mu is not None else ens.chem_pot
    # quenched-annealed frozen matrix: obstacles that enter the overlap test but
    # are never inserted/deleted/moved/counted. Empty => ordinary bulk run.
    matrix: FloatArray | None = ens.matrix_positions
    has_matrix = matrix is not None and len(matrix) > 0
    total_vol = kern.total_volume
    ljaux = kern.potential_type == "ljaux"
    ljbranch = kern.potential_type == "ljbranch"  # true LJ+branch, no auxiliary
    # The auxiliary is ACTIVE whenever one is configured, INCLUDING at the no-op
    # radius rc_sel == uc.
    #
    # `auxiliary.cutoff == run.upper_cutoff` -- and, in typed mode,
    # `aux: {control: true}` -- exist for exactly one purpose: to be the control
    # arm that decides whether the correction is inserted in the right place.
    # The gate states it as "the auxiliary control arm must be an EXACT no-op,
    # not statistically close. If it is not, the correction is inserted in the
    # wrong place." A short-circuit before the correction turns
    # that arm into a test that the short-circuit works: it never enters the
    # code whose insertion point is in question, so it cannot answer the
    # question, and it passes.
    #
    # So the correction RUNS at the no-op radius, where its value is exactly
    # 0.0 -- exactly, not nearly: `_lj_set_energy_split` applies both radii to one
    # neighbour query as two identical sums over the same array in the same order,
    # so at rc_short == rc_full the difference is a float zero. Adding an exact
    # zero in the right place changes nothing; adding one in the wrong place still
    # moves the acceptance, the RNG draw after it, or the None-ness of the tail.
    # `tests/engine/test_aux_noop_control.py` is that gate, and it asserts the
    # correction was evaluated as well as that the chain did not move.
    #
    # THE COST FALLS ONLY ON THE CONTROL ARM. Every production run configures
    # rc_sel < uc, where this flag is True anyway; the extra work happens only
    # for a run that deliberately asks for the no-op, which is a run whose whole
    # purpose is to do that work and get zero.
    aux_on = ljbranch and _aux_rc is not None
    lj = (
        kern.potential_type in ("lj", "ljbranch") or ljaux
    )  # LJ physics: floor, translation, tail
    # the hard-core weight is 1 for every admissible trial and reads no distance;
    # every other weigher reads the rows, so only this one may ask for less
    hs_needs_dists = kern.potential_type != "hard_sphere"
    branch = kern.branch
    channel = kern.channel  # branch variant: cuboid + sweep
    # Move split: 1 takes the frozen three-way call, so every single-species
    # characterisation trajectory is unaffected.
    n_species = kern.n_species
    # ---- the typed insert path --------------------------------
    #
    # `typed_insert` is the single switch. When it is False NOTHING below draws,
    # branches or costs anything: no species is sampled, no correction is
    # computed, and `commit_exchange` is called with the same two arguments it
    # always was. That is not an optimisation -- `rng` draws consume the stream,
    # so a single-species run must not make one, or every characterisation
    # trajectory in `tests/ref` and every counter digest moves. The digest gate
    # is what holds that claim.
    # THE TYPED ENERGY, or None. Non-None exactly when `build_potential` handed
    # back a table-aware potential, which it does only for a typed `lj` run --
    # never for a sweep, whose locked kernels are still single-sigma. Every
    # expression guarded by `is not None` below is otherwise the one that was
    # there, so an untyped run is unchanged bit for bit and the digest gate says
    # so.
    _typed_lj = potential if isinstance(potential, LennardJonesTable) else None
    typed_insert = n_species > 1 and kern.species is not None
    if typed_insert:
        assert kern.species is not None  # narrowed by typed_insert
        # Guests only. `p_trial` is (T,) over every entry with framework rows at
        # zero, and the guests sum to 1, so the guest block is the leading
        # run of positive entries and is already normalised.
        _pt = np.asarray(kern.species.p_trial, dtype=float)
        n_guest = int(np.count_nonzero(_pt > 0.0))
        p_trial_guest = _pt[:n_guest]
        log_p_trial = np.log(_pt)
        # `z` is the authoritative state point: whichever route the params
        # file took -- mu, fugacity or z -- it was reduced to an activity, and the
        # swap acceptance is a ratio of activities and nothing else.
        log_z = np.log(np.asarray(kern.species.z, dtype=float))
        guest_ids = list(range(n_guest))
    else:
        n_guest = 1
        p_trial_guest = None
        log_p_trial = None
        log_z = None
        guest_ids = []
    n_trials_base = n_trials  # full-span resets n_trials/move
    #: `channel_length = "full"`: the body's length is drawn per move from the
    #: orientation, so it is read here rather than off the config.
    channel_full = bool(samp.channel_full)
    #: The largest distance the SWEEP reads, which is what the full-span body's
    #: image guard has to hold its periodic copies beyond. `rc_sel` on the LJ
    #: path (the auxiliary radius when one is on, `upper_cutoff` otherwise) and
    #: the overlap floor on the hard-core one -- the same quantity `sim_setup`'s
    #: diagonal guard uses for the fixed cuboid.
    chan_reach = float(rc_sel if kern.potential_type != "hard_sphere" else lower)
    full_repacking = samp.full_repacking  # whole-cavity repack
    reweight = kern.potential_type == "reweight"
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
    # The cell width is the selection radius: the trial-cloud screen at rc_sel
    # is the hottest query, so the grid is sized for it.
    _grid = rc_sel
    # The species column travels with the coordinates into the cell list, never
    # by index: particles reorder inside the window, so a parallel array
    # held outside would be scrambled by the first swap-remove. A single-species
    # run has `state.species is None`, `species_of` materialises zeros, and the
    # list reads type 0 everywhere.
    cl = neighbors.CellList.from_positions(
        pos,
        side,
        dim,
        _grid,
        matrix=matrix if has_matrix else None,
        species=state.species_of,
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
        chosen_coords: FloatArray,
        chosen_slots: IntArray | list[int] | None,
        chosen_species: IntArray | list[int] | None = None,
    ) -> None:
        """Mirror an accepted exchange into the cell list and `alive`. The chosen
        slots were already tombstoned when the proposal was set up and are never
        reactivated, so it is safe for an insert to rebuild and reclaim them;
        that ordering is the cell list's slot-lifetime contract.

        `chosen_species` is the type of each inserted particle, positionally
        aligned with `chosen_coords`. None means every insertion is type 0, which
        is the single-species case and what every caller passes today.

        The species is written at INSERT time rather than patched afterwards
        because a reclaimed slot arrives carrying its previous tenant's type: a
        second pass over `alive` would be correct only until an insert triggered a
        rebuild partway through.
        """
        if chosen_slots is not None:
            alive_drop_slots(chosen_slots)
        if chosen_species is None:
            for p_ in chosen_coords:
                alive_push(cl.insert(p_))
        else:
            if len(chosen_species) != len(chosen_coords):
                raise ValueError(
                    f"{len(chosen_species)} species for "
                    f"{len(chosen_coords)} inserted coordinates"
                )
            for p_, s_ in zip(chosen_coords, chosen_species):
                alive_push(cl.insert(p_, species=int(s_)))

    # The window's log columns, raw counts and sums only, in log order.
    c: dict[str, Any] = {
        "translate_tried": 0,
        "translate_accepted": 0,
        "translate_sum_dr2": 0.0,
    }
    if canonical:
        c.update(swap_tried=0, swap_accepted=0)
    elif nvt:
        c.update(
            repack_tried=0,
            repack_evaluated=0,
            repack_accepted=0,
            repack_sum_size=0,
            repack_sum_new=0,
            repack_sum_size_accepted=0,
            repack_sum_new_accepted=0,
        )
    else:
        c.update(
            insert_tried=0,
            insert_evaluated=0,
            insert_accepted=0,
            delete_tried=0,
            delete_evaluated=0,
            delete_accepted=0,
        )
        if typed_insert:
            c.update(swap_tried=0, swap_accepted=0)
    if not canonical:
        # guests found in the cavity or channel body, summed over the exchanges
        # that drew one, and how many did
        c.update(cavity_sum_n=0, cavity_draws=0)

    # energy change of every accepted move (LJ only; a hard core has U = 0)
    du = 0.0
    _table = None if _typed_lj is None else _typed_lj.table

    def _group_energy(coords: FloatArray, spc: IntArray | None) -> float:
        """U of a group against every active slot (fluid and matrix) within uc,
        plus the pairs inside the group, minimum imaged."""
        coords = np.asarray(coords, float)
        if len(coords) == 0:
            return 0.0
        if _table is not None and spc is None:
            spc = np.zeros(len(coords), dtype=np.int64)
        total = 0.0
        if cl.n_active:
            d, sl = cl.query(coords, uc)
            live = np.isfinite(d)
            if live.any():
                if _table is None:
                    total += _lj_energy(d[live], uc)
                else:
                    rows = np.broadcast_to(np.arange(len(coords))[:, None], d.shape)
                    total += _lj_energy_typed(
                        d[live],
                        np.asarray(spc)[rows[live]],
                        cl.species_at(sl[live]),
                        uc,
                        _table,
                    )
        if len(coords) > 1:
            ii, jj = np.triu_indices(len(coords), k=1)
            dvec = coords[ii] - coords[jj]
            dvec -= np.round(dvec / box) * box
            dr = np.sqrt((dvec**2).sum(axis=1))
            if _table is None:
                total += _lj_energy(dr, uc)
            else:
                sp = np.asarray(spc)
                total += _lj_energy_typed(dr, sp[ii], sp[jj], uc, _table)
        return total

    step = 0
    while step < io_.sampling_window:
        if nvt:
            # NVT: only number-conserving moves; no insertions or deletions, so N
            # stays at the target exactly. An empty system has nothing to do, so
            # the window completes as a no-op rather than spinning forever on
            # null proposals.
            if N == 0:
                break
            # CANONICAL vs SEMI-GRAND, and the difference is the whole reason
            # they are two ensemble names. `semigrand` repacks: the repack
            # conserves N and DRAWS the species of every particle it inserts, so
            # composition is free, and it cannot hold a two-phase box because a
            # single phase costs no interface and nothing stops the box reaching
            # one. Measured: from an interface at k = 0.8 the composition ran to
            # nearly pure within half a million cycles.
            #
            # `NVT` with species proposes translations and pair identity
            # exchanges instead. Both conserve every species count exactly, so a
            # box held at x = 0.5 between two coexisting compositions cannot
            # become one phase, and two slabs are the equilibrium rather than a
            # transient.
            if canonical:
                choice = moves.draw_canonical_move(rng)
                # A CANONICAL WINDOW IS COUNTED IN PROPOSALS, not in exchanges.
                # `step` advances at the bottom of the exchange path, and both
                # canonical moves `continue` long before reaching it -- so
                # without this line the window never completes and the run hangs
                # having produced no output at all. That is exactly how it was
                # found: 115,708 neighbour queries in a window asking for 20
                # proposals.
                #
                # The unit differs from every other ensemble's, and that is
                # unavoidable rather than sloppy: a fixed-composition run
                # proposes no exchange, so "exchange steps" would make every
                # window empty. Here `sampling_window` counts translations plus
                # pair exchanges, and the log's mv_stp + swp_stp sum to it.
                step += 1
            else:
                choice = "repack"
        else:
            # One fixed move-type distribution regardless of N. Forcing "add" at
            # N=0 would drop the proposal ratio q(0->1)/q(1->0)=4 out of the
            # acceptance and bias P(N=0) by that factor. A move or remove
            # proposed at N=0 is a null / rejected attempt, counted so the window
            # advances as from any state.
            choice = moves.draw_move(rng, n_species)

        # ---------------------------------------------------------- translate
        if choice == "move":
            c["translate_tried"] += 1
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
                dr2 = (mf * (ranf1 - 0.5)) ** 2 + (mf * (ranf2 - 0.5)) ** 2
            else:
                # three independent displacement components; a z derived from
                # the other two draws would pin every displacement to a fixed 2D
                # surface and starve z mixing.
                ranf1, ranf2, ranf3 = rng.uniform(0, 1, 3)
                mx = (old[0] + mf * (ranf1 - 0.5)) % side[0]
                my = (old[1] + mf * (ranf2 - 0.5)) % side[1]
                mz = (old[2] + mf * (ranf3 - 0.5)) % side[2]
                moved = np.array([mx, my, mz])
                dr2 = (
                    (mf * (ranf1 - 0.5)) ** 2
                    + (mf * (ranf2 - 0.5)) ** 2
                    + (mf * (ranf3 - 0.5)) ** 2
                )
            if ens.pore_geometry is not None and not geometry.inside_pore(
                moved, ens, samp
            ):
                continue  # move into the wall: rejected
            if N == 1 and not has_matrix:
                # lone sphere in an empty box: no neighbours, no energy change,
                # so the standard Metropolis move is always accepted. Short-circuit
                # before any query, which is also what keeps the rng draw count
                # matching the untouched engine on this path.
                cl.move(slot, moved)
                c["translate_accepted"] += 1
                c["translate_sum_dr2"] += dr2
                continue
            # The overlap test must also see the frozen matrix, so a fluid
            # particle cannot move onto an obstacle; the cell list holds both.
            # Deactivating the mover's slot excludes it at O(1), with no rebuild.
            # A radius query returns
            # every neighbour inside the cutoff, so `query_saturating`'s doubling
            # loop has nothing left to guarantee.
            cl.remove(slot)
            accepted = False
            dU = 0.0
            if lj:
                d_new, s_new = cl.query_point(moved, uc, inclusive=False)
                # TYPED: the mover's own type against each neighbour's, so both
                # the overlap floor and the energy are per pair. `_typed_lj` is
                # None on an untyped run and every expression below is the one
                # that was there, bit for bit.
                mover_type = int(cl.species[slot]) if _typed_lj is not None else 0
                if d_new.size and _overlaps_any(
                    d_new, s_new, lower, mover_type, _typed_lj, cl
                ):
                    pass
                else:
                    d_old, s_old = cl.query_point(old, uc, inclusive=False)

                    def _u(
                        d: FloatArray,
                        slots: IntArray,
                        # bound at definition: this closure is rebuilt each
                        # iteration and B023 is right that a free loop variable
                        # here would be a latebinding bug waiting to happen
                        _mt: int = mover_type,
                    ) -> float:
                        if d.size == 0:
                            return 0.0
                        if _typed_lj is not None:
                            return _typed_lj.total_energy_typed(
                                d, _mt, cl.species_at(slots)
                            )
                        return float(np.sum(lj_pair_energy(d)))

                    dU = _u(d_new, s_new) - _u(d_old, s_old)
                    accepted = np.log(rng.uniform(0, 1)) <= -dU / temp
            else:
                accepted = cl.count_within(moved, geometry.overlap_radius(lower)) == 0
            if accepted:
                cl.move(slot, moved)  # in place; `alive` does not change
                c["translate_accepted"] += 1
                c["translate_sum_dr2"] += dr2
                du += dU
            else:
                cl.reactivate(slot)
            continue

        # ---------------------------------------------------------------- swap
        #
        # One particle changes identity in place. N is conserved and no geometry
        # moves, so nothing about the neighbour structure changes -- only the
        # species column of one slot.
        #
        # This branch did not exist. `moves.draw_move` has emitted "swap" for
        # n_species > 1 since M8, and `advance` fell through to the raise below,
        # so EVERY multi-species run crashed within a handful of steps. That is
        # the 8 remaining tier-2 failures, and it is the good failure mode: the
        # same omission in `_advance_single_particle` was a bare `else` that executed
        # the swap as a deletion instead.
        #
        # WHY THERE IS NO COMBINATORIAL TERM. A specific labelled particle changes
        # identity; this is not an aggregate counts move. Adding an n_s/(n_s'+1)
        # factor would double count the uniform-particle selection, and its
        # symptom is a mixture biased toward the minority species with total N
        # conserved -- easy to mistake for physics. The derivation is in
        # `moves.swap_log_acceptance`; detailed balance with the symmetric
        # uniform-particle / uniform-other-species proposal is satisfied by the
        # activity ratio alone.
        # -------------------------------------------------- pair identity swap
        #
        # THE CANONICAL DEMIXING MOVE. Two particles of DIFFERENT species trade
        # labels. Nothing moves, no particle is created or destroyed, and every
        # species count is identical before and after -- which is exactly the
        # property the semi-grand repack does not have and the reason a
        # fixed-composition box can hold an interface.
        #
        # It is also the move that does the physics. Demixing at fixed positions
        # is a labelling problem, and a translation cannot solve it at liquid
        # density: an A buried in a B-rich domain has nowhere to walk. The pair
        # exchange moves it in one step, at no cost in particle transport.
        #
        # PROPOSAL: two particles drawn uniformly from all N, rejected as a null
        # if they share a species. That is symmetric by inspection -- every
        # unordered pair has probability 2/(N(N-1)) forward and the same in
        # reverse, since the swap leaves N and the pair set alone -- so the
        # acceptance is min(1, exp(-beta dU)) with no proposal ratio.
        #
        # The alternative (draw i, then draw j from the particles unlike i) wastes
        # nothing but needs an argument: its per-pair probability involves the
        # count of unlike particles, which differs for the two species unless the
        # composition is exactly even. The two orderings do sum to the same total
        # forward and reverse, so it IS correct -- but "correct after an argument"
        # is worth less here than "correct by inspection", and the null proposals
        # cost one uniform draw and a comparison. The null rate is sum_s x_s^2,
        # which is a number worth knowing rather than a hidden inefficiency: it
        # rises as the box demixes, so a canonical run gets slower exactly as it
        # succeeds.
        if choice == "pair_swap":
            c["swap_tried"] += 1
            if N < 2 or not typed_insert:
                continue
            i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
            slot_i, slot_j = int(alive[i]), int(alive[j])
            a, b = int(cl.species[slot_i]), int(cl.species[slot_j])
            if a == b:
                continue  # null proposal: swapping two like labels is identity
            # dU over the two neighbourhoods, each EXCLUDING the other particle.
            # The i-j pair itself is invariant -- (a, b) becomes (b, a) and the
            # table is symmetric -- so leaving it out of both sums is not an
            # approximation, it is the term cancelling.
            du_pair = 0.0
            if _typed_lj is not None:
                for slot, old_t, new_t, other in (
                    (slot_i, a, b, slot_j),
                    (slot_j, b, a, slot_i),
                ):
                    d_s, sl_s = cl.query_point(cl.coords[slot], uc, inclusive=False)
                    if not d_s.size:
                        continue
                    keep = sl_s != other
                    d_s, sl_s = d_s[keep], sl_s[keep]
                    if not d_s.size:
                        continue
                    nb = cl.species_at(sl_s)
                    du_pair += _typed_lj.total_energy_typed(
                        d_s, new_t, nb
                    ) - _typed_lj.total_energy_typed(d_s, old_t, nb)
            if du_pair <= 0.0 or np.log(rng.uniform(0, 1)) <= -beta * du_pair:
                cl.species[slot_i] = b
                cl.species[slot_j] = a
                c["swap_accepted"] += 1
                du += du_pair
            continue

        if choice == "swap":
            c["swap_tried"] += 1
            if N == 0 or not typed_insert:
                continue  # null proposal: nothing to reassign
            assert log_z is not None  # narrowed by typed_insert
            j = int(rng.choice(N))
            slot = int(alive[j])
            cur = int(cl.species[slot])
            tgt = moves.draw_swap_target(rng, cur, guest_ids)
            # dU: the particle's surroundings do not move, but its identity
            # does, so every pair it participates in is re-read from the table at
            # the new type. Zero when the energy is not typed -- a single-sigma
            # potential cannot tell the two labels apart, and 0 is then the
            # honest value under the model the engine actually integrates, not an
            # approximation to a value it declines to compute.
            #
            # The overlap floor is NOT re-tested here. A swap to a larger species
            # can put a pair inside its new floor, and `table_pair_energy` returns
            # a huge but finite repulsion there rather than an exclusion, so the
            # move is rejected by the Boltzmann factor instead of by a hard test.
            # That is the same convention `pair_floor` documents (a pair at
            # exactly its floor is admissible, its energy large), and using the
            # acceptance keeps the swap reversible: a hard rejection in one
            # direction with none in the other would not be.
            du_swap = 0.0
            if _typed_lj is not None:
                d_s, sl_s = cl.query_point(cl.coords[slot], uc, inclusive=False)
                if d_s.size:
                    nb = cl.species_at(sl_s)
                    du_swap = _typed_lj.total_energy_typed(
                        d_s, tgt, nb
                    ) - _typed_lj.total_energy_typed(d_s, cur, nb)
            log_a = moves.swap_log_acceptance(log_z, cur, tgt, beta, du_swap)
            if np.log(rng.uniform(0, 1)) <= log_a:
                cl.species[slot] = tgt
                c["swap_accepted"] += 1
                du += du_swap
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
            c["repack_tried"] += 1
            n_add = n_remove = depth + 1  # remove k, insert k; N conserved
        elif choice == "add":
            c["insert_tried"] += 1
            n_add, n_remove = depth + 1, depth
        elif choice == "remove":
            c["delete_tried"] += 1
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
        # The channel body, when there is one. Bound here rather than inside the
        # depth >= 1 branch because the trial cloud below is shared with the
        # depth-0 path, which never builds a body: channel requires depth >= 1,
        # so on that path this stays None and the plain cavity cloud is drawn.
        chan_body: channel_move.ChannelBody | None = None
        if depth == 0 and not full_repacking:
            if choice == "add":
                center = rng.uniform(0.0, 1.0, size=3)[:dim] * side
                anchor = -1
            else:  # remove or repack: local, anchored on an existing particle
                anchor = int(alive[rng.choice(N)])
                center = (
                    cl.coords[anchor]
                    + rng.uniform(-samp.cavity_radius, samp.cavity_radius, size=dim)
                ) % side
            if n_remove > 0:
                idx_cav = geometry.indices_in_cavity_cl(
                    cl, center, samp.cavity_radius, side, two_d
                )
                n_cav = len(idx_cav)
            else:
                idx_cav = np.empty(0, dtype=int)
                n_cav = 0
            c["cavity_sum_n"] += n_cav
            c["cavity_draws"] += 1
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
            chan_R = channel_move.draw_rotation(rng, dim, channel=channel)
            # THE BODY'S LENGTH, BEFORE ITS POSITION, which is the order the
            # removal leg needs. On `channel_length = "full"` the length comes
            # from the orientation (`full_span_length`), so it is known here,
            # one line after the rotation is drawn and before anything has
            # looked at the configuration. Both legs compute the same number
            # from the same R, which is what lets the removal leg offset the
            # body around its anchor over the body's own extent -- the fixed
            # cuboid's construction, unchanged, with a length that happens to
            # vary per move.
            chan_len = samp.channel_length
            if channel_full and chan_R is not None:
                chan_len = channel_move.full_span_length(
                    chan_R, box, samp.channel_width, dim, chan_reach
                )
                if chan_len <= 0.0:
                    # This direction runs within reach of its own periodic image
                    # along its whole length, so there is no body to propose
                    # into. The step is already counted above (`add_step` and
                    # friends), so this bails like the short-pool one below:
                    # a proposal made and not accepted.
                    continue
            anchor_d1 = -1
            if choice == "add":
                center = rng.uniform(0.0, 1.0, size=3)[:dim] * side
            else:
                anchor_d1 = int(alive[rng.choice(N)])
                _base = cl.coords[anchor_d1]
                if channel:
                    assert chan_R is not None  # channel implies a drawn rotation
                    center = channel_move.draw_center_offset(
                        rng,
                        _base,
                        chan_R,
                        side,
                        dim,
                        half_len=0.5 * chan_len,
                        half_wid=0.5 * samp.channel_width,
                    )
                else:
                    center = (
                        _base
                        + rng.uniform(-samp.cavity_radius, samp.cavity_radius, size=dim)
                    ) % side
            anchor = anchor_d1 if choice != "add" else -1
            chan_body = (
                channel_move.ChannelBody.build(
                    center,
                    chan_R,
                    side,
                    box,
                    dim,
                    channel_length=chan_len,
                    channel_width=samp.channel_width,
                    trial_density=samp.trial_density,
                    n_trials_base=n_trials_base,
                    # The floor a chord-derived trial count must respect: the
                    # reverse-move mirror claims one trial slot per
                    # removed molecule, so a body holding fewer than the group
                    # size cannot express the move at all. `n_add`/`n_remove` are
                    # still the depth-based pair here -- `full_repacking`
                    # overwrites them BELOW, from an occupancy this body has not
                    # been asked for yet -- so this is `max(2, depth + 1)`, which
                    # is the literal 2 it replaces at depth 0 and 1. The
                    # full_repacking path keeps its own `max(n_add, n_remove) >
                    # n_trials` bail, which is the guard for the occupancy case.
                    n_trials_min=max(2, n_add, n_remove),
                    full=channel_full,
                    reach=chan_reach,
                )
                if channel and chan_R is not None
                else None
            )
            if chan_body is not None:
                n_trials = chan_body.n_trials
            idx_cav = channel_move.cavity_members(
                cl, chan_body, center, samp.cavity_radius, side, two_d
            )
            n_cav = len(idx_cav)
            c["cavity_sum_n"] += n_cav
            c["cavity_draws"] += 1
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
                if max(n_add, n_remove) > n_trials:
                    # Both counts must fit the trial pool, not just the
                    # insertions: `keep_mask` is sized n_trials and the removed
                    # molecules claim n_remove of its slots, so gating on n_add
                    # alone let the remove branch through at n_cav == n_trials+1
                    # (n_add = n_cav-1 fits, n_remove = n_cav does not) and it
                    # died in rng.choice. Gating on the max also keeps the two
                    # directions inverses of each other: an add at occupancy m
                    # proposes (m+1, m) and the remove that undoes it proposes
                    # (m, m+1) at occupancy m+1, so both are blocked at the same
                    # m >= n_trials instead of the add alone.
                    continue  # add or remove count exceeds the trial cloud
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
                # An add of (n_cav, n_cav - 1) instead would leave one occupant
                # behind, and its reverse (remove n_cav, insert n_cav - 1) is a
                # move the remove branch never proposes: detailed balance fails
                # whenever n_cav < depth. The n_cav == 0 add rule above, (1, 0),
                # and full_repacking use the same convention.
                if choice == "repack":
                    # repack must stay balanced: remove and add the same count,
                    # capped at what the cavity actually holds, so N is conserved
                    n_add = n_remove = n_cav
                elif choice == "add":
                    n_add, n_remove = n_cav + 1, n_cav
                else:
                    n_add, n_remove = n_cav - 1, n_cav
                if max(n_add, n_remove) > n_trials:
                    # see the full_repacking branch above: the removal count
                    # claims trial slots too, so gating on n_add alone is both a
                    # crash at n_cav == n_trials+1 and a direction-dependent
                    # threshold
                    continue  # add or remove count exceeds the trial cloud

        # Split the system into in-cavity (to remove) and the rest. The rest is
        # never materialised: it is exactly the cell list's active fluid once the
        # chosen slots are tombstoned, and its only other use was a count.
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
            continue
        # trial molecules in the cavity
        if chan_body is not None:
            trials = chan_body.trial_cloud(rng, n_trials)
        else:
            trial_nums = rng.uniform(
                -samp.cavity_radius, samp.cavity_radius, size=(dim, n_trials)
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

        # A trial atom's SPECIES is drawn here, with its position. An insertion
        # that defaulted to type 0 while removals took any type would let every
        # species but the first leave the box and never arrive: add and remove
        # must be each other's mirror, which is the one thing the acceptance
        # rule assumes.
        #
        # Drawn from `p_trial`, a PROPOSAL knob: it must not be able to move
        # the equilibrium composition, and `moves.species_draw_correction` is what
        # divides it back out of the acceptance. That the two cancel is not taken
        # on trust -- it is the arbiter in
        # tests/engine/test_species_bookkeeping_arbiter.py.
        trial_species = (
            rng.choice(n_guest, size=n_trials, p=p_trial_guest)
            if typed_insert
            else None
        )

        # The exclusion set is fluid-rest PLUS frozen matrix: trials and shadows
        # that overlap either are rejected. Deactivating the chosen slots excludes
        # them in O(n_remove), with no rebuild. The slots stay tombstoned until the move
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
            pore_valid = cast(  # trials is (M, dim): the batch arm of inside_pore
                "BoolArray | None",
                geometry.inside_pore(trials, ens, samp)
                if ens.pore_geometry is not None
                else None,
            )
            # one query for both bodies: the trials and the members share a
            # cuboid, and a query costs numpy dispatch, not arithmetic
            # The members' species come off the SLOTS being tombstoned, so the
            # column and the coordinates are read from the same index in the same
            # breath and cannot come apart.
            member_species = (
                cl.species[removed_slots]
                if (typed_insert and len(removed_slots))
                else None
            )
            raw, raw_in = geometry.find_neighbors_pair(
                tree,
                trials,
                disk_in,
                lower,
                rc_sel,  # member weights are a proposal quantity: aux radius
                nn,
                keep_mask=keep_mask,
                valid=pore_valid,
                trial_species=trial_species,
                member_species=member_species,
                # per-pair admissibility when the energy is typed; None
                # leaves the scalar floor test exactly as it was
                core_floor=(None if _typed_lj is None else _typed_lj.table.core_floor),
                dists_needed=hs_needs_dists,
            )
            if raw is None:
                if n_add != 0:
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
                    raw,
                    tree,
                    kern,
                    pot_spec,
                    ens,
                    potential,
                    choice,
                    N,
                    n_rest,
                )

            if n_remove != 0:
                if raw_in is None:
                    continue
                mol_in_cav, _ = _weighted_trials(
                    raw_in,
                    tree,
                    kern,
                    pot_spec,
                    ens,
                    potential,
                    choice,
                    N,
                    n_rest,
                )
                if len(mol_in_cav) != n_remove:
                    continue
            else:
                mol_in_cav = TrialSet(np.empty((0, dim)), np.empty(0))

            # assemble the pool the forward selection draws from
            #
            # PROVENANCE. The pool is `surviving trials + tombstoned
            # members`, and then it is PERMUTED. Which rows are members is
            # therefore known only here, from the permutation itself -- and the
            # recorded trap is that `pool = pool[rng.permutation(len(pool))]`
            # threw the permutation away, so by the time the acceptance needed it
            # the information was gone. Wiring the insert path without carrying it
            # is wrong at a rate set by the member-reselection frequency, measured
            # at 12 of 214 in one hs_branch window, which is exactly the size of
            # error that looks like a plausible isotherm.
            #
            # `pool_row` is stamped AFTER the permutation, so a row number means
            # the same thing everywhere downstream: to the DP, which returns pool
            # rows; to the Rosenbluth selection, which carries the column through
            # its deletes; and to `member_rows` here. Nothing is identified by
            # coordinate at any point.
            if n_remove != 0:
                assert km is not None  # raw was handled above; km rides with it
                surviving = trial_set.mask(km)
                n_surv = len(surviving)
                pool = surviving.merge(mol_in_cav)
                perm = rng.permutation(len(pool))
                pool = pool[perm]
                member_rows = np.flatnonzero(perm >= n_surv)
            else:
                pool = trial_set.copy()
                member_rows = np.empty(0, dtype=np.int64)
            pool.pool_row = np.arange(len(pool), dtype=np.int64)
            # The typed clique arguments, named once so the four kernel call
            # sites below cannot disagree. A forward and a reverse weight
            # computed against different conflict graphs would break detailed
            # balance in a way that looks like a density offset.
            _clique_kw: dict[str, Any] = (
                {}
                if _typed_lj is None
                else {
                    "species": pool.species,
                    "floor": _typed_lj.table.core_floor,
                    "table": _typed_lj.table,
                }
            )
            # SNAPSHOT, because `select_forward` shrinks the pool IN PLACE.
            # `TrialSet.delete` mutates and returns self (its docstring says so,
            # and the Rosenbluth loop relies on it), so by the time the acceptance
            # is assembled the caller's `pool` has lost every atom the selection
            # picked or masked away -- while `chosen_rows` still numbers rows of
            # the pool as it was. Indexing the shrunken object with those numbers
            # is what the arbiter's first run caught, as an IndexError rather than
            # a wrong answer only because the pool happened to be short enough.
            #
            # The species column is what the correction needs, and it is taken
            # here, once, before anything can mutate it.
            pool_species = pool.species.copy()

            # channel mode: one sweep over the pool serves both the forward count
            # (Z_{n_add}) and the reverse count (Z_{n_remove}); kmax covers both.
            chan_sweep: ChannelSweepHardCore | ChannelSweepLJ | None = None
            if chan_body is not None and len(pool):
                kmx = max(n_add, n_remove)
                chan_sweep = chan_body.make_sweep(
                    pool,
                    kmx,
                    ljbranch=ljbranch,
                    lower=lower,
                    rc_sel=rc_sel,  # DP interaction window: aux radius
                    beta=beta,
                    # TYPED: per-pair conflict floors, and the table for the
                    # per-pair intra-clique energies. None on an untyped run,
                    # which then takes the scalar classes it always did.
                    floor=(None if _typed_lj is None else _typed_lj.table.core_floor),
                    table=None if _typed_lj is None else _typed_lj.table,
                )

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
                if len(trial_set) < n_add:
                    continue
                if branch:
                    if ljbranch:
                        if channel:
                            if chan_sweep is None:
                                continue  # empty pool: no clique, reject
                            assert isinstance(chan_sweep, ChannelSweepLJ)
                            sum_new = chan_sweep.log_weight(n_add)
                            if not np.isfinite(sum_new):
                                continue
                            idx: Sequence[int] | None = chan_sweep.sample(n_add, rng)
                            if idx is None:
                                continue
                            chosen_rows = list(idx)
                            chosen_coords = pool.coords[chosen_rows]
                        else:
                            # true LJ+branch: Boltzmann-weighted sum over all
                            # non-overlapping k-cliques, one drawn proportional to its
                            # weight. Returns the chosen clique's coords and
                            # log W_new = log sum over cliques.
                            _cc, _lw, _rows = _ljbranch_forward(
                                pool,
                                n_add,
                                lower,
                                box,
                                rc_sel,
                                beta,
                                rng,
                                **_clique_kw,
                            )
                            if _cc is None or _lw is None or _rows is None:
                                continue
                            chosen_coords, sum_new = _cc, _lw
                            chosen_rows = list(_rows)
                        e_u_r_new = 0.0
                    else:
                        if channel:
                            if chan_sweep is None:
                                continue  # empty pool: no clique, reject
                            assert isinstance(chan_sweep, ChannelSweepHardCore)
                            num_new = chan_sweep.count(n_add)
                            if num_new == 0:
                                continue
                            idx = chan_sweep.sample(n_add, rng)
                            assert idx is not None  # num_new > 0
                        else:
                            num_new, idx = count_and_sample(
                                pool.coords,
                                n_add,
                                lower,
                                box_lengths=box,
                                rng=rng,
                                species=_clique_kw.get("species"),
                                floor=_clique_kw.get("floor"),
                            )
                            if num_new == 0 or idx is None:
                                continue
                        chosen_rows = list(idx)
                        chosen_coords = pool.coords[chosen_rows]
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
                    chosen_rows = [int(r) for r in chosen.pool_row]
            else:
                sum_new = 0.0
                e_u_r_new = 0.0
                chosen_coords = np.empty((0, dim))
                chosen_rows = []

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
                                pool, n_remove, lower, box, rc_sel, beta, **_clique_kw
                            )
                        e_u_r_current = 0.0
                    else:
                        if channel:
                            if chan_sweep is not None:
                                assert isinstance(chan_sweep, ChannelSweepHardCore)
                                num_cur = chan_sweep.count(n_remove)
                            else:
                                num_cur = 0
                        else:
                            num_cur, _ = count_and_sample(
                                pool.coords,
                                n_remove,
                                lower,
                                box_lengths=box,
                                species=_clique_kw.get("species"),
                                floor=_clique_kw.get("floor"),
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

            if choice == "repack":
                # clique composition: positions drawn, and those not a removed
                # member put back
                _cpos = len(chosen_coords)
                _cnew = _cpos - _count_reused(chosen_coords, disk_in)
                c["repack_sum_size"] += _cpos
                c["repack_sum_new"] += _cnew

            # LJ tail: a system-level energy change, applied exactly once per move
            # with the same sign in both directions. For branch the per-particle
            # pair energy is inside sum_new / sum_current; for rosenbluth it is
            # inside the Boltzmann trial weights. Either way only the mean-field
            # tail remains as an explicit term.
            if lj:
                tail = pot_spec.lj_tail_pref
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
                # TYPED: the correction is a per-pair quantity or it is a wrong
                # number, not a coarse one. `chosen_species` is the drawn clique's
                # types (M21: by pool row, never by coordinate); the removed
                # group's come from the cell list at the slots that were
                # tombstoned, which still carry their species -- tombstoning
                # clears `active`, not `species`.
                _aux_tab = None if _typed_lj is None else _typed_lj.table
                if _aux_tab is None:
                    _aux_ins_spc = _aux_rem_spc = None
                else:
                    # `typed_insert` is False on a ONE-species typed run (there
                    # is no species to draw), but the table is still per-pair and
                    # its single pair is not sigma = 1 in general -- so the types
                    # are all zero rather than absent. Passing None there would
                    # silently fall back to the unit potential, which is the
                    # defect this whole path exists to close.
                    _aux_ins_spc = (
                        np.asarray(pool_species)[chosen_rows]
                        if typed_insert
                        else np.zeros(len(chosen_coords), dtype=np.int64)
                    )
                    _aux_rem_spc = (
                        cl.species_at(chosen_slots)
                        if chosen_slots is not None
                        else np.zeros(0, dtype=np.int64)
                    )
                u_ins_s, u_ins_f = _lj_set_energy_split(
                    chosen_coords,
                    box,
                    rc_sel,
                    uc,
                    rest_neighbors=cl,
                    species=_aux_ins_spc,
                    table=_aux_tab,
                )
                u_rem_s, u_rem_f = _lj_set_energy_split(
                    disk_in,
                    box,
                    rc_sel,
                    uc,
                    rest_neighbors=cl,
                    species=_aux_rem_spc,
                    table=_aux_tab,
                )
                delta_corr = beta * ((u_ins_f - u_ins_s) - (u_rem_f - u_rem_s))
                # aux_on implies ljbranch implies lj, so the tail term exists
                delta_E_lj = (delta_E_lj or 0.0) + delta_corr

            # The depth >= 1 acceptance is the original's, complete as written;
            # the depth-0 mirror scheme likewise needs no extra terms.
            corr = 0.0

            # THE SPECIES HALF OF THE TRIAL-DENSITY MEASURE.
            #
            # `p_trial` decides which species a trial atom is proposed as, and it
            # is a proposal knob: it must not be able to move the equilibrium
            # composition. This term is what divides it back out. Exactly zero on
            # a single-species run -- not approximately, not "negligible": the
            # branch is not taken, so the acceptance is the same float it was.
            #
            # Sign. `species_draw_correction` returns a contribution to
            # log(acceptance ratio); `neg_log_acc` below is MINUS that, so it
            # enters as `- species_corr`, in the same place and with the same sign
            # convention as `corr`. Spelled out because an inverted combinatorial
            # prefactor in this acceptance has cost this project a session once
            # already, and the two conventions meet within a few lines.
            #
            # Provenance is by POOL ROW throughout: `member_rows` came from the
            # permutation, `chosen_rows` from whichever selector ran, and
            # `pool.species` is row-aligned with both.
            if typed_insert:
                assert log_p_trial is not None  # narrowed by typed_insert
                species_corr = moves.species_draw_correction(
                    log_p_trial, pool_species, member_rows, chosen_rows
                )
            else:
                species_corr = 0.0
            chosen_species = pool_species[chosen_rows] if typed_insert else None

            # THE PER-SPECIES ACTIVITY. The single-species engine carries
            # one `beta * mu` for the NET particle an exchange moves; a mixture
            # has no single activity, so each inserted atom enters at its own
            # `z_s` and each deleted one leaves at its own.
            #
            # `None` on an untyped run, and each branch below then uses the exact
            # `beta * mu` expression it always did -- which is what keeps the
            # single-species chain bit-identical.
            #
            # Note this is non-zero for a REPACK, where the untyped engine
            # correctly has no activity term at all: a repack conserves particle
            # number but not composition.
            activity = (
                moves.species_activity_term(
                    log_z, pool_species, member_rows, chosen_rows
                )
                if typed_insert
                else None
            )

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

            # neg_log_acc is minus the log acceptance ratio, clamped at 0; the
            # move is accepted when neg_log_acc <= -log(u), u ~ U(0,1).
            if choice == "repack":
                c["repack_evaluated"] += 1
                # number-conserving: the mu prefactors and the k_trials factors
                # cancel, leaving the Rosenbluth ratio and the energy change
                mu_excess = -sum_new + sum_current
                neg_log_acc = (
                    mu_excess
                    + delta_E
                    - corr
                    - species_corr
                    - (activity if activity is not None else 0.0)
                )
            elif choice == "add":
                c["insert_evaluated"] += 1
                mu_ideal = -np.log(total_vol / (N + 1))
                if branch:
                    k_trials = -np.log((n_trials - n_remove) / n_add)
                else:
                    k_trials = -np.log(n_trials - n_remove)
                mu_excess = -sum_new + sum_current - k_trials
                neg_log_acc = (
                    mu_ideal
                    - (beta * mu if activity is None else activity)
                    + mu_excess
                    + delta_E
                    - corr
                    - species_corr
                )
            else:
                c["delete_evaluated"] += 1
                mu_ideal = -np.log(N / total_vol)
                if branch:
                    k_trials = -np.log((n_trials - n_add) / n_remove)
                else:
                    k_trials = -np.log(n_trials - n_add)
                mu_excess = -sum_new + sum_current + k_trials
                neg_log_acc = (
                    mu_ideal
                    - (-beta * mu if activity is None else activity)
                    + mu_excess
                    + delta_E
                    - corr
                    - species_corr
                )
            neg_log_acc = max(0.0, neg_log_acc)
            if neg_log_acc <= -np.log(rng.uniform(0, 1)):
                if lj:
                    # energy change: the inserted group against the rest plus its
                    # own pairs, less the same for the removed group, plus the
                    # tail. Reuses the full-radius energies already in hand.
                    if aux_on:
                        d_pair = u_ins_f - u_rem_f
                    elif ljaux:
                        d_pair = u_ins - u_rem
                    else:
                        rem_spc = (
                            None
                            if _table is None or chosen_slots is None
                            else cl.species_at(chosen_slots)
                        )
                        d_pair = _group_energy(
                            chosen_coords, chosen_species
                        ) - _group_energy(disk_in, rem_spc)
                    du += d_pair + (tail or 0.0) * (n_new**2 - n_old**2)
                commit_exchange(chosen_coords, chosen_slots, chosen_species)
                committed = True
                if choice == "repack":
                    c["repack_accepted"] += 1
                    c["repack_sum_size_accepted"] += _cpos
                    c["repack_sum_new_accepted"] += _cnew
                elif choice == "add":
                    c["insert_accepted"] += 1
                else:
                    c["delete_accepted"] += 1
        finally:
            if not committed:
                for _s in removed_slots:
                    cl.reactivate(int(_s))

    # The window's only materialisation of the configuration, in `alive` order.
    state.positions = np.array(cl.coords[alive[:N]], dtype=float)
    # A typed run always writes its species column back, even when the caller
    # passed none: the cell list was seeded from zeros and holds the real labels.
    if state.species is not None or n_species > 1:
        state.species = np.array(cl.species[alive[:N]], dtype=np.int64)
    state.step += io_.sampling_window
    out: dict[str, Any] = {"N": N, "columns": c, "dU": du if lj else None}
    if state.species is not None:
        _counts = np.bincount(state.species, minlength=n_species)
        _names = (
            kern.species.names
            if kern.species is not None
            else tuple(str(i) for i in range(n_species))
        )
        out["n_by_species"] = {n: int(v) for n, v in zip(_names, _counts)}
    return out
