"""The MOF metropolis loop: GCMC of one or more guest species in a rigid framework.

This mirrors `_advance_metropolis` in MC_engine.py move for move -- the same
textbook grand-canonical acceptance, the same "null proposal at N=0" handling --
with exactly three differences, all forced by the framework:

  1. every insertion and every move adds the guest-framework energy
     (FrameworkField, the A3.5-gated core) to the guest-guest energy;
  2. overlap is checked against the framework too, per pair type (A3.3);
  3. there is no long-range tail (pore convention), so the tail term is simply
     absent rather than set to something.

It is a SEPARATE function, not a branch inside _advance_metropolis, so that
frozen path stays byte-identical for every existing mode. Everything here runs in
the engine's reduced units: the caller reduces positions, box, cutoff and
activity by the reference scale before calling, exactly as the reduction
pipeline (P0.4) defines.

There is ONE loop, `advance_metropolis_mof_ms` (A4.2/A4.3): a species column
parallel to positions (D2), a per-species reduced activity, an insertion that
first picks a species with fixed probabilities, and an identity-swap move that
resamples a chosen guest's species. At a single guest species it takes the
three-way 0.5/0.25/0.25 move split with no species or swap draw, so it consumes
randomness exactly as the original Phase 3 single-species loop did; that
trajectory is pinned by a golden fixed-seed test (test_mof_metropolis_ms). The
single-species entry point `advance_metropolis_mof` is a thin wrapper kept for
its callers' convenience, not a second implementation.

Correctness is not asserted by this module; it is asserted by the move-delta
parity test against ref_energy.py (A3.5), which recomputes U(after) - U(before)
from scratch in angstrom/kelvin and requires agreement to 1e-10. The loop is
written to make that check pass, not to be trusted on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import moves
from ._types import FloatArray, IntArray
from .mof_field import FrameworkField
from .potentials import overlap
from .potentials.lj_table import LennardJonesTable
from .potentials.pair_table import PairTable


@dataclass
class MofResult:
    """What one window of the MOF metropolis loop produced.

    `species` and `n_by_type` are populated only by the multi-species loop; the
    single-species loop leaves them None, so its callers and result shape are
    unchanged (additive, A4.2)."""

    n: int
    positions: FloatArray  # reduced units, (N, dim)
    counters: dict[str, int]
    loading: float  # guests per unit cell, a reporting convenience
    species: IntArray | None = None  # per-particle type id, aligned to positions
    n_by_type: dict[int, int] | None = None  # per-species counts N_s (A4.2)
    n_evals: int = 0  # guest-guest pair distances evaluated (efficiency metric)


def _guest_guest_energy(
    trial: FloatArray,
    others: FloatArray,
    other_species: IntArray,
    box: FloatArray,
    pot: LennardJonesTable,
    trial_species: int,
    table: PairTable,
    nn: int,
    uc: float,
    tree: Any = None,
) -> tuple[float, bool, int]:
    """Guest-guest energy of `trial` (a `trial_species` guest) against a fluid of
    mixed species, whether it overlaps any of them, and the number of guest-guest
    pair distances evaluated (native n_evals accounting for the efficiency metric;
    it never enters the counters dict, so the single-species golden pin is
    unaffected).

    The neighbours carry their own type ids, so the tree query keeps its indices
    to look up `other_species`, the overlap floor is the per-pair c*sigma_ij
    between the trial and each neighbour (A1.4/A3.3), and the energy is summed
    per pair type. `pot` must already carry trial_type = trial_species. With one
    species every neighbour is the trial type and the per-pair floor collapses
    to the scalar self-pair diameter, which is what keeps the single-species
    stream identical to the pinned Phase 3 one (A4.5).
    """
    if len(others) == 0:
        return 0.0, False, 0
    # `tree` lets the caller hand in a kd-tree it has already built over exactly
    # these `others`. Profiling the multi-species metropolis path found this
    # function called 105344 times in one test with a tree rebuilt on EVERY call,
    # and the translate move calls it TWICE over the same `others` -- once for the
    # displaced position and once for the original -- so half those builds were
    # duplicates of the build beside them. Passing the tree in is bit-identical by
    # construction: the same tree over the same points answers the same query.
    if tree is None:
        tree = cKDTree(others, boxsize=box)
    k = min(nn, len(others))
    d, idx = tree.query(trial, k=k, distance_upper_bound=uc)
    d = np.atleast_1d(d)
    idx = np.atleast_1d(idx)
    finite = np.isfinite(d)
    d = d[finite]
    idx = idx[finite]
    if d.size == 0:
        return 0.0, False, 0
    neigh = np.asarray(other_species, int)[idx]
    # per-pair floor c*sigma_ij between the trial type and each neighbour's type,
    # compared in geometry so the convention lives in exactly one place
    if overlap.pair_overlaps(d, trial_species, neigh, table.core_floor).any():
        return 0.0, True, int(d.size)
    u = pot.total_energy(d, neighbour_types=neigh)
    return u, False, int(d.size)


def advance_metropolis_mof(
    positions: FloatArray,
    rng: np.random.Generator,
    *,
    field: FrameworkField,
    table: PairTable,
    guest_type: int,
    beta: float,
    z_star: float,
    box: FloatArray,
    volume: float,
    n_cells: float,
    upper_cutoff: float,
    move_factor: float,
    n_neigh: int,
    window: int,
) -> MofResult:
    """One sampling window of single-species GCMC in the framework.

    A thin wrapper over `advance_metropolis_mof_ms` with a one-entry activity
    map: at a single species the ms loop takes the same three-way move choice
    and draws no species or swap randomness, so the trajectory is the original
    Phase 3 single-species one, pinned bit-for-bit by the golden fixed-seed test.
    The result is stripped back to the single-species shape (no species column,
    no swap counters) so existing callers see exactly what they always did.

    `z_star` is the reduced activity z* = f sigma_ref^3 / (kB T) (P0.4), so the
    insertion acceptance is z* V / (N+1) * exp(-beta dU) with dU the FULL energy
    change including the framework. Deletion is the exact reverse. `volume` and
    `box` are reduced; `n_cells` is only for the loading report.
    """
    spc = np.full(len(positions), guest_type, dtype=np.int64)
    res = advance_metropolis_mof_ms(
        positions,
        spc,
        rng,
        field=field,
        table=table,
        z_star={guest_type: z_star},
        beta=beta,
        box=box,
        volume=volume,
        n_cells=n_cells,
        upper_cutoff=upper_cutoff,
        move_factor=move_factor,
        n_neigh=n_neigh,
        window=window,
    )
    counters = {k: v for k, v in res.counters.items() if not k.startswith("swap")}
    return MofResult(
        n=res.n, positions=res.positions, counters=counters, loading=res.loading
    )


# The move split now lives in gcmc/moves.py (M8), so the main engine and this
# loop cannot drift apart on it. Re-exported under the old private names because
# tests and the golden pin refer to them, and because the reason they are what
# they are -- swap carved out of the translate budget so P(add) == P(remove) --
# is documented there rather than twice.
_MS_MOVES = moves.MS_MOVES
_MS_PROBS = moves.MS_PROBS


def mof_energy_decomposition(
    positions: FloatArray,
    species: IntArray,
    *,
    field: FrameworkField,
    table: PairTable,
    beta: float,
    upper_cutoff: float,
) -> dict[str, float]:
    """Full reduced-unit energy decomposition of a configuration, computed from
    scratch (O(N^2), off the sampling hot path). Returns host-guest per species
    (`u_hg_<name>`), guest-guest per unordered species pair (`u_gg_<a>_<b>`), and
    `u_total`. Used by the record-run stream (R2.4b) so fluctuation-formula
    enthalpies and partial-molar properties are recoverable at any truncation;
    it also doubles as the drift oracle for an incremental-U tracker (Q9)."""
    pos = np.ascontiguousarray(positions, float)
    spc = np.ascontiguousarray(species, np.int64).reshape(-1)
    types = sorted({int(t) for t in spc.tolist()})
    names = {t: table.types[t] for t in types}
    pots = {
        t: LennardJonesTable(
            beta=beta, table=table, trial_type=t, upper_cutoff=upper_cutoff
        )
        for t in types
    }
    box = field.box
    out: dict[str, float] = {}
    u_hg: dict[int, float] = dict.fromkeys(types, 0.0)
    for i in range(len(pos)):
        s = int(spc[i])
        u_hg[s] += float(field.energy(pos[i], s))
    u_gg: dict[tuple[int, int], float] = {}
    for i in range(len(pos)):
        si = int(spc[i])
        for j in range(i + 1, len(pos)):
            sj = int(spc[j])
            dr = pos[i] - pos[j]
            dr -= box * np.round(dr / box)
            r = float(np.sqrt(dr @ dr))
            if 0.0 < r < upper_cutoff:
                key = (si, sj) if si <= sj else (sj, si)
                u_gg[key] = u_gg.get(key, 0.0) + float(
                    pots[si].total_energy(np.array([r]), neighbour_types=np.array([sj]))
                )
    total = 0.0
    for t in types:
        label = names[t]
        out[f"u_hg_{label}"] = u_hg[t]
        total += u_hg[t]
    for (a, b), v in sorted(u_gg.items()):
        la = names.get(a, str(a))
        lb = names.get(b, str(b))
        out[f"u_gg_{la}_{lb}"] = v
        total += v
    out["u_total"] = total
    return out


def advance_metropolis_mof_ms(
    positions: FloatArray,
    species: IntArray,
    rng: np.random.Generator,
    *,
    field: FrameworkField,
    table: PairTable,
    z_star: dict[int, float],
    beta: float,
    box: FloatArray,
    volume: float,
    n_cells: float,
    upper_cutoff: float,
    move_factor: float,
    n_neigh: int,
    window: int,
    species_probs: dict[int, float] | None = None,
) -> MofResult:
    """One sampling window of multi-species GCMC in the framework (A4.2/A4.3).

    `z_star` maps each guest type id to its reduced activity z*_s = f_s
    sigma_ref^3 / (kB T); its keys ARE the guest alphabet. `species` is the type
    id of each current guest, aligned to `positions`. `species_probs` is the fixed
    insertion selection probability per guest type (default uniform); it is a
    property of the move set, never of the current occupancy, so detailed balance
    holds (the trial-set invariant).

    Acceptance factors, per-species (n_s = current count of species s):
      insert s : z*_s V / (n_s + 1) exp(-beta dU),  s picked with prob p_s
      delete s : n_s / (z*_s V) exp(+beta dU),       s picked with prob p_s, then a
                 uniform particle of species s
      swap s->s' : (z*_s' / z*_s) exp(-beta dU), position fixed, particle chosen
                   uniformly among all N and re-typed to a uniform other species
    The p_s in insert and delete cancel because add and remove select a species
    the same way with equal move probability, leaving the single-species forms.

    Single species (len(z_star) == 1) is a hard special case: the three-way
    0.5/0.25/0.25 move choice, the absence of any species or swap draw, and
    `rng.integers(N)` for the chosen particle reproduce the original Phase 3
    single-species stream call for call, so a one-species run is bit-identical
    to the pre-Phase-4 trajectory (A4.5, pinned by the golden fixed-seed test).
    """
    pos = np.ascontiguousarray(positions, float)
    spc = np.ascontiguousarray(species, np.int64).reshape(-1)
    if len(spc) != len(pos):
        raise ValueError("species column and positions differ in length")
    guest_types = sorted(z_star)
    n_species = len(guest_types)
    if species_probs is None:
        species_probs = dict.fromkeys(guest_types, 1.0 / n_species)
    gt_arr = np.array(guest_types, dtype=np.int64)
    p_arr = np.array([species_probs[t] for t in guest_types], dtype=float)
    dim = pos.shape[1] if pos.ndim == 2 and len(pos) else field.pos.shape[1]

    # one potential per guest type; construction is a few refs, the hot path just
    # indexes this by the trial species
    pots = {
        t: LennardJonesTable(
            beta=beta, table=table, trial_type=t, upper_cutoff=upper_cutoff
        )
        for t in guest_types
    }
    c = {
        "move_step": 0,
        "move_accept": 0,
        "add_step": 0,
        "add_accept": 0,
        "remove_step": 0,
        "remove_accept": 0,
        "swap_step": 0,
        "swap_accept": 0,
    }
    box = np.asarray(box, float)
    side = box
    logV = math.log(volume)
    logz = {t: math.log(z_star[t]) for t in guest_types}

    def pick_species() -> int:
        """Insertion/deletion species draw; skipped (no RNG) when single-species,
        which is what keeps the one-species reduction bit-identical."""
        if n_species == 1:
            return guest_types[0]
        return int(rng.choice(gt_arr, p=p_arr))

    # native n_evals: guest-guest pair distances evaluated this window.
    # `gg` wraps the energy helper so the six call sites stay 2-tuple and the
    # count never touches control flow, RNG, or the counters dict (the
    # single-species golden pin is therefore unaffected).
    _gg_fn = _guest_guest_energy
    n_evals = 0

    def gg(
        trial: FloatArray,
        others: FloatArray,
        others_s: IntArray,
        box_: FloatArray,
        pot: LennardJonesTable,
        ts: int,
        tbl: PairTable,
        nn: int,
        uc: float,
        tree: Any = None,
    ) -> tuple[float, bool]:
        nonlocal n_evals
        u, ov, ne = _gg_fn(
            trial, others, others_s, box_, pot, ts, tbl, nn, uc, tree=tree
        )
        n_evals += ne
        return u, ov

    step = 0
    while step < window:
        N = len(pos)
        choice = moves.draw_move(rng, n_species)

        # ----- translate ----- (species-agnostic selection: uniform over all N)
        if choice == "move":
            c["move_step"] += 1
            if N == 0:
                continue
            j = int(rng.integers(N))
            s = int(spc[j])
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            without_s = np.delete(spc, j)
            disp = move_factor * (rng.uniform(0.0, 1.0, dim) - 0.5)
            moved = (old + disp) % side
            if field.overlaps(moved, s):
                continue
            # one tree for both legs of this move; see _guest_guest_energy
            tree_wo = cKDTree(without, boxsize=box) if len(without) else None
            u_gg_new, ov = gg(
                moved,
                without,
                without_s,
                box,
                pots[s],
                s,
                table,
                n_neigh,
                upper_cutoff,
                tree=tree_wo,
            )
            if ov:
                continue
            u_gf_new = field.energy(moved, s)
            u_gg_old, _ = gg(
                old,
                without,
                without_s,
                box,
                pots[s],
                s,
                table,
                n_neigh,
                upper_cutoff,
                tree=tree_wo,
            )
            u_gf_old = field.energy(old, s)
            du = (u_gg_new + u_gf_new) - (u_gg_old + u_gf_old)
            if math.log(rng.uniform(0.0, 1.0)) <= -beta * du:
                pos = np.append(without, [moved], axis=0)
                spc = np.append(without_s, s)
                c["move_accept"] += 1
            continue

        # ----- identity swap (A4.3) ----- resample a chosen guest's species
        if choice == "swap":
            c["swap_step"] += 1
            step += 1
            if N == 0:
                continue
            j = int(rng.integers(N))
            s = int(spc[j])
            s_new = moves.draw_swap_target(rng, s, guest_types)
            point = pos[j]
            without = np.delete(pos, j, axis=0)
            without_s = np.delete(spc, j)
            # the new identity must clear the framework and neighbour floors too
            if field.overlaps(point, s_new):
                continue
            u_gg_new, ov = gg(
                point,
                without,
                without_s,
                box,
                pots[s_new],
                s_new,
                table,
                n_neigh,
                upper_cutoff,
            )
            if ov:
                continue
            u_new = u_gg_new + field.energy(point, s_new)
            u_gg_old, _ = gg(
                point, without, without_s, box, pots[s], s, table, n_neigh, upper_cutoff
            )
            u_old = u_gg_old + field.energy(point, s)
            # A specific labelled particle changes identity, so the ratio is just
            # the activity ratio times the Boltzmann factor. There is NO
            # n_s/(n_s'+1) combinatorial term: that belongs to an aggregate
            # counts move, and adding it here double counts the uniform-particle
            # selection, biasing the mixture toward the minority species (it
            # conserves total N while distorting the ratio). Detailed balance with
            # the symmetric uniform-particle / uniform-other-species proposal is
            # satisfied by z*_s'/z*_s alone (checked analytically for 1 and 2
            # particles, and by the ideal-gas ratio test).
            log_acc = logz[s_new] - logz[s] - beta * (u_new - u_old)
            if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
                spc[j] = s_new
                c["swap_accept"] += 1
            continue

        # ----- insertion ----- pick a species, then a uniform position
        if choice == "add":
            c["add_step"] += 1
            step += 1
            s = pick_species()
            trial = np.array([rng.uniform(0.0, side[a]) for a in range(dim)])
            if field.overlaps(trial, s):
                continue
            u_gg, ov = gg(
                trial, pos, spc, box, pots[s], s, table, n_neigh, upper_cutoff
            )
            if ov:
                continue
            u = u_gg + field.energy(trial, s)
            n_s = int(np.count_nonzero(spc == s))
            log_acc = logz[s] + logV - math.log(n_s + 1) - beta * u
            if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
                pos = np.append(pos, [trial], axis=0) if N > 0 else trial.reshape(1, -1)
                spc = np.append(spc, s)
                c["add_accept"] += 1
            continue

        # ----- deletion ----- pick a species, then a uniform particle of it
        c["remove_step"] += 1
        step += 1
        s = pick_species()
        idx_s = np.nonzero(spc == s)[0]
        n_s = len(idx_s)
        if n_s == 0:
            continue  # nothing of this species to remove; a wasted proposal
        j = int(idx_s[rng.integers(n_s)])
        chosen = pos[j]
        without = np.delete(pos, j, axis=0)
        without_s = np.delete(spc, j)
        u_gg, _ = gg(
            chosen, without, without_s, box, pots[s], s, table, n_neigh, upper_cutoff
        )
        u = u_gg + field.energy(chosen, s)
        log_acc = math.log(n_s) - logz[s] - logV + beta * u
        if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
            pos = without
            spc = without_s
            c["remove_accept"] += 1

    n_by_type = {t: int(np.count_nonzero(spc == t)) for t in guest_types}
    return MofResult(
        n=len(pos),
        positions=pos,
        counters=c,
        loading=len(pos) / n_cells,
        species=spc,
        n_by_type=n_by_type,
        n_evals=n_evals,
    )
