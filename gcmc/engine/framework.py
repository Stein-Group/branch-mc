"""The framework run: its move set and its reporting, in a rigid host.

The move loop itself is NOT here. It is `MC_engine.metropolis_window`, shared
with the bulk path, because the two were once separate implementations of the
same textbook grand-canonical acceptance and one acceptance criterion with two
homes gets fixed in one of them. What is here is what is genuinely about the
framework:

  advance_metropolis_mof_ms   the ADAPTER: builds a `MetropolisModel` whose
                              overlap test is the framework's, whose pair energy
                              is per-pair typed, and whose one-body term is the
                              host, then hands it to the shared loop
  advance_metropolis_mof      the single-species convenience wrapper
  channel_repack_move_ms      the whole-window heat-bath repack and the
                              two energy helpers that serve only it
  mof_energy_decomposition    the per-term energy report the isotherm CSV streams
  MofResult                   what a framework window returns

Everything here runs in the engine's reduced units: the caller reduces positions,
box, cutoff and activity by the reference scale before calling, exactly as the
reduction pipeline defines. There is no long-range tail (the pore convention is
no-tail on both sides), so the term is absent rather than set to something.

Correctness is not asserted by this module; it is asserted by the move-delta
parity test against ref_energy.py, which recomputes U(after) - U(before)
from scratch in angstrom/kelvin and requires agreement to 1e-10, and by a golden
fixed-seed trajectory (`tests/unit/test_mof_metropolis_ms.py`) that any change
to the shared loop has to reproduce bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._types import RNG, FloatArray, IntArray
from ..forcefield.lj_table import FrameworkField, LennardJonesTable
from ..forcefield.overlap import OVERLAP_TOL as _TOL
from ..forcefield.pair_table import PairTable, table_pair_energy_sq
from ..proposals import moves
from ..proposals.channel_move import assign_species, window_guard_ok
from ..sampling.channel import ChannelSweepLJ
from ..space.body import cuboid_members, local_from_box, oriented_cuboid_trials
from ..system.state import zero_counters
from .MC_engine import MetropolisModel, metropolis_window


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


# The move split now lives in gcmc/proposals/moves.py, so the main engine and this
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
    it also doubles as the drift oracle for an incremental-U tracker."""
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
    """One sampling window of multi-species GCMC in the framework.

    A FRAMEWORK ADAPTER over `MC_engine.metropolis_window`. What is
    framework-specific is three answers a `MetropolisModel` gives -- a framework
    overlap, a per-pair typed energy, a one-body host term -- and the move set
    and the acceptance algebra are shared with every other single-particle path.

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
    dim = pos.shape[1] if pos.ndim == 2 and len(pos) else field.pos.shape[1]

    # one potential per guest type; construction is a few refs, the hot path just
    # indexes this by the trial species
    pots = {
        t: LennardJonesTable(
            beta=beta, table=table, trial_type=t, upper_cutoff=upper_cutoff
        )
        for t in guest_types
    }
    model = MetropolisModel(
        beta=beta,
        box=np.asarray(box, float),
        volume=volume,
        upper_cutoff=upper_cutoff,
        # unused on the typed path: the floor is per pair, out of table.core_floor
        lower_cutoff=0.0,
        n_neigh=n_neigh,
        dim=dim,
        move_factor=move_factor,
        log_z={t: math.log(z_star[t]) for t in guest_types},
        species_probs=tuple(species_probs[t] for t in guest_types),
        species_ids=tuple(guest_types),
        hard_core=False,
        # no tail in a pore, on either side: the species file forbids it too
        tail_pref=None,
        field=field,
        table=table,
        pots=pots,
    )

    counters = zero_counters()
    pos, out_spc, n_evals = metropolis_window(pos, spc, rng, model, window, counters)
    assert out_spc is not None
    spc = out_spc

    # The eight counters this result has always carried. `zero_counters` holds a
    # superset (the cavity samplers' repack keys, and `add_attempt`); projecting
    # here rather than teaching the shared loop two counter schemas is what keeps
    # the golden pin's dict exactly the shape it was.
    c = {
        k: counters[k]
        for k in (
            "move_step",
            "move_accept",
            "add_step",
            "add_accept",
            "remove_step",
            "remove_accept",
            "swap_step",
            "swap_accept",
        )
    }
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


def _exterior_logweights(
    pool_P: FloatArray,
    pool_s: IntArray,
    ext_P: FloatArray,
    ext_s: IntArray,
    field: FrameworkField,
    table: PairTable,
    box: FloatArray,
    beta: float,
    rc: float,
    z_star: dict[int, float],
) -> tuple[FloatArray, IntArray]:
    """One-body log-weight of each pool member: log z*_s - beta*(u_framework +
    u_exterior), where u_exterior is the LJ interaction with the FROZEN guests
    outside the window (within rc, minimum image). A member overlapping the
    framework or any exterior guest for its species is inadmissible."""
    n = len(pool_P)
    logw = np.full(n, -np.inf)
    adm = np.zeros(n, dtype=bool)
    floor = table.core_floor
    for i in range(n):
        s = int(pool_s[i])
        p = pool_P[i]
        if field.overlaps(p, s):
            continue
        u = field.energy(p, s)
        if len(ext_P):
            dr = ext_P - p
            dr -= box * np.round(dr / box)
            d2 = np.einsum("ij,ij->i", dr, dr)
            fl = floor[s, ext_s]
            if (d2 <= (fl - _TOL) ** 2).any():
                continue  # overlaps a frozen exterior guest
            within = d2 < rc * rc
            if within.any():
                for sj in np.unique(ext_s[within]):
                    g = within & (ext_s == sj)
                    u += float(np.sum(table_pair_energy_sq(d2[g], s, int(sj), table)))
        logw[i] = np.log(z_star[s]) - beta * u
        adm[i] = True
    return logw, adm


def _guest_guest_split(
    P: FloatArray,
    sp: IntArray,
    ext_P: FloatArray,
    ext_s: IntArray,
    table: PairTable,
    box: FloatArray,
    rc_short: float,
    rc_full: float,
) -> tuple[float, float]:
    """(U_short, U_full) guest-guest energy of the group `P`: against the frozen
    exterior guests plus pairs inside the group, at two radii from one distance
    computation.

    THE FRAMEWORK TERM IS ABSENT ON PURPOSE, and that is the whole design decision
    for the MOF half. The dual-cutoff correction is a difference between the
    proposal potential and the target potential; the framework field is evaluated
    EXACTLY in both, so it cancels identically and must not appear here. Shortening
    it would have bought nothing anyway -- it enters the sweep as a one-body scalar
    per trial, so a shorter framework cutoff does not shrink the DP at all, while
    it would add correction variance for free.

    Type-dependent throughout: every pair uses table_pair_energy_sq for its own
    (species_i, species_j), so a mixture is handled by construction rather than by
    an effective single-component sigma.
    """
    box = np.asarray(box, float)
    us = uf = 0.0
    r2s, r2f = rc_short * rc_short, rc_full * rc_full
    if len(ext_P):
        for i in range(len(P)):
            dr = ext_P - P[i]
            dr -= box * np.round(dr / box)
            d2 = np.einsum("ij,ij->i", dr, dr)
            si = int(sp[i])
            for sj in np.unique(ext_s):
                g = ext_s == sj
                d2g = d2[g]
                for r2, acc in ((r2f, "f"), (r2s, "s")):
                    m = d2g < r2
                    if m.any():
                        e = float(
                            np.sum(table_pair_energy_sq(d2g[m], si, int(sj), table))
                        )
                        if acc == "f":
                            uf += e
                        else:
                            us += e
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            dr = P[i] - P[j]
            dr -= box * np.round(dr / box)
            d2 = float(dr @ dr)
            si, sj = int(sp[i]), int(sp[j])
            if d2 < r2f:
                uf += float(table_pair_energy_sq(np.array([d2]), si, sj, table)[0])
            if d2 < r2s:
                us += float(table_pair_energy_sq(np.array([d2]), si, sj, table)[0])
    return us, uf


def channel_repack_move_ms(
    pos: FloatArray,
    spc: IntArray,
    rng: RNG,
    *,
    field: FrameworkField,
    table: PairTable,
    z_star: dict[int, float],
    beta: float,
    box: FloatArray,
    rc: float,
    guest_types: list[int],
    L_long: float,
    w: float,
    n_trials: int,
    species_probs: FloatArray | None = None,
    rc_aux: float | None = None,
    stats: dict[str, float] | None = None,
) -> tuple[FloatArray, IntArray, bool]:
    """One whole-window heat-bath repack (A5.6 wiring).

    Place a random oriented cuboid, take its current occupants as the block to
    resample (count k), build a pool of those occupants plus n_trials fresh
    candidates (each carrying its one-body weight = z* * exp(-beta*(framework +
    frozen-exterior))), run the multi-species LJ sweep, and draw a fresh k-clique
    in proportion to its weight. Because the pool contains the current occupants
    and the block count k is conserved, this is a heat-bath move: forward and
    reverse share the pool and the k_trials factors cancel, so it is accepted
    with probability 1 (pore convention: no tail). It resamples the window's
    positions AND its composition at fixed local count; conventional insert /
    delete / swap moves carry the number changes.

    Returns (pos, spc, moved). `moved` is False when the window is empty or the
    guard fails (nothing to do), leaving the state untouched.
    """
    box = np.asarray(box, float)
    gt = list(guest_types)
    if species_probs is None:
        species_probs = np.full(len(gt), 1.0 / len(gt))
    if stats is not None:
        stats["attempts"] = stats.get("attempts", 0.0) + 1.0
    cutoff = float(rc)
    if not window_guard_ok(L_long, w, box, cutoff):
        return pos, spc, False

    center = rng.uniform(0.0, 1.0, size=3) * box
    # a fixed orientation for this window (per-axis box: non-cubic safe)
    _, R = oriented_cuboid_trials(rng, center, 1, L_long, w, box)
    half_long, half_w = 0.5 * L_long, 0.5 * w
    inside = cuboid_members(pos, center, R, half_long, half_w, box)
    k = len(inside)
    if stats is not None:
        # k is the whole ballgame for V5: the DP only beats single-particle moves
        # when one window holds several guests. k == 1 means the sweep is doing
        # single-particle work while paying transfer-matrix overhead.
        stats["k_sum"] = stats.get("k_sum", 0.0) + float(k)
        stats["k_max"] = max(stats.get("k_max", 0.0), float(k))
    if k == 0:
        return pos, spc, False

    # fresh trial cloud in the same cuboid, species assigned
    a, b = half_long, half_w
    Uf = np.column_stack(
        [
            rng.uniform(-a, a, n_trials),
            rng.uniform(-b, b, n_trials),
            rng.uniform(-b, b, n_trials),
        ]
    )
    Pf = np.mod(center + Uf @ R.T, box)
    sf = assign_species(n_trials, gt, species_probs, rng)

    occ_P = pos[inside]
    occ_s = spc[inside].astype(np.int64)
    outside = np.ones(len(pos), dtype=bool)
    outside[inside] = False
    ext_P = pos[outside]
    ext_s = spc[outside].astype(np.int64)

    pool_P = np.vstack([occ_P, Pf])
    pool_s = np.concatenate([occ_s, sf])
    # PROPOSAL radius. With no auxiliary this is `rc` and every expression below
    # is what it was, heat-bath and all. With one, the guest-guest terms -- the
    # exterior one-body sum here and the intra-pool window in the sweep -- are
    # evaluated at the shorter radius, and the mismatch is divided back out in an
    # acceptance test that this move did not previously need.
    rc_sel = rc if rc_aux is None else float(rc_aux)
    logw, adm = _exterior_logweights(
        pool_P, pool_s, ext_P, ext_s, field, table, box, beta, rc_sel, z_star
    )
    # Rosenbluth correction: a fresh trial's species is drawn with probability
    # p_s, so its Boltzmann weight is divided by p_s to unbias the generation.
    # For the shipped uniform default this is a global constant over a fixed-k
    # clique and cancels; for non-uniform species_probs it keeps the composition
    # exactly balanced.
    p_map = {int(t): float(species_probs[i]) for i, t in enumerate(gt)}
    logp = np.array([np.log(p_map[int(s)]) for s in pool_s])
    logw = logw - logp
    member_weights = np.where(adm, np.exp(np.clip(logw, -700, 700)), 0.0)
    U = local_from_box(pool_P, center, R, box)

    sweep = ChannelSweepLJ(
        U,
        member_weights,
        float("nan"),
        rc_sel,
        beta,
        k,
        species=pool_s,
        floor=table.core_floor,
        table=table,
    )
    if not np.isfinite(sweep.log_weight(k)):
        return pos, spc, False
    idx = sweep.sample(k, rng)
    if idx is None:
        return pos, spc, False

    new_P = pool_P[list(idx)]
    new_s = pool_s[list(idx)]

    # ACCEPTANCE. Without an auxiliary this move is a heat bath and is accepted
    # unconditionally: the pool contains the current occupants, k is conserved, so
    # the Rosenbluth sums and the k_trials factors cancel between the forward and
    # reverse legs and nothing survives.
    #
    # A shorter PROPOSAL radius breaks exactly that cancellation, and only that
    # one. What survives is the same term as in the bulk engine:
    #
    #   Delta_corr = beta * [ (U_new^rc - U_new^aux) - (U_old^rc - U_old^aux) ]
    #
    # over guest-guest energy only -- group against the frozen exterior plus pairs
    # within the group. The framework field is evaluated exactly at both radii and
    # cancels identically, which is why it is absent from _guest_guest_split.
    #
    # So the move stops being a heat bath the moment rc_aux is set. That is a real
    # cost, not a technicality: it needs a reject path, and this one restores
    # nothing because `pos`/`spc` are only rebuilt on the accept branch.
    #
    # THE CONDITION IS "AN AUXILIARY IS CONFIGURED", not "it is shorter than rc".
    # It read `rc_aux is not None and rc_sel < rc`, which made the no-op control
    # arm -- `sweep_rc_aux == rc`, the spec `test_run_mof_stream_integrity` calls
    # `noop` -- skip the correction entirely. Same defect as the bulk engine's
    # `aux_on = ljbranch and rc_sel < uc` and the typed `aux: {control: true}`,
    # in a third place: the arm whose whole job is to check where the correction
    # is inserted never reached it, and passed.
    #
    # At rc_sel == rc the two splits are the same sum over the same pairs, so
    # `delta_corr` is exactly 0.0, `delta_corr > 0.0` is False, no RNG value is
    # drawn and nothing is rejected. The chain is therefore bit-identical to the
    # auxiliary-off chain -- which is now a measured claim rather than a
    # structural one, and the reject path below is what it is measuring.
    if rc_aux is not None:
        us_new, uf_new = _guest_guest_split(
            new_P, new_s, ext_P, ext_s, table, box, rc_sel, rc
        )
        us_old, uf_old = _guest_guest_split(
            occ_P, occ_s, ext_P, ext_s, table, box, rc_sel, rc
        )
        delta_corr = beta * ((uf_new - us_new) - (uf_old - us_old))
        if stats is not None:
            # running moments of the auxiliary correction: sd(Delta_corr) is THE
            # diagnostic for a proposal radius that is too short (the README's
            # aux_corr_sd, which the MOF path did not report). Kept as sums so it
            # survives an arbitrary walltime truncation like every other counter.
            stats["corr_n"] = stats.get("corr_n", 0.0) + 1.0
            stats["corr_sum"] = stats.get("corr_sum", 0.0) + float(delta_corr)
            stats["corr_sumsq"] = stats.get("corr_sumsq", 0.0) + float(delta_corr) ** 2
        if delta_corr > 0.0 and np.log(rng.random()) >= -delta_corr:
            return pos, spc, False

    pos = np.vstack([ext_P, new_P])
    spc = np.concatenate([ext_s, new_s])
    if stats is not None:
        stats["accepts"] = stats.get("accepts", 0.0) + 1.0
    return pos, spc, True
