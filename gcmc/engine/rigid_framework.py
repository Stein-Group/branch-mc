"""The framework run: its move set and its reporting, in a rigid host.

The move loop itself is NOT here. It is `metropolis_dynamics.metropolis_window`, shared
with the bulk path, because the two were once separate implementations of the
same textbook grand-canonical acceptance and one acceptance criterion with two
homes gets fixed in one of them. What is here is what is genuinely about the
framework:

  advance_metropolis_mof_ms   the ADAPTER: builds a `MetropolisModel` whose
                              overlap test is the framework's, whose pair energy
                              is per-pair typed, and whose one-body term is the
                              host, then hands it to the shared loop
  framework_ladder_move       the bulk engine's ladder window move
  ladder_window_moves_ms      (glauber_dynamics.LadderMove) configured for and
                              applied to a framework: branch or channel body,
                              Suwa-Todo or heat-bath kernel
  mof_energy                  host-guest and guest-guest energy from scratch, the
                              oracle the running energy is checked against
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
from typing import Any

import numpy as np

from .._types import RNG, FloatArray, IntArray
from ..forcefield.lj_table import FrameworkField, LennardJonesTable
from ..forcefield.pair_table import PairTable
from ..space import neighbors
from .glauber_dynamics import LADDER_ENGINES, LadderMove, SlotTable, new_window_stats
from .metropolis_dynamics import MetropolisModel, metropolis_window


@dataclass
class MofResult:
    """What one window of the MOF metropolis loop produced.

    `species` and `n_by_type` are populated only by the multi-species loop; the
    single-species loop leaves them None, so its callers and result shape are
    unchanged (additive, A4.2)."""

    n: int
    positions: FloatArray  # reduced units, (N, dim)
    counters: dict[str, Any]  # the window's log columns (raw counts and sums)
    loading: float  # guests per unit cell, a reporting convenience
    species: IntArray | None = None  # per-particle type id, aligned to positions
    n_by_type: dict[int, int] | None = None  # per-species counts N_s (A4.2)
    n_evals: int = 0  # guest-guest pair distances evaluated (efficiency metric)
    du_guest: float = 0.0  # guest-guest energy change of the accepted moves
    du_host: float = 0.0  # host-guest energy change of the accepted moves


def mof_energy(
    positions: FloatArray,
    species: IntArray,
    *,
    field: FrameworkField,
    table: PairTable,
    beta: float,
    upper_cutoff: float,
) -> tuple[float, float]:
    """(host-guest, guest-guest) energy of a configuration, from scratch.

    The oracle the running energy is checked against at each checkpoint.
    """
    pos = np.ascontiguousarray(positions, float)
    spc = np.ascontiguousarray(species, np.int64).reshape(-1)
    pots = {
        t: LennardJonesTable(
            beta=beta, table=table, trial_type=t, upper_cutoff=upper_cutoff
        )
        for t in {int(t) for t in spc.tolist()}
    }
    box = field.box
    u_host = 0.0
    for t in np.unique(spc).tolist():
        u_host += float(np.sum(field.energy_many(pos[spc == t], int(t))))
    u_guest = 0.0
    for i in range(len(pos) - 1):
        dr = pos[i] - pos[i + 1 :]
        dr -= box * np.round(dr / box)
        r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        live = (r > 0.0) & (r < upper_cutoff)
        if live.any():
            u_guest += float(
                pots[int(spc[i])].total_energy(
                    r[live], neighbour_types=spc[i + 1 :][live]
                )
            )
    return u_host, u_guest


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
    exchange: bool = True,
    move_probs: tuple[float, float, float] | None = None,
    count_every_move: bool = False,
) -> MofResult:
    """One sampling window of multi-species GCMC in the framework.

    `move_probs` / `count_every_move`: a framework params file's stated
    (translation, exchange, identity swap) probabilities, and a window that
    counts every proposal; both default to the historical behaviour.

    `exchange=False` drops the single-particle insert/delete from the window
    (translation and, for a mixture, identity swap remain at their 2:1 ratio),
    for a run whose number changes come from `ladder_window_moves_ms` alone.

    A FRAMEWORK ADAPTER over `metropolis_dynamics.metropolis_window`. What is
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
        exchange=exchange,
        move_probs=move_probs,
        count_every_move=count_every_move,
    )

    counters: dict[str, Any] = {
        "translate_tried": 0,
        "translate_accepted": 0,
        "translate_sum_dr2": 0.0,
    }
    if exchange:
        counters.update(
            insert_tried=0, insert_accepted=0, delete_tried=0, delete_accepted=0
        )
    if n_species > 1:
        counters.update(swap_tried=0, swap_accepted=0)
    energy: dict[str, float] = {}
    pos, out_spc, n_evals = metropolis_window(
        pos, spc, rng, model, window, counters, energy
    )
    assert out_spc is not None
    spc = out_spc

    n_by_type = {t: int(np.count_nonzero(spc == t)) for t in guest_types}
    return MofResult(
        n=len(pos),
        positions=pos,
        counters=counters,
        loading=len(pos) / n_cells,
        species=spc,
        n_by_type=n_by_type,
        n_evals=n_evals,
        du_guest=energy["guest"],
        du_host=energy["host"],
    )


def framework_ladder_move(
    *,
    field: FrameworkField,
    table: PairTable,
    z_star: dict[int, float],
    beta: float,
    box: FloatArray,
    rc: float,
    engine: str,
    sampler: str,
    channel_length: float,
    channel_width: float,
    trial_density: float,
    M: int,
    kmax: int,
    kmax_margin: int = 0,
    species_probs: dict[int, float] | None = None,
) -> LadderMove:
    """The bulk engine's ladder window move, configured for a rigid host.

    NOT A SECOND IMPLEMENTATION. `glauber_dynamics.LadderMove` is the one window
    move -- body, pool, one-body weights, ladder, kernel, commit -- and this only
    states the framework's state point to it: the host `field` (which enters the
    one-body weight and nowhere else), the per-pair table, the guests' activities
    and draw probabilities. `sampler` is "channel" (the rod, transfer matrix) or
    "branch" (the cube cavity, clique algebra); `engine` is "suwa_todo_dynamics"
    or "glauber_dynamics". M and kmax come from `sim_setup.ladder_window`, the
    same derivation and the same refusals as a bulk params file.

    Always typed, even for one guest: guest type ids are the force field's, so a
    pure-Kr run is type 1, and the untyped path's all-zero species column would
    silently make it Xe.
    """
    if engine not in LADDER_ENGINES:
        raise ValueError(f"engine={engine!r} is not a ladder dynamics {LADDER_ENGINES}")
    if sampler not in ("channel", "branch"):
        raise ValueError(f"sampler={sampler!r}: the ladder needs 'channel' or 'branch'")
    guest_ids = sorted(int(t) for t in z_star)
    probs = np.array(
        [
            1.0 / len(guest_ids) if species_probs is None else species_probs[t]
            for t in guest_ids
        ],
        dtype=float,
    )
    n_t = 1 + max(guest_ids)
    log_z_arr = np.zeros(n_t)
    log_p_arr = np.zeros(n_t)
    for t, p in zip(guest_ids, probs):
        log_z_arr[t] = math.log(z_star[t])
        log_p_arr[t] = math.log(p)
    box = np.asarray(box, float)
    floor = table.core_floor
    return LadderMove(
        dim=int(box.shape[0]),
        side=box,
        box=box,
        beta=float(beta),
        uc=float(rc),
        lower=float(np.min(floor[np.ix_(guest_ids, guest_ids)])),
        lj=True,
        M=int(M),
        kmax=int(kmax),
        full_span=False,
        w_chan=float(channel_width),
        sigma_chan=float(max(table.sigma[t, t] for t in guest_ids)),
        channel_length=float(channel_length),
        channel_width=float(channel_width),
        trial_density=float(trial_density),
        kmax_margin=int(kmax_margin),
        cavity=sampler == "branch",
        typed=True,
        guest_ids=guest_ids,
        p_trial_guest=probs,
        log_z={t: math.log(z_star[t]) for t in guest_ids},
        log_z_arr=log_z_arr,
        log_p_arr=log_p_arr,
        floor=floor,
        table=table,
        ext_radius=float(rc),
        fixed_n=False,
        tail=None,  # pore convention: no tail, on either side
        suwa_todo=engine == "suwa_todo_dynamics",
        field=field,
    )


def ladder_window_moves_ms(
    move: LadderMove,
    positions: FloatArray,
    species: IntArray,
    rng: RNG,
    n_moves: int,
    stats: dict[str, float] | None = None,
) -> tuple[FloatArray, IntArray]:
    """Apply `n_moves` ladder window moves to a framework configuration.

    The guests go into one cell list for the batch, exactly as the bulk engine
    holds them for a sampling window; the framework stays outside it, as a field.
    `stats` accumulates `glauber_dynamics.new_window_stats` sums (max for
    `max_k`).
    """
    pos = np.ascontiguousarray(positions, float)
    spc = np.ascontiguousarray(species, np.int64).reshape(-1)
    cl = neighbors.CellList.from_positions(
        pos, move.box, move.dim, move.uc, species=spc
    )
    slots = SlotTable(cl, first=0, n=len(pos))
    w = new_window_stats()
    for _ in range(int(n_moves)):
        move.step(cl, slots, rng, w)
    n = slots.N
    out_pos = np.array(cl.coords[slots.alive[:n]], dtype=float)
    out_spc = np.array(cl.species[slots.alive[:n]], dtype=np.int64)
    if stats is not None:
        for k, v in w.items():
            stats[k] = (
                max(stats.get(k, 0.0), v) if k == "max_k" else stats.get(k, 0.0) + v
            )
    return out_pos, out_spc
