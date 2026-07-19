"""The MOF metropolis loop: single-species GCMC in a rigid framework.

This mirrors `_advance_metropolis` in MC_engine.py move for move -- the same
textbook grand-canonical acceptance, the same 0.5/0.25/0.25 move split, the same
"null proposal at N=0" handling -- with exactly three differences, all forced by
the framework:

  1. every insertion and every move adds the guest-framework energy
     (FrameworkField, the A3.5-gated core) to the guest-guest energy;
  2. overlap is checked against the framework too, per pair type (A3.3);
  3. there is no long-range tail (pore convention), so the tail term is simply
     absent rather than set to something.

It is a SEPARATE function, not a branch inside _advance_metropolis, so that
frozen path stays byte-identical for every existing mode. Everything here runs in
the engine's reduced units: the caller reduces positions, box, cutoff and
activity by the Xe scale before calling, exactly as the reduction pipeline
(P0.4) defines. Single species only -- the guest type is fixed -- because
multi-species is Phase 4; the guest_type argument is here so Phase 4 can lift it
without reshaping the loop.

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

from ._types import FloatArray
from .mof_field import FrameworkField
from .potentials.lj_table import LennardJonesTable


@dataclass
class MofResult:
    """What one window of the MOF metropolis loop produced."""

    n: int
    positions: FloatArray  # reduced units, (N, dim)
    counters: dict[str, int]
    loading: float  # guests per unit cell, a reporting convenience


def _guest_guest_energy(
    trial: FloatArray,
    others: FloatArray,
    box: FloatArray,
    pot: LennardJonesTable,
    guest_type: int,
    nn: int,
    uc: float,
) -> tuple[float, bool]:
    """Guest-guest energy of `trial` against the fluid, and whether it overlaps.

    Single species, so every neighbour is `guest_type`. Returns (energy, overlap);
    when overlap is True the energy is meaningless and the caller rejects.
    """
    if len(others) == 0:
        return 0.0, False
    tree = cKDTree(others, boxsize=box)
    # saturating query so a summed energy never drops an in-range pair
    k = min(nn, len(others))
    d, _ = tree.query(trial, k=k, distance_upper_bound=uc)
    d = np.atleast_1d(d)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, False
    floor = pot.diameter  # c*sigma for this guest type against itself
    if (d <= floor - 1e-12).any():
        return 0.0, True
    u = pot.total_energy(d, neighbour_types=np.full(d.size, guest_type))
    return u, False


def advance_metropolis_mof(
    positions: FloatArray,
    rng: np.random.Generator,
    *,
    field: FrameworkField,
    table: Any,
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

    `z_star` is the reduced activity z* = f sigma_ref^3 / (kB T) (P0.4), so the
    insertion acceptance is z* V / (N+1) * exp(-beta dU) with dU the FULL energy
    change including the framework. Deletion is the exact reverse. `volume` and
    `box` are reduced; `n_cells` is only for the loading report.
    """
    pos = np.ascontiguousarray(positions, float)
    dim = pos.shape[1] if pos.ndim == 2 and len(pos) else field.pos.shape[1]
    pot = LennardJonesTable(
        beta=beta, table=table, trial_type=guest_type, upper_cutoff=upper_cutoff
    )
    c = {
        "move_step": 0,
        "move_accept": 0,
        "add_step": 0,
        "add_accept": 0,
        "remove_step": 0,
        "remove_accept": 0,
    }
    box = np.asarray(box, float)
    side = box  # per-axis; a scalar box would broadcast, but MOF cells are per-axis
    logV = math.log(volume)

    step = 0
    while step < window:
        N = len(pos)
        choice = rng.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])

        # ----- translate -----
        if choice == "move":
            c["move_step"] += 1
            if N == 0:
                continue
            j = rng.integers(N)
            old = pos[j]
            without = np.delete(pos, j, axis=0)
            disp = move_factor * (rng.uniform(0.0, 1.0, dim) - 0.5)
            moved = (old + disp) % side
            if field.overlaps(moved, guest_type):
                continue  # into a framework atom: rejected
            u_gg_new, ov = _guest_guest_energy(
                moved, without, box, pot, guest_type, n_neigh, upper_cutoff
            )
            if ov:
                continue
            u_gf_new = field.energy(moved, guest_type)
            u_gg_old, _ = _guest_guest_energy(
                old, without, box, pot, guest_type, n_neigh, upper_cutoff
            )
            u_gf_old = field.energy(old, guest_type)
            du = (u_gg_new + u_gf_new) - (u_gg_old + u_gf_old)
            if math.log(rng.uniform(0.0, 1.0)) <= -beta * du:
                pos = np.append(without, [moved], axis=0)
                c["move_accept"] += 1
            continue

        # ----- insertion -----
        if choice == "add":
            c["add_step"] += 1
            step += 1
            trial = np.array([rng.uniform(0.0, side[a]) for a in range(dim)])
            if field.overlaps(trial, guest_type):
                continue
            u_gg, ov = _guest_guest_energy(
                trial, pos, box, pot, guest_type, n_neigh, upper_cutoff
            )
            if ov:
                continue
            u = u_gg + field.energy(trial, guest_type)
            # z* V / (N+1) exp(-beta U); log form, z* folds mu and the de Broglie
            # volume the reduced activity already carries
            log_acc = math.log(z_star) + logV - math.log(N + 1) - beta * u
            if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
                pos = np.append(pos, [trial], axis=0) if N > 0 else trial.reshape(1, -1)
                c["add_accept"] += 1
            continue

        # ----- deletion -----
        c["remove_step"] += 1
        step += 1
        if N == 0:
            continue
        j = rng.integers(N)
        chosen = pos[j]
        without = np.delete(pos, j, axis=0)
        u_gg, _ = _guest_guest_energy(
            chosen, without, box, pot, guest_type, n_neigh, upper_cutoff
        )
        u = u_gg + field.energy(chosen, guest_type)
        # reverse of insertion: N / (z* V) exp(+beta U)
        log_acc = math.log(N) - math.log(z_star) - logV + beta * u
        if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
            pos = without
            c["remove_accept"] += 1

    return MofResult(n=len(pos), positions=pos, counters=c, loading=len(pos) / n_cells)
