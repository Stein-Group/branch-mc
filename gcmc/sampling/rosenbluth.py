"""Rosenbluth selection.

`select_forward` is the forward (insertion-direction) weight: it grows a chosen
configuration one molecule at a time, each pick drawn with probability set by
the weigher, recording the running normalisation (sum_weight) at every level.
This is the kernel intended for a future C port, so its inputs and outputs are
flat: a TrialSet in, a TrialSet of chosen molecules plus the sum_weight array
out, with the single random draw injected via `rand`.

It reproduces the original `rosenbluth_weight_new` exactly. The RNG draw is a
zero-argument callable returning a uniform in [0, 1); passing np.random.rand
matches the original stream, while the engine will later pass a threaded
generator's draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import FloatArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..proposals.trials import TrialSet

from ..proposals.trials import TrialSet


def select_forward(
    pool: TrialSet,
    n_atoms: int,
    weigher: Any,
    *,
    rand: Callable[[], float],
    side: float | FloatArray,
    r_squared: float,
    gamma_schedule: Mapping[int, float],
) -> tuple[TrialSet, FloatArray] | tuple[None, None]:
    """Returns (chosen TrialSet, sum_weight array) or (None, None) if the pool
    is exhausted before n_atoms are chosen."""
    assert len(pool) > 0, "no non-overlapping trial molecules"
    assert n_atoms != 0, "n_atoms must be > 0"

    sum_arr = np.empty(n_atoms, dtype=float)
    c_coords, c_ew, c_cos, c_sin, c_ct = [], [], [], [], []
    # The chosen atoms' SPECIES and their POOL ROWS travel out with them (M19,
    # M21). Collected alongside the five historical columns rather than recovered
    # afterwards, because this loop deletes and masks the pool between levels: a
    # row number read from the shrunken pool means nothing, and identifying the
    # chosen atoms by coordinate is exactly what M21 forbids.
    c_spc, c_row = [], []

    for i in range(n_atoms):
        sum_weight, probability = weigher.selection(pool)
        sum_arr[i] = sum_weight

        # an all-zero (or non-finite) weight pool has no valid pick: dividing by
        # its sum would make probability nan and searchsorted would return a
        # garbage index. Treat it as an exhausted pool, i.e. a rejected move,
        # rather than propagating nan into the acceptance.
        if not (sum_weight > 0.0) or probability is None:
            return None, None

        cumulative = np.cumsum(probability)
        # rand() can land above cumulative[-1] by rounding; clip to the pool
        chosen_index = min(int(np.searchsorted(cumulative, rand())), len(pool) - 1)
        coord, ew, cos, sin, ct = pool[chosen_index]
        c_coords.append(coord)
        c_ew.append(ew)
        c_cos.append(cos)
        c_sin.append(sin)
        c_ct.append(ct)
        c_spc.append(int(pool.species[chosen_index]))
        c_row.append(int(pool.pool_row[chosen_index]))

        if i == n_atoms - 1:
            break

        pool = pool.delete(chosen_index)
        if len(pool) == 0:
            return None, None

        delta = pool.coords - coord
        delta -= side * np.round(delta / side)
        sq = np.sum(delta**2, axis=1)

        mask = sq >= r_squared
        pool = pool.mask(mask)
        if len(pool) == 0:
            return None, None

        # the schedule shapes the weights level i + 1 selects from
        weigher.recompute(pool, coord, sq[mask], gamma_schedule[min(i + 1, 9)])

    chosen = TrialSet(
        np.array(c_coords),
        np.array(c_ew),
        np.array(c_cos),
        np.array(c_sin),
        np.array(c_ct),
        None,
        np.array(c_spc, dtype=np.int64),
        np.array(c_row, dtype=np.int64),
    )
    return chosen, sum_arr


def select_reverse(
    trial_set: TrialSet | None,
    mol_in_cav: TrialSet,
    n_atoms: int,
    weigher: Any,
    *,
    side: float | FloatArray,
    r_squared: float,
    gamma_schedule: Mapping[int, float],
) -> tuple[FloatArray, FloatArray]:
    """Reverse (deletion-direction) weight, reproducing the original
    `rosenbluth_weight_current`.

    For each molecule being removed it accumulates the level normalisation from
    the still-present cavity molecules plus the surviving trial pool, records
    that molecule's own weight, then refreshes both the trial weights and the
    remaining cavity weights against the just-processed molecule. `mol_in_cav`
    is mutated in place exactly as before. Returns (sum_weight, per-molecule
    weight) arrays. There is no random draw here; the result is deterministic.
    """
    assert len(mol_in_cav) == n_atoms

    if trial_set is None:
        return weigher.reverse_no_trial(mol_in_cav, n_atoms)

    sum_arr = np.empty(n_atoms, dtype=float)
    e_arr = np.empty(n_atoms, dtype=float)

    for i in range(n_atoms):
        base_sum, e_i = weigher.base_terms(mol_in_cav, i)
        sum_arr[i] = base_sum
        e_arr[i] = e_i

        chosen = mol_in_cav[i][0]

        if len(trial_set) != 0:
            sum_arr[i] += weigher.trial_sum(trial_set)
            diff = trial_set.coords - chosen
            diff -= side * np.round(diff / side)
            dist_sq = np.sum(diff**2, axis=1)
            mask_curr = dist_sq >= r_squared
            trial_set = trial_set.mask(mask_curr)

        if i == n_atoms - 1:
            return sum_arr, e_arr

        gamma = gamma_schedule[min(i + 1, 9)]
        if len(trial_set) != 0:
            weigher.recompute_trial(trial_set, chosen, dist_sq[mask_curr], gamma)
        weigher.recompute_cav(mol_in_cav, i, chosen, gamma, side)

    raise AssertionError("unreachable: the loop returns at i == n_atoms - 1")
