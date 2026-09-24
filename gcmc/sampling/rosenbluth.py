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

ADAPTIVE GAMMA. Both passes take an optional `target`. With `target = None` they
read the fitted `gamma_schedule` and every line below is the frozen path. With a
`ParticipationTarget` they solve the exponent per level from the weights in hand
(`sampling.participation`), and three things about that are load-bearing:

  * THE LEVEL COUNT IS NOT CAPPED. The schedule is nine numbers and the frozen
    path reads `gamma_schedule[min(i + 1, 9)]`, so a move inserting ten or more
    molecules silently reuses level 9's exponent. There is no ninth number to
    run out of here, so every level gets its own solve. (`n_levels` in this
    module means the number of Rosenbluth levels, which is `n_atoms`. It is
    never called L; L in this project is the trial box side.)
  * EACH PASS SOLVES ITS OWN LEVEL-1 GAMMA, here rather than at pool-build
    time. The forward level-1 pool is the permuted merge of surviving trials
    and tombstoned members; the reverse level-1 pool is the removed members
    plus whatever trial cloud survived the overlap mask. Those are different
    sets, so gamma_1 differs between the passes by construction.
  * THE SOLVE IS COLD EVERY TIME. Nothing is cached per level, per move or
    across the passes. A warm start would make the kernel a function of chain
    history and forfeit exact stationarity, while still producing plausible
    numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import FloatArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..proposals.trials import TrialSet
    from .participation import ParticipationTarget

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
    target: ParticipationTarget | None = None,
    solved_gammas: list[float] | None = None,
) -> tuple[TrialSet, FloatArray] | tuple[None, None]:
    """Returns (chosen TrialSet, sum_weight array) or (None, None) if the pool
    is exhausted before n_atoms are chosen.

    `target` selects the exponent policy: None reads `gamma_schedule`, a
    `ParticipationTarget` solves per level. `solved_gammas`, if given, has each
    solved exponent appended in level order; it is an out-parameter rather than
    a third return value because the return tuple is unpacked positionally at
    both call sites and by the reverse-reproducibility script.
    """
    assert len(pool) > 0, "no non-overlapping trial molecules"
    assert n_atoms != 0, "n_atoms must be > 0"

    def _record(g: float) -> None:
        if solved_gammas is not None:
            solved_gammas.append(g)

    if target is not None:
        # LEVEL 1. `_weighted_trials` handed over the bias unexponentiated in
        # `pool.log_bias` precisely so this solve could happen here, after the
        # forward pool was assembled and permuted. Doing it at pool-build time
        # would bake one gamma_1 into a pool the reverse pass also reads, and
        # the reverse pass has a different level-1 set.
        g1 = target.solve(pool.energy_weight, pool.log_bias)
        weigher.apply_increment(pool, np.exp(g1 * pool.log_bias))
        _record(g1)

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

        if target is None:
            # The schedule is consumed HERE and not at the top of the loop: it
            # shapes the weights level i + 1 selects from, which is why the
            # adaptive solve below sits at the same point.
            weigher.recompute(pool, coord, sq[mask], gamma_schedule[min(i + 1, 9)])
        else:
            # Solve against the pool level i + 1 will actually select from:
            # the survivors' current weights times the gamma-free part of the
            # new increment, with the increment's bias as the tunable part.
            # No cap on i: every level gets its own solve.
            base_inc, log_inc = weigher.recompute_split(pool, coord, sq[mask])
            g = target.solve(pool.energy_weight * base_inc, log_inc)
            weigher.apply_increment(pool, base_inc * np.exp(g * log_inc))
            _record(g)

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
    target: ParticipationTarget | None = None,
    solved_gammas: list[float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Reverse (deletion-direction) weight, reproducing the original
    `rosenbluth_weight_current`.

    For each molecule being removed it accumulates the level normalisation from
    the still-present cavity molecules plus the surviving trial pool, records
    that molecule's own weight, then refreshes both the trial weights and the
    remaining cavity weights against the just-processed molecule. `mol_in_cav`
    is mutated in place exactly as before. Returns (sum_weight, per-molecule
    weight) arrays. There is no random draw here; the result is deterministic.

    `target` and `solved_gammas` are as in `select_forward`. THE GAMMAS SOLVED
    HERE ARE NOT THE FORWARD ONES, at any level, and that is required rather
    than tolerated: a reverse level's pool is the members not yet re-selected
    plus the surviving trial cloud, which is a different set from the forward
    pool at the same level. Equality of the two would mean a gamma had leaked
    between the passes, which is what gate G3 looks for.
    """
    assert len(mol_in_cav) == n_atoms

    def _record(g: float) -> None:
        if solved_gammas is not None:
            solved_gammas.append(g)

    if target is not None:
        # LEVEL 1, over this pass's own level-1 pool: the removed members plus
        # whatever trial cloud survived. One gamma over the union, applied to
        # both halves, because the union IS the set level 1 normalises over.
        _cav_log = mol_in_cav.log_bias
        if trial_set is None:
            base_1, log_1 = mol_in_cav.energy_weight, _cav_log
        else:
            base_1 = np.concatenate([mol_in_cav.energy_weight, trial_set.energy_weight])
            log_1 = np.concatenate([_cav_log, trial_set.log_bias])
        g1 = target.solve(base_1, log_1)
        weigher.apply_increment(mol_in_cav, np.exp(g1 * _cav_log))
        if trial_set is not None:
            weigher.apply_increment(trial_set, np.exp(g1 * trial_set.log_bias))
        _record(g1)

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

        if target is None:
            gamma = gamma_schedule[min(i + 1, 9)]
            if len(trial_set) != 0:
                weigher.recompute_trial(trial_set, chosen, dist_sq[mask_curr], gamma)
            weigher.recompute_cav(mol_in_cav, i, chosen, gamma, side)
        else:
            # ONE GAMMA OVER THE UNION. Level i + 1's pool is the members from
            # row i + 1 on together with the surviving trials, and those two
            # halves get their increments from different geometry
            # (`recompute_cav_split` minimum-images member-member separations,
            # `recompute_trial_split` is handed the trial distances). Solving
            # them separately would put two different exponents on one level,
            # which is not a variant of the method but a different and wrong
            # one, and it is the failure only gate G6 catches.
            cav_base, cav_log = weigher.recompute_cav_split(mol_in_cav, i, chosen, side)
            cav_w = mol_in_cav.energy_weight[i + 1 :] * cav_base
            if len(trial_set) != 0:
                tr_base, tr_log = weigher.recompute_trial_split(
                    trial_set, chosen, dist_sq[mask_curr]
                )
                union_base = np.concatenate([cav_w, trial_set.energy_weight * tr_base])
                union_log = np.concatenate([cav_log, tr_log])
            else:
                tr_base = tr_log = None
                union_base, union_log = cav_w, cav_log
            g = target.solve(union_base, union_log)
            if tr_base is not None and tr_log is not None:
                weigher.apply_increment(trial_set, tr_base * np.exp(g * tr_log))
            weigher.apply_increment_cav(mol_in_cav, i, cav_base * np.exp(g * cav_log))
            _record(g)

    raise AssertionError("unreachable: the loop returns at i == n_atoms - 1")
