"""M8 of multispecies_plan.md: which move to propose, and what to swap into.

Three loops need this decision -- `metropolis_dynamics.advance`,
`metropolis_dynamics._advance_single_particle`
and the framework adapter -- and it is exactly the kind of small numeric choice that
gets copied rather than shared, then drifts. The probabilities and the destination
draw live here once.

WHY THE PROBABILITIES ARE WHAT THEY ARE. The multi-species vector is the
single-species one scaled uniformly by (1 - p_swap):

    0.5, 0.25, 0.25   x 0.8  ->  0.4, 0.2, 0.2   plus  swap 0.2

So EVERY ratio among the non-swap moves is preserved, not just add against
remove. That is the property the acceptance derivations need: the
grand-canonical ratio assumes P(add) == P(remove), and the translate share
relative to them is what sets how much configurational relaxation happens
between exchanges, so a split that shifted it would change the sampler's
character as well as its algebra.

Note this corrects the comment that stood in `mof_metropolis`, which said the
swap was "carved out of the translate budget". It is not -- add and remove pay
0.05 each and translation pays 0.1. The equal-probability invariant survives
because all three shrink by the same FACTOR, which is a stronger and simpler
statement than the one about translation.

WHY THE GUARD IS STRUCTURAL. At one species there is nothing to swap into, so the
single-species branch takes the old three-way call with the old list and the old
probability vector. That is not an optimisation -- `rng.choice` consumes different
amounts of the stream for different argument shapes, so a four-way call with
p=[0.4, 0.2, 0.2, 0.0] would be a DIFFERENT CHAIN even though it can never return
"swap". Every single-species trajectory in `tests/ref` and in the P4 snapshot
depends on the guard taking the old path exactly.

WHY THE DESTINATION DRAW IS UNIFORM. Uniform over the other n-1 species is a
symmetric proposal, so the acceptance ratio is z_s'/z_s * exp(-beta dU) with no
proposal correction, for any n. Drawing from `p_trial` instead would need a
p_s/p_s' factor; we are not doing that, and M8 asks for the uniform draw to be
asserted in the test suite so nobody optimises it into a silent bias later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._types import RNG, BoolArray, IntArray

# The frozen single-species split. Kept as module constants so the three call
# sites cannot disagree, and so a reader can see that the multi-species vector
# below preserves P(add) == P(remove).
SINGLE_MOVES = ["move", "add", "remove"]
SINGLE_PROBS = [0.5, 0.25, 0.25]

MS_MOVES = ["move", "add", "remove", "swap"]
MS_PROBS = [0.4, 0.2, 0.2, 0.2]


def draw_move(rng: RNG, n_species: int) -> str:
    """The move type for one iteration.

    At `n_species == 1` this is byte-for-byte the call the engine has always
    made, so the stream is untouched. See the module note on why a four-way call
    with a zero swap probability would not be.
    """
    if n_species == 1:
        return str(rng.choice(SINGLE_MOVES, p=SINGLE_PROBS))
    return str(rng.choice(MS_MOVES, p=MS_PROBS))


#: How a CANONICAL fixed-composition window splits its budget. Two moves only,
#: and both conserve every species count exactly: a translation, which relaxes
#: the structure, and a pair identity exchange, which is what actually demixes --
#: it lets an A buried in a B-rich region trade labels with a B far away, at no
#: cost in particle transport.
CANONICAL_MOVES = ["move", "pair_swap"]
CANONICAL_PROBS = [0.5, 0.5]


def draw_canonical_move(rng: RNG) -> str:
    """The move type for one iteration of a fixed-composition window.

    Separate from `draw_move` and not a fourth branch of it: this ensemble
    proposes no insertion, no deletion and no single-particle identity change,
    so folding it in would mean a probability vector with three zeros in it --
    and `rng.choice` consumes different amounts of the stream for different
    argument shapes, which is the reason `draw_move` already keeps the
    single-species split on its own path.
    """
    return str(rng.choice(CANONICAL_MOVES, p=CANONICAL_PROBS))


def draw_swap_target(rng: RNG, current: int, species_ids: Sequence[int]) -> int:
    """A new species for a swap, uniform over the members of `species_ids` other
    than `current`.

    Takes the actual ids rather than a count. They are contiguous 0..T-1 in the
    typed engine, but `engine.rigid_framework` derives its candidates from the state-point
    dict, so a run over a SUBSET of a force field's guests has non-contiguous ids
    -- and `range(n)` would then swap into a species the run does not contain, or
    miss one it does. Type-indexed, sized from the model, per M13.

    At two candidates the destination is forced and no randomness is drawn:
    asking the RNG which of one thing to pick would consume a value for no
    decision. At three or more it is a uniform draw over the others, which is
    what keeps the proposal symmetric.

    Raises with fewer than two candidates rather than returning `current`: a swap
    proposed in a single-species run is a caller bug, and silently making it a
    no-op would hide a broken guard behind a plausible acceptance rate.
    """
    others = [t for t in species_ids if t != current]
    if not others:
        raise ValueError(
            f"a swap needs at least two species, got {list(species_ids)}; the "
            "single-species guard in draw_move should have prevented this"
        )
    if len(others) == 1:
        return int(others[0])
    return int(rng.choice(others))


def swap_log_acceptance(
    log_z: np.ndarray, current: int, target: int, beta: float, du: float
) -> float:
    """log of the swap acceptance ratio: ln z_s' - ln z_s - beta dU.

    There is NO n_s / (n_s' + 1) combinatorial term. That belongs to an aggregate
    counts move; here a specific labelled particle changes identity, and adding
    the term double counts the uniform-particle selection. The symptom is a
    mixture biased toward the minority species with total N conserved, which is
    easy to mistake for physics -- so the absence of the term is stated here
    rather than left implicit in an expression.

    Detailed balance with the symmetric uniform-particle / uniform-other-species
    proposal is satisfied by the activity ratio alone.
    """
    return float(log_z[target] - log_z[current] - beta * du)


def species_draw_correction(
    log_p_trial: np.ndarray,
    pool_species: IntArray,
    member_rows: IntArray | Sequence[int],
    chosen_rows: IntArray | Sequence[int],
) -> float:
    """M21: the species half of the trial-density measure, for one exchange.

    NOT YET VALIDATED. This is a derivation; the arbiter is the degenerate-species
    run described in multispecies_plan.md section 16, and until that has been
    measured nothing should treat this as known-correct.

    The engine's exchange proposes over a shared pool, `P = surviving trials +
    tombstoned members`, and the DP picks a clique K from it. A trial atom's
    position AND species were drawn; a member's were not, they are part of the
    current configuration. Writing out which atoms are drawn in which direction
    gives three cases, not one:

        K n M_rem   a member reselected at its own row. The same particle, with
                    the same species, in BOTH configurations. Drawn in neither
                    direction, so it contributes nothing. This is the case a
                    per-insertion factor gets wrong.
        K \\ M_rem   a genuine insertion. Species drawn forward with p_s; in the
                    reverse move it is a member and is not drawn.
        M_rem \\ K   a genuine deletion. Not drawn forward; the reverse move must
                    regenerate it as a trial draw.

    so the contribution to the log acceptance ratio is

        sum_{M_rem \\ K} log p_s  -  sum_{K \\ M_rem} log p_s

    Provenance is decided by POOL ROW, never by comparing coordinates: a reselected
    member and a trial atom that happens to land on it are indistinguishable by
    position, and the row index is exact. That is also why the caller must keep the
    permutation it applied to the pool and apply it to the row bookkeeping in the
    same breath.

    Assumption, stated so it can be checked rather than inherited: the sweep's
    one-body weight folds `z_s` and NOT `z_s p_s`. If it ever folds `p_s`, this
    correction is already accounted for there and adding it again double counts.
    One or the other, never both.

    SIGN CONVENTION. The return value is a contribution to log(acceptance ratio).
    The engine assembles `neg_log_acc`, which is MINUS that, so a caller adds
    `-species_draw_correction(...)` there, exactly as it already does with `corr`.
    Spelled out because an inverted combinatorial prefactor in this acceptance has
    cost this project a session once already, and the two conventions meet within
    a few lines of each other.

    A useful check the sign can be read off: at uniform `p_s = 1/T` the sum
    collapses to `(n_deleted - n_inserted) log(1/T)`, i.e. `+dN log T` where `dN`
    is the net particles added. Adding a particle to a T-species mixture means the
    proposal had to guess its species from T options, so the forward proposal is T
    times less likely and the ratio gains a factor of T. That is verified in the
    test suite rather than argued.
    """
    spc, deleted, inserted = _partition_rows(pool_species, member_rows, chosen_rows)
    lp = np.asarray(log_p_trial, dtype=float)
    return float(lp[spc[deleted]].sum() - lp[spc[inserted]].sum())


def _partition_rows(
    pool_species: IntArray,
    member_rows: IntArray | Sequence[int],
    chosen_rows: IntArray | Sequence[int],
) -> tuple[IntArray, BoolArray, BoolArray]:
    """Split an exchange's pool rows into genuinely deleted and genuinely inserted.

    Returns (pool_species, deleted_mask, inserted_mask).

    THE THIRD CASE IS WHY THIS IS SHARED. A member reselected at its own row is
    the same particle with the same species in BOTH configurations -- neither
    deleted nor inserted, contributing to neither sum. Two callers need exactly
    that partition (`species_draw_correction` for the trial-density measure and
    `species_activity_term` for the activities), and a second copy of it would be
    a second place for the reselection case to be got wrong. It is the case a
    per-insertion factor gets wrong, and it is not rare: member reselection was
    measured at 12 of 214 in one `hs_branch` window.

    Provenance is by ROW throughout, never by comparing coordinates.
    """
    spc = np.asarray(pool_species, dtype=np.int64)
    member = np.zeros(len(spc), dtype=bool)
    member[np.asarray(member_rows, dtype=np.int64)] = True
    chosen = np.zeros(len(spc), dtype=bool)
    chosen[np.asarray(chosen_rows, dtype=np.int64)] = True
    return spc, member & ~chosen, chosen & ~member


def species_activity_term(
    log_z: np.ndarray,
    pool_species: IntArray,
    member_rows: IntArray | Sequence[int],
    chosen_rows: IntArray | Sequence[int],
) -> float:
    """The per-species activity contribution to one exchange's log acceptance.

        sum_{inserted} log z_s  -  sum_{deleted} log z_s

    WHAT THIS REPLACES, AND WHY IT IS NOT OPTIONAL. The single-species engine
    carries one `beta * mu` for the NET particle an exchange adds or removes. A
    mixture has no single activity: each inserted atom enters at its own `z_s` and
    each deleted one leaves at its own. One scalar means every species is
    exchanged at the FIRST species' chemical potential.

    Measured on the tree before this existed: with the identity-swap move
    suppressed, a degenerate mixture at mu = [1.5, 0, -1.5] came out UNIFORM,
    [0.323, 0.346, 0.332], where the activities demand [0.419, 0.327, 0.254].
    With swap active it looked correct, because swap equilibrates the composition
    at the right activities and so hides the fact that the exchange never used
    them. That is the same masking that made the first run of the M21 arbiter
    meaningless, and the second time in this stage that a passing gate meant
    nothing until an equilibrating move was switched off.

    THE REPACK CASE. A repack removes k atoms and inserts k, so N is conserved and
    the single-species engine correctly carries no activity term at all. This term
    is still generally NON-ZERO there, because the k that leave and the k that
    arrive need not be the same species. An exchange that conserves particle
    number does not conserve composition.

    SIGN. The return value is a contribution to log(acceptance ratio), matching
    `species_draw_correction`. The engine assembles `neg_log_acc`, which is MINUS
    that, so it enters there with a minus sign.

    REDUCTION. One species: the sums collapse to
    `(n_inserted - n_deleted) * log z_0` -- `+log z_0` for an add, `-log z_0` for
    a remove, `0` for a repack, exactly the three scalar cases, with
    `log z_0 == beta * mu` when the state point came in by the mu route at unit
    thermal wavelength. `tests/engine/test_typed_energy.py` asserts the resulting
    chain is bit-identical to the untyped one.
    """
    spc, deleted, inserted = _partition_rows(pool_species, member_rows, chosen_rows)
    lz = np.asarray(log_z, dtype=float)
    return float(lz[spc[inserted]].sum() - lz[spc[deleted]].sum())
