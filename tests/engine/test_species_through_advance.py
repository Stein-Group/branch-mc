"""P3: the species column survives a window of the main engine.

The column now travels into the cell list with the coordinates and comes back out
of the same slot order. This pins that plumbing, and pins the boundary it stops
at, so the gap is a recorded fact rather than something a later reader discovers
by getting a wrong isotherm.

WHAT IS WIRED. `advance` builds its cell list with `state.species_of`, so every
particle's type is slot-aligned for the whole window and reorders with it; and it
materialises the column back from `cl.species[alive[:N]]`, the same slot order the
positions come from, so the two cannot come apart.

WHAT IS NOT. The exchange move does not yet CHOOSE a species for the particles it
inserts, so an insertion arrives as type 0. `commit_exchange` takes the column and
writes it at insert time, but no caller passes one until the trial cloud draws
species from `p_trial` (P5). Until then a typed run's composition drifts toward
type 0 through exchanges, which is why the typed engine is not yet claimed to
work and why P3's statistical gates are still open.

That gap is asserted below rather than glossed. A test suite that only covered
what works would let the next reader assume the rest does too.
"""

from __future__ import annotations

import _harness as H
import numpy as np
import pytest

from gcmc.engine import MC_engine


def test_a_single_species_run_leaves_the_column_as_none():
    """None means single species, and it must STAY None: io then writes no species
    array, so a schema-2 checkpoint and a schema-3 single-species checkpoint are
    identical in content, and the run fingerprint gains no null key. Writing zeros
    instead would be numerically harmless and would break both of those."""
    cfg, pot, wgh, st = H.make_mode("hs_branch")
    assert st.species is None
    MC_engine.advance(st, cfg, pot, wgh)
    assert st.species is None
    # and the accessor still materialises the right length on demand
    assert st.species_of.shape == (st.n,)
    assert not st.species_of.any()


def test_the_column_comes_back_aligned_with_the_positions():
    """Length and alignment, which is what the slot-order materialisation buys.

    Particles reorder inside a window -- `alive` is a swap-remove list, not the
    incoming order -- so a column carried outside the cell list would be scrambled
    by the first accepted removal. Reading it back from the same `alive[:N]` gather
    the positions use is what makes that impossible rather than merely unlikely.
    """
    cfg, pot, wgh, st = H.make_mode("hs_branch")
    st.species = np.zeros(st.n, dtype=np.int64)
    st.species[::3] = 1
    MC_engine.advance(st, cfg, pot, wgh)
    assert st.species is not None
    assert st.species.shape == (len(st.positions),)
    assert st.species.dtype == np.int64
    assert set(np.unique(st.species).tolist()) <= {0, 1}


@pytest.mark.xfail(
    strict=False,
    reason=(
        "P5 not landed: the DP's pool is the surviving trial cloud PLUS the "
        "tombstoned cavity members, so an accepted exchange can reinsert a "
        "removed member at its own exact coordinates -- measured at 12 of 214 "
        "unchanged-position particles in one hs_branch window. That reinsertion "
        "goes through commit_exchange as a fresh particle and takes type 0, so a "
        "member's species is lost even though nothing about it moved. The fix is "
        "that chosen_species must be looked up PER POOL ATOM (drawn from p_trial "
        "for a trial-cloud atom, carried over for a tombstoned member), not drawn "
        "fresh for every insertion. Expected to XPASS when P5 wires that."
    ),
)
def test_survivors_keep_their_own_species_across_a_window():
    """Identity, not just bookkeeping. A particle that ends the window where it
    started should come out with the type it went in with.

    This currently fails, and the reason is worth more than the test. The
    exchange move does not preserve identity even for particles that end up at
    their original coordinates: `pool = trial_set.mask(km).merge(mol_in_cav)`
    deliberately includes the tombstoned members, because a member's own position
    is a guaranteed-valid placement, and the DP may well choose it. That particle
    is then re-inserted rather than reactivated, so it is a new particle as far as
    the cell list is concerned and takes whatever species the insertion supplies.

    Position is a sound key here despite translations, because a translated
    particle simply drops out of the comparison; what it cannot distinguish is a
    member that was removed and re-inserted at the same spot, which is exactly the
    case that fails.
    """
    cfg, pot, wgh, st = H.make_mode("hs_branch")
    rng = np.random.default_rng(0)
    st.species = rng.integers(1, 3, size=st.n).astype(np.int64)  # no type 0 at all
    before = {tuple(p): int(s) for p, s in zip(st.positions, st.species)}
    assert len(before) == st.n, "duplicate positions; the key is not unique"

    MC_engine.advance(st, cfg, pot, wgh)
    assert st.species is not None

    checked = 0
    for p, s in zip(st.positions, st.species):
        was = before.get(tuple(p))
        if was is None:
            continue  # inserted, or translated away from its old spot
        assert int(s) == was, f"a survivor at {p} changed species {was} -> {int(s)}"
        checked += 1
    assert checked > 20, f"only {checked} survivors traced; the test proved little"


def test_exchange_insertions_are_still_type_zero():
    """The documented gap, asserted so it cannot change unnoticed.

    `commit_exchange` accepts a species column and writes it at insert time, but
    nothing passes one yet: the trial cloud does not draw species until P5. So an
    accepted insertion arrives as type 0 and a typed run's composition drifts
    toward it. When P5 lands, this test SHOULD fail -- that is the point of it, and
    the replacement is the p_s-independence gate.
    """
    cfg, pot, wgh, st = H.make_mode("hs_branch")
    n0 = st.n
    st.species = np.ones(n0, dtype=np.int64)  # nothing is type 0 to begin with
    MC_engine.advance(st, cfg, pot, wgh)
    assert st.species is not None
    inserted = int(np.count_nonzero(st.species == 0))
    assert inserted > 0, (
        "no type-0 particles appeared, so either the window accepted no "
        "insertions or the trial cloud has started choosing species -- if the "
        "latter, this test has served its purpose and P5's gates replace it"
    )
