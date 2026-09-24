"""R8: every Config field is physics XOR execution, and identity is the physics half.

WHY THIS TEST EXISTS. `hash_fingerprint._IDENTITY_FIELDS` is a hand-curated
tuple of 25 names, and hand-curated lists drift. Measured on the tree this plan
starts from:

  * `potential.tail` was ABSENT, so `tail: true` and `tail: false` -- two
    different Hamiltonians -- addressed the SAME directory and would resume
    each other.
  * Eight of the twenty-five keys hashed as literal `null` on an ordinary run.
  * Four were derived from other keys already in the set (`potential_type` from
    `sampler`, which is itself hashed; `rc` and `upper_cutoff` are one number
    hashed twice).
  * `pore_axis_len` was absent while `pore_H` and `pore_R` were present.
  * Two documented sweep axes (`trial_box_side`, `trials`) changed the id only
    by ACCIDENT, leaking through derived keys.

None of that is a bug in anyone's judgement. It is what a maintained list does.

So the fix is not a better list -- it is a PARTITION. Every field of `Config` is
declared below as exactly one of:

    PHYSICS    changes the Hamiltonian OR the Markov chain  -> identity-bearing
    EXECUTION  how the same chain is computed               -> never hashed
    DERIVED    a function of other fields                   -> never hashed

and the test asserts (a) the partition is total, so a new field cannot land in
neither, and (b) the identity set equals the physics set exactly.

THE RULE FOR THE PHYSICS SIDE, stated once because the current code contains two
contradictory versions of it: a field is physics if changing it changes the
sampled ensemble OR the Markov chain used to sample it. Proposal knobs count.
That is why `trial_box_side` and the auxiliary cutoff are physics even though
the equilibrium distribution is invariant under them -- two runs that differ
there are not the same experiment and must not share a directory.

This test is expected to FAIL on the tree as received. That is the point: it
pins the eight findings above so the S2 rebuild has something to turn green.
"""

from __future__ import annotations

import dataclasses

from gcmc import config_fields as CF
from gcmc.run import hash_fingerprint, sim_setup

# ---------------------------------------------------------------- the partition
#
# Grouped by the sub-config each field moves into at S2 (R8). The grouping is
# not cosmetic: after the split, these lists become `dataclasses.fields(...)` of
# the sub-configs and this file shrinks to a two-line assertion.

# The partition lives in the PACKAGE now, at `gcmc.config_fields` -- not here.
# That is the whole of R8: identity is DERIVED from the grouping rather than
# maintained beside it, so this module asserts properties OF the grouping and
# never restates it. A copy here would be a second list to drift, which is the
# failure mode the grouping exists to end.
POTENTIAL = frozenset(CF.POTENTIAL)
SAMPLER = frozenset(CF.SAMPLER)
ENSEMBLE = frozenset(CF.ENSEMBLE)
PHYSICS = frozenset(CF.PHYSICS)

#: Execution and IO are one thing for this test's purposes: both are "how the
#: same chain is computed", and neither is ever hashed. They are separate groups
#: in the package because they become separate sub-configs at R8's second half.
EXECUTION = frozenset(CF.EXECUTION + CF.IO)
DERIVED = frozenset(CF.DERIVED)


def _config_fields() -> set[str]:
    """Every field name on the current Config dataclass."""
    cfg = sim_setup.from_params(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "sampler": {"type": "branch", "trials": 20},
            "engine": {"depth": 1},
            "ensemble": {"mu": 1.0, "dimension": 3, "box": 8.0},
            "run": {"sampling_window": 20},
        }
    )
    return {f.name for f in dataclasses.fields(cfg)}


def test_partition_is_total() -> None:
    """Every Config field is classified. A new field cannot land in neither.

    This is the assertion that makes identity self-maintaining: adding a physics
    knob without classifying it fails here, loudly, at the moment it is added --
    rather than silently producing runs that collide months later.
    """
    classified = PHYSICS | EXECUTION | DERIVED
    actual = _config_fields()

    unclassified = sorted(actual - classified)
    assert not unclassified, (
        f"Config fields not classified: {unclassified}\n"
        "Add each to PHYSICS (changes the Hamiltonian or the Markov chain), "
        "EXECUTION (changes only how the same chain is computed), or DERIVED "
        "(a function of other fields). See the module docstring for the rule."
    )


def test_partition_has_no_phantoms() -> None:
    """The classification names nothing that does not exist on Config.

    `CF.PROPERTIES` is excluded: three classified names -- `potential_type`,
    `species_identity`, `n_species` -- are computed PROPERTIES rather than
    stored fields. They are classified anyway because the old identity tuple
    hashed `potential_type`, and a grouping that silently omitted it would let
    it drift back in unnoticed.

    Keeping the two cases apart matters: "I invented this name" and "this is
    computed, not stored" are different mistakes with different fixes, and a
    test that conflates them tells you neither.
    """
    phantom = sorted(set(CF.ALL) - _config_fields() - CF.PROPERTIES)
    assert not phantom, (
        f"classified names that are not Config fields: {phantom}\n"
        "Either the field was renamed or removed, or it is scheduled and not "
        "yet added; in the second case add it to Config in the same commit. "
        "If it is a computed property, add it to CF.PROPERTIES instead."
    )


def test_partition_is_disjoint() -> None:
    """No field is on two sides. A field that is both is a field nobody owns."""
    for a, b, an, bn in (
        (PHYSICS, EXECUTION, "PHYSICS", "EXECUTION"),
        (PHYSICS, DERIVED, "PHYSICS", "DERIVED"),
        (EXECUTION, DERIVED, "EXECUTION", "DERIVED"),
    ):
        overlap = sorted(a & b)
        assert not overlap, f"{overlap} appear in both {an} and {bn}"


def test_identity_equals_the_physics_half() -> None:
    """The hashed set is exactly the physics set -- no more, no less.

    An xfail until S2: the hand-maintained tuple omitted `lj_tail_pref` (a live
    directory collision between two Hamiltonians) and `pore_axis_len`, and
    included four derived keys. True BY CONSTRUCTION now --
    `_IDENTITY_FIELDS` IS `config_fields.PHYSICS` -- so what this really asserts
    is that nobody has reintroduced an override between the two."""
    hashed = set(hash_fingerprint._IDENTITY_FIELDS)

    missing = sorted(PHYSICS - hashed)
    extra = sorted(hashed - PHYSICS)

    assert not missing and not extra, (
        f"physics fields NOT hashed (runs will collide): {missing}\n"
        f"non-physics fields hashed (runs will split spuriously): {extra}"
    )


def test_the_tail_correction_is_identity_bearing() -> None:
    """Two different Hamiltonians must not address the same directory.

    Split out from the test above because it is not a bookkeeping mismatch: it
    is a correctness bug with a measured consequence. A bulk LJ run with the
    mean-field tail correction on samples a different density from one with it
    off, and today they share a run id and resume each other.
    """
    base = {
        "potential": {"main": "lennard_jones", "T": 1.0, "tail": True},
        "sampler": {"type": "branch", "trials": 20},
        "engine": {"depth": 1},
        "ensemble": {"mu": 3.0, "dimension": 3, "box": 10.0},
        "run": {"sampling_window": 50},
    }
    off = {**base, "potential": {**base["potential"], "tail": False}}

    id_on = hash_fingerprint.run_id(sim_setup.from_params(base))
    id_off = hash_fingerprint.run_id(sim_setup.from_params(off))

    assert id_on != id_off, (
        f"tail on and tail off both hash to {id_on}. Two different Hamiltonians "
        "share a run directory and will resume each other."
    )
