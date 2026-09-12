"""G0.9, tier 0: the periodic guard, scanned over every shipped configuration.

STATIC. No simulation runs: each params file is turned into a Config by
`from_params` -- the only authority on what is legal -- and asked for its guard
margin at the radius its own sweep scores pairs at.

THE VERDICT IS STRUCTURAL, AND THAT IS THE FINDING. Section 3.5 of the plan says
"`sim_setup` never validates `channel_length` against box size", and budgets a
group-2C item -- a shipped result that may be wrong -- for the case where a
scan turns up a violation. On this tree it cannot: `from_params` already
computes the body diagonal and refuses anything that does not fit with a margin
of the radius the sweep reads. So the scan reports zero violations because none
can exist, not because none were looked for, and the two halves of that
sentence are what this file separates: `scan_params_files` records a
guard-refusal explicitly rather than skipping it, so an "all pass" verdict
cannot come from files that never reached the audit.

WHAT THE HEAT BATH ADDS is therefore not the check but two tightenings, and only
one of them can matter: `upper_cutoff` in place of the auxiliary radius (moot,
since the dual-cutoff auxiliary is refused outright under that engine), and the
largest active PAIR FLOOR in place of the single-component diameter on a typed
hard-core run -- the one place the shipped guard was measured at too short a
radius.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gcmc.run import guard_audit
from gcmc.run.sim_setup import from_params

ROOT = Path(__file__).resolve().parents[2]


def _records():
    return guard_audit.scan_params_files(
        [ROOT / "run-files", ROOT / "examples", ROOT / "tests"]
    )


def test_no_shipped_configuration_violates_the_guard_at_its_own_radius():
    """The G0.9 gate. A violation here is the plan's first genuine 2C item and
    stops G1: it would mean the bulk channel sampler has been computing local
    Euclidean distances under a violated precondition, so a shipped result may
    be wrong."""
    s = guard_audit.summarise(_records())
    assert s["n_violations"] == 0, (
        f"{s['n_violations']} shipped configuration(s) violate the periodic "
        f"guard at their own rc_sel: {s['violations']}. Open a "
        "docs/decisions/ entry, reproduce the affected case, and decide whether "
        "pins move -- BEFORE any engine work."
    )
    assert s["audited"] > 0, "the scan found nothing to audit; it has gone blind"


def test_the_scan_records_a_guard_refusal_rather_than_skipping_it():
    """Otherwise "all pass" would be circular.

    `from_params` refuses a body that does not fit, and a scan that swallowed
    every refusal would report zero violations precisely because the violating
    configurations never reached it. Checked with a params file that is legal in
    every respect except the guard.
    """
    import json
    import tempfile

    p = {
        "potential": {"main": "hard_core"},
        "sampler": {
            "type": "channel",
            "depth": 1,
            "trial_density": 1.0,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        "ensemble": {"dimension": 3, "box": 6.0, "mu": 2.0},
    }
    with pytest.raises(ValueError, match="periodic guard"):
        from_params(p)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "bad.json").write_text(json.dumps(p))
        recs = guard_audit.scan_params_files([Path(d)])
    assert len(recs) == 1 and recs[0]["refused_at_config_time"], recs


def test_the_glauber_tightens_the_hard_core_radius_to_the_pair_floor():
    """The one real gap the Glauber engine closes.

    `guard_cut` on the hard-core path is `diameter`, the SINGLE-COMPONENT floor.
    A typed run whose largest `c * sigma_ij` exceeds it was guarded at too short
    a radius -- harmless under Metropolis, where a wrapped pair beyond the guard
    is divided back out of the acceptance, and not harmless at all under a
    kernel with no acceptance to divide it out of.
    """
    base = {
        "potential": {
            "main": "hard_core",
            "species_file": "ff/species_binary_reduced.yaml",
        },
        "sampler": {
            "type": "channel",
            "depth": 1,
            "trial_density": 1.0,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        "ensemble": {"dimension": 3, "box": 8.0, "z": {"A": 0.1, "B": 0.1}},
    }
    try:
        metro = from_params(base)
    except (ValueError, FileNotFoundError) as exc:
        pytest.skip(f"typed hard-core channel is not configurable here: {exc}")
    floor = float(metro.species.table.core_floor.max())
    if floor <= metro.diameter + 1e-12:
        pytest.skip(
            f"this species file's largest pair floor ({floor:.3f}) does not "
            f"exceed the single-component diameter ({metro.diameter:.3f}), so "
            "there is no gap to close for it"
        )
    hb = dict(base)
    hb["sampler"] = {**base["sampler"], "engine": "glauber_dynamics"}
    # the Glauber engine's guard is measured at the larger radius, so a box that is
    # legal under Metropolis can be refused here
    diag = (6.0**2 + 2 * 1.5**2) ** 0.5
    assert diag <= 8.0 - metro.diameter
    if diag > 8.0 - floor:
        with pytest.raises(ValueError, match="periodic guard"):
            from_params(hb)
