"""`run.esjd`: the counters the efficiency comparison is read from.

ESJD per second is the quantity adaptive gamma is meant to improve, so the
counters behind it have to be right in a specific and checkable way rather than
plausible. Four claims:

  ABSENT UNLESS ASKED. A new integer in every muVT summary would move every
    pinned counter digest in the project to record nothing, which is why the
    Glauber engine never got a counter on this side in the first place. So the
    block appears exactly when `run.esjd` is set, and the log columns with it.
  dN_sq IS THE ACCEPTED EXCHANGE COUNT, exactly. An exchange moves N by +/-1 at
    every depth (an add inserts d+1 and removes d), so E[dN^2] per move is the
    exchange acceptance rate. Asserting the identity is what stops someone
    reading `esjd_dNsq` as a measure of how much of the configuration moved --
    it is not, and `clq*_new_a` is.
  dr_sq IS MINIMUM IMAGED. A particle that wrapped the box moved a short way,
    not a long one, and counting the wrap would make ESJD a function of where
    the origin sits.
  THE SUMS ARE RAW. Never rates: a statistic over any span of log rows is then
    exact rather than a mean of means, which is the rule the clique counters
    state and the reason `wall_s` is a separate column.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from gcmc.engine import metropolis_dynamics as md
from gcmc.run import sim_setup

sys.path.insert(0, "tests/scripts")
import _harness as H

G3_CSV = "data/g3_analysis_data.csv"
BOX, N_SEED = 8.0, 80


def _params(*, esjd: bool, adaptive: bool = False, depth: int = 2) -> dict:
    smp: dict = {"type": "rosenbluth", "trials": 30, "trial_box_side": 3.0}
    if adaptive:
        smp |= {"gamma_mode": "adaptive", "pr_target": 0.25}
    run: dict = {
        "kind": "single",
        "seed": 7,
        "sampling_window": 300,
        "upper_cutoff": 2.5,
    }
    if esjd:
        run["esjd"] = True
    return {
        "potential": {"main": "lennard_jones", "T": 1.2},
        "sampler": smp,
        "engine": {"depth": depth},
        "ensemble": {"dimension": 3, "box": BOX, "mu": -3.0, "type": "muVT"},
        "run": run,
        "auxiliary": {"type": "pmf", "g3_csv": G3_CSV, "min_g3": 0.03},
    }


def _windows(params: dict, n: int = 2) -> list[dict]:
    rng = np.random.default_rng(1007)
    pos = H.lattice_seed(BOX, N_SEED, 3, rng)
    cfg, pot, wgh, st = H.build(params, 7, positions=pos)
    return [md.advance(st, cfg, pot, wgh) for _ in range(n)]


def test_the_block_is_absent_unless_the_run_asks_for_it() -> None:
    assert "esjd" not in _windows(_params(esjd=False))[-1]
    assert "esjd" in _windows(_params(esjd=True))[-1]


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_dn_sq_is_exactly_the_accepted_exchange_count(depth: int) -> None:
    """At EVERY depth, which is the point: an add proposes (d+1, d) and a remove
    (d, d+1), so the number change is +/-1 however many positions the move
    replaced. A depth sweep that saw dN_sq grow with depth would mean the
    counter had started measuring something else."""
    for w in _windows(_params(esjd=True, depth=depth)):
        c = w["counters"]
        assert w["esjd"]["dN_sq"] == c["add_accept"] + c["remove_accept"]


def test_the_denominator_counts_every_proposed_move() -> None:
    """Translations have dN = 0 and belong in the denominator of E[dN^2] exactly
    as a rejected exchange does, so `moves` counts them too and is at least the
    translation count."""
    for w in _windows(_params(esjd=True)):
        e = w["esjd"]
        assert e["moves"] >= e["translations"] > 0
        assert e["moves"] >= w["counters"]["add_step"] + w["counters"]["remove_step"]


def test_dr_sq_is_bounded_by_the_proposal_and_therefore_minimum_imaged() -> None:
    """One bound, two claims, and the second is the one worth having.

    The proposal is uniform in a cube of side `move_factor`, so no accepted
    translation can displace a particle by more than 3 (mf/2)^2 -- which is
    0.047 at the default mf of 0.25. An UN-IMAGED wrap, by contrast, records at
    least (box - mf)^2, which is 95: three and a half orders of magnitude above
    the bound, on one move.

    So this bound is the minimum-image test, and it is not a vacuous one:
    a particle wraps whenever it starts within mf/2 of a face, which at mf =
    0.25 in a box of 10 is about 3 * 2 * 0.125 / 10 = 7.5% of moves. Over the
    hundreds of accepted translations in these windows, wraps happen in the
    dozens, and a single one counted the long way round would break the bound
    by itself.
    """
    p = _params(esjd=True)
    cfg = sim_setup.from_params(p)
    mf = cfg.move_factor
    per_move = 3.0 * (mf / 2.0) ** 2
    wrap = (cfg.box - mf) ** 2
    assert wrap > 100.0 * per_move, "the bound cannot separate a wrap here"
    for w in _windows(p):
        accepted = w["counters"]["move_accept"]
        assert accepted > 50, "too few accepted translations to have seen a wrap"
        assert 0.0 < w["esjd"]["dr_sq"] <= accepted * per_move + 1e-9


def test_the_sums_are_raw_and_add_across_windows() -> None:
    """Raw per-window sums, never rates. Summing the columns over any span of
    rows and dividing at the end has to be exact, which a per-window rate could
    not be once the reporting interval changed."""
    outs = _windows(_params(esjd=True), n=4)
    total = sum(w["esjd"]["dN_sq"] for w in outs)
    exch = sum(
        w["counters"]["add_accept"] + w["counters"]["remove_accept"] for w in outs
    )
    assert total == exch
    assert all(w["esjd"]["moves"] > 0 for w in outs)


def test_esjd_and_adaptive_gamma_are_independent_switches() -> None:
    """Both arms of the efficiency comparison have to measure the same way, so
    the counters must not depend on the gamma mode -- and the fixed arm needs
    them just as much as the adaptive one."""
    fixed = _windows(_params(esjd=True, adaptive=False))[-1]
    adapt = _windows(_params(esjd=True, adaptive=True))[-1]
    assert set(fixed["esjd"]) == set(adapt["esjd"])
    assert "gamma_adaptive" not in fixed
    assert "gamma_adaptive" in adapt
