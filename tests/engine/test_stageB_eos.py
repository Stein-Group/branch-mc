"""Stage B changes which particle a draw selects, so it cannot be checked by
bit-identity. It is checked three ways instead.

1. Invariants. Whatever the engine did to its bookkeeping, the configuration it
   hands back must be a legal one: no overlapping pair, every centre in the box,
   `len(state.positions)` equal to the reported N, and NVT conserving N exactly.
   These catch a mis-wired swap-remove, which is the failure mode that index
   bookkeeping invites.

2. Statistical parity against `tests/ref/engineA` -- the frozen stage-A build,
   which was bit-identical to the pre-refactor engine. Selection is uniform over
   the live particles in both, so mean N, the acceptance rates and the mean
   cavity occupancy must agree; only the realised chain differs. Agreement is
   asserted at two combined standard errors over independent seeds.

3. Determinism. A seed still names one chain: run twice, get the same bytes.

Test-only code, so scipy is used freely as the overlap oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import _harness as H
import numpy as np
import pytest
from scipy.spatial import cKDTree

from gcmc import MC_engine

# R1: this module's oracle is the frozen stage-A engine, which lives in tag
# v2-engineA rather than in the tree. Skipped as a module when it is absent --
# tier 3 does the `git worktree add` and runs these for real.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _frozen

_frozen.skip_module_without(_frozen.ENGINEA)


# --------------------------------------------------------------- 1. invariants
@pytest.mark.parametrize("mode", sorted(H.REF_MODES))
def test_invariants_after_five_windows(mode):
    cfg, pot, wgh, st = H.make_mode(mode)
    for _ in range(5):
        stats = MC_engine.advance(st, cfg, pot, wgh)

    p = np.asarray(st.positions, float)
    assert len(p) == stats["N"], "reported N disagrees with the configuration"
    if len(p) == 0:
        return
    assert (p >= 0).all() and (p < cfg.box).all(), "a centre left the box"

    # no overlapping pair, minimum image
    tree = cKDTree(p, boxsize=[cfg.box] * cfg.dimension)
    pairs = tree.query_pairs(cfg.lower_cutoff, output_type="ndarray")
    if len(pairs):
        d = p[pairs[:, 0]] - p[pairs[:, 1]]
        d -= np.round(d / cfg.box) * cfg.box
        dist = np.sqrt((d * d).sum(axis=1))
        bad = dist <= cfg.lower_cutoff - 1e-12  # the project's convention
        assert not bad.any(), (
            f"{mode}: {bad.sum()} overlapping pairs, closest {dist.min():.6f} "
            f"vs sigma {cfg.lower_cutoff}"
        )

    mpos = getattr(cfg, "matrix_positions", None)
    if mpos is not None and len(mpos):
        dm = np.asarray(mpos, float)
        tm = cKDTree(dm % cfg.box, boxsize=[cfg.box] * cfg.dimension)
        assert tm.query(p, k=1)[0].min() > cfg.lower_cutoff - 1e-12, (
            "a fluid particle sits on a frozen obstacle"
        )


def _nvt_channel_params(window):
    return H._p(
        {"main": "hard_core"},
        {
            "type": "channel",
            "full_repacking": True,
            "trial_density": 3.0,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        {"dimension": 3, "box": 12.0, "type": "NVT", "N": 400},
        {"sampling_window": window},
    )


def test_nvt_conserves_n_exactly():
    rng = np.random.default_rng(3)
    pos = H.lattice_seed(12.0, 400, 3, rng)
    cfg, pot, wgh, st = H.build(_nvt_channel_params(60), 3, positions=pos)
    for _ in range(5):
        stats = MC_engine.advance(st, cfg, pot, wgh)
        assert stats["N"] == 400, f"NVT lost particles: N = {stats['N']}"
        assert len(st.positions) == 400


# ---------------------------------------------------------- 2. statistics vs A
#
# Which observables can be compared, and which cannot.
#
# `mean N` over a 30-window muVT chain is only an equivalence statistic if the
# chain is STATIONARY. For hard spheres at mu = 3 in this box it is: N sits at
# 819 +/- 15 with no drift, and the two engines agree to z = 0.02. For LJ at
# T = 0.85 it is not: the box is condensing toward liquid density and N climbs
# for as long as anyone is willing to run it. Measured null distribution of the
# statistic there, stage-A engine against ITSELF on three disjoint seed sets:
#
#     seeds  1-10   mean N 963.97   sem 0.72
#     seeds 21-30   mean N 965.37   sem 1.18
#     seeds 41-50   mean N 967.47   sem 1.28
#
# The seed means spread by 3.5 while their standard errors say ~1: the across-
# seed SE understates the true error, because the seed means are not draws from
# a stationary distribution but samples of one common transient. A 2-sigma test
# on that quantity fails for an unchanged engine. So the LJ mode is run in NVT
# instead, where N is conserved exactly and every remaining observable is
# stationary; muVT LJ is validated by the EOS gate in B3, against the
# never-converted `_advance_metropolis`.
_HS_CHANNEL = H._p(
    {"main": "hard_core"},
    {
        "type": "channel",
        "depth": 1,
        "trial_density": 3.0,
        "channel_length": 6.0,
        "channel_width": 1.5,
    },
    {"mu": 3.0, "dimension": 3, "box": 12.0},
    {"sampling_window": 60},
)

_LJ_CHANNEL_NVT = H._p(
    {"main": "lennard_jones", "T": 0.85},
    {
        "type": "channel",
        "full_repacking": True,
        "trial_density": 1.2,
        "channel_length": 5.0,
        "channel_width": 1.2,
    },
    {"dimension": 3, "box": 12.0, "type": "NVT", "N": 800},
    {"sampling_window": 40},
)

_STAT_MODES = {
    # name: (params, n_seed, rate counters, stationary muVT?)
    "hs_channel": (_HS_CHANNEL, 845, ("move", "add", "remove"), True),
    # NVT proposes repacks only -- no translations -- so "move" would be a
    # vacuous 0.0 == 0.0 comparison rather than a test of anything.
    "lj_channel_nvt": (_LJ_CHANNEL_NVT, 800, ("repack",), False),
}

SEEDS = tuple(range(1, 11))
BURN, MEASURE = 12, 30


def _run_chain(params, n_seed, seed, engine, rates):
    rng = np.random.default_rng(500 + seed)
    pos = H.lattice_seed(params["ensemble"]["box"], n_seed, 3, rng)
    eng = None if engine == "current" else H.load_engine(engine)
    advance = (MC_engine if eng is None else eng[1]).advance
    cfg, pot, wgh, st = H.build(params, seed, positions=pos, engine=eng)
    for _ in range(BURN):
        advance(st, cfg, pot, wgh)
    n, ncav, acc = [], [], {k: [] for k in rates}
    for _ in range(MEASURE):
        s = advance(st, cfg, pot, wgh)
        n.append(s["N"])
        ncav.append(s["n_cav"])
        c = s["counters"]
        for k in rates:
            steps = c[f"{k}_step"]
            acc[k].append(c[f"{k}_accept"] / steps if steps else 0.0)
    third = MEASURE // 3
    out = {
        "N": float(np.mean(n)),
        "n_cav": float(np.mean(ncav)),
        "_drift": float(np.mean(n[-third:]) - np.mean(n[:third])),
        "_n_min": int(min(n)),
        "_n_max": int(max(n)),
    }
    out.update({f"acc_{k}": float(np.mean(v)) for k, v in acc.items()})
    return out


def _z(a, b):
    """|mean difference| in combined standard errors, over independent seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sa = a.std(ddof=1) / np.sqrt(len(a))
    sb = b.std(ddof=1) / np.sqrt(len(b))
    se = np.hypot(sa, sb)
    if se == 0:
        # both samples constant: either the observable is degenerate (nothing is
        # being tested) or the engines differ with no scale to measure it against
        raise AssertionError(
            f"degenerate comparison: both samples are constant at "
            f"{a.mean()} / {b.mean()}; this observable tests nothing"
        )
    return abs(a.mean() - b.mean()) / se


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(_STAT_MODES))
def test_statistical_parity_with_stage_a(mode):
    params, n_seed, rates, stationary = _STAT_MODES[mode]
    cur = [_run_chain(params, n_seed, s, "current", rates) for s in SEEDS]
    ref = [_run_chain(params, n_seed, s, "stageA", rates) for s in SEEDS]

    keys = ["n_cav"] + [f"acc_{k}" for k in rates]
    if stationary:
        # only comparable because it IS stationary; the guard below enforces that
        keys.insert(0, "N")
        drift = np.mean([abs(r["_drift"]) for r in cur])
        assert drift < 0.02 * np.mean([r["N"] for r in cur]), (
            f"{mode}: N drifts by {drift:.1f} within the measurement window; "
            f"its mean is a transient, not an equilibrium value, and the z below "
            f"would not mean what it says"
        )
    else:
        n_fixed = params["ensemble"]["N"]
        for r in cur + ref:
            assert r["_n_min"] == r["_n_max"] == n_fixed, "NVT lost particles"

    # Documented intentional divergences from the stage-A engine. The depth >= 1
    # proposal used to pin the cavity centre exactly ON an existing particle for
    # remove/repack, while an add centred it uniformly; the two directions could
    # therefore never propose the same body, and detailed balance failed. Both
    # now use the depth-0 mirror geometry (anchor + uniform offset over the body,
    # anchor forced into the removal set). Consequences, all expected:
    #   * n_cav falls -- a particle-pinned centre guaranteed an occupant, so the
    #     old cavity occupancy was biased high (channel: 3.78 -> 3.33).
    #   * acc_repack shifts with it.
    # Stage A is the *old* engine, so these observables cannot agree; the fix is
    # validated against the exact grand-canonical density instead (see
    # tests/scripts/check_mirror_eos.py). Anything NOT listed here must still
    # match stage A, so the guard below stays at 2.0 for every other observable.
    _EXPECTED_DIVERGENCE = {
        "hs_channel": {"n_cav", "acc_repack", "acc_add", "acc_remove"},
        "lj_channel_nvt": {"n_cav", "acc_repack"},
    }
    exempt = _EXPECTED_DIVERGENCE.get(mode, set())

    report, worst = [], (0.0, None)
    for key in keys:
        a = [r[key] for r in cur]
        b = [r[key] for r in ref]
        z = _z(a, b)
        report.append(
            f"    {key:12s} B {np.mean(a):9.4f}   A {np.mean(b):9.4f}   z {z:5.2f}"
            + ("   (expected divergence)" if key in exempt else "")
        )
        if z > worst[0] and key not in exempt:
            worst = (z, key)
    print(f"\n  {mode}\n" + "\n".join(report))
    assert worst[0] <= 2.0, (
        f"{mode}: {worst[1]} differs from the stage-A engine by "
        f"{worst[0]:.2f} combined standard errors. Before believing it, run the "
        f"same-engine control (current vs current on disjoint seeds): this "
        f"statistic's null is only well behaved on stationary observables.\n"
        + "\n".join(report)
    )


# ------------------------------------------------------------- 3. determinism
@pytest.mark.parametrize(
    "mode", ["hs_channel", "lj_channel_full", "hs_full_repacking", "hs_matrix"]
)
def test_same_seed_same_chain(mode):
    def run():
        cfg, pot, wgh, st = H.make_mode(mode)
        for _ in range(3):
            MC_engine.advance(st, cfg, pot, wgh)
        return np.asarray(st.positions, float)

    a, b = run(), run()
    assert a.shape == b.shape and np.array_equal(a, b), (
        f"{mode}: the same seed produced two different chains"
    )
