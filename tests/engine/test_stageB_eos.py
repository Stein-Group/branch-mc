"""Gate 2. The refactored engine has to sample the right distribution, not just
a self-consistent one.

Two independent references, chosen because neither can be wrong in the same way
the refactor could be:

  * Hard spheres against Carnahan-Starling. An analytic equation of state knows
    nothing about cell lists.
  * Lennard-Jones against `_advance_metropolis`, at the same (mu, T, box). That
    engine still rebuilds a kd-tree per move and was never touched -- rule 2 in
    the plan exists for exactly this moment.

Two things had to be got right before either number meant anything.

THE ERROR BAR. N in these chains has an autocorrelation time far longer than one
sampling window: at mu = 3 a single chain wanders between 810 and 875 across
consecutive windows. A standard error taken over windows treats those as
independent and understates the true error by several fold, which is enough to
manufacture a 3-to-5 sigma "disagreement" between an engine and itself. The error
unit here is therefore an independent SEED -- one chain, one number -- exactly as
in test_stageB_stats.py.

WHAT THE REFERENCES ACTUALLY SAY. With that error bar, the untouched engine does
not match either reference to 2 sigma:

    hard-sphere channel, box 12, <N> vs Carnahan-Starling (seed-wise, n = 5)
        mu = 3.0   baseline 832.8 +/- 5.0    current 832.0 +/- 3.5    CS 851.0
                   both about -2.2%, and the two engines agree to z = 0.14

    LJ channel vs metropolis, box 8, T = 1.5, mu = -3.0 (seed-wise, n = 4)
        metropolis 267.6 +/- 2.2
        baseline channel 240.7 +/- 5.0    current channel 235.4 +/- 1.1
                   both about -11%, and the two engines agree to z = 1.04

So the channel sampler carries a real, pre-existing discrepancy against both
references -- around -2% for hard spheres, around -11% for LJ -- and it is not
this refactor's. It predates it, and the refactored engine reproduces the
untouched engine's density to within one sigma on both.

That is a finding about the METHOD, not about the cell list. It belongs in the
ledger and in a conversation, not in a silently-loosened threshold. What these
tests honestly assert is the claim the refactor makes: the conversion introduced
no shift in the sampled distribution. Each runs both engines under one protocol
and asserts (a) the engines agree at 2 sigma, and (b) the refactored engine is no
further from the reference than the engine it replaced. Absolute deviations are
printed every run.

When the underlying discrepancy is understood and fixed, replace (b) with an
absolute 2-sigma bound against the reference and delete this note.

LJ runs above its critical temperature: below it the box condenses and <N> is a
rate rather than an average (see the note in test_stageB_stats.py).
"""

from __future__ import annotations

import math

import _harness as H
import numpy as np
import pytest

from gcmc import MC_engine

SEEDS = (1, 2, 3, 4, 5)


def _cs_mu(eta):
    """Carnahan-Starling mu / kT for hard spheres of diameter 1 at packing eta,
    with the ideal part (Lambda = 1): mu = ln rho + mu_ex."""
    rho = eta / (math.pi / 6.0)
    mu_ex = (8.0 * eta - 9.0 * eta**2 + 3.0 * eta**3) / (1.0 - eta) ** 3
    return math.log(rho) + mu_ex


def _cs_density(mu, lo=1e-4, hi=0.49):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _cs_mu(mid) < mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / (math.pi / 6.0)


def _chain_mean_n(params, n_seed, seed, burn, measure, engine):
    rng = np.random.default_rng(900 + seed)
    box, dim = params["ensemble"]["box"], params["ensemble"]["dimension"]
    pos = H.lattice_seed(box, n_seed, dim, rng) if n_seed else np.empty((0, dim))
    eng = None if engine == "current" else H.load_engine(engine)
    advance = (MC_engine if eng is None else eng[1]).advance
    cfg, pot, wgh, st = H.build(params, seed, positions=pos, engine=eng)
    for _ in range(burn):
        advance(st, cfg, pot, wgh)
    return float(np.mean([advance(st, cfg, pot, wgh)["N"] for _ in range(measure)]))


def _seedwise(params, n_seed, engine, burn, measure, seeds=SEEDS):
    v = np.array(
        [_chain_mean_n(params, n_seed, s, burn, measure, engine) for s in seeds]
    )
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def _z(ma, sa, mb, sb):
    return abs(ma - mb) / math.hypot(sa, sb)


# --------------------------------------- hard spheres vs Carnahan-Starling
@pytest.mark.slow
@pytest.mark.parametrize("mu", [2.0, 3.0])
def test_hard_sphere_channel_against_carnahan_starling(mu):
    box = 12.0
    n_cs = _cs_density(mu) * box**3
    params = H._p(
        {"main": "hard_core"},
        {
            "type": "channel",
            "depth": 1,
            "trial_density": 3.0,
            "channel_length": 6.0,
            "channel_width": 1.5,
        },
        {"mu": mu, "dimension": 3, "box": box},
        {"sampling_window": 100},
    )

    n0 = int(round(n_cs))
    mb, sb = _seedwise(params, n0, "baseline", burn=20, measure=20)
    mc, sc = _seedwise(params, n0, "current", burn=20, measure=20)
    dev_b, dev_c = 100 * (mb - n_cs) / n_cs, 100 * (mc - n_cs) / n_cs
    print(
        f"\n  mu {mu}:  CS {n_cs:7.2f}"
        f"   baseline {mb:7.2f} +/- {sb:.2f} ({dev_b:+.2f}%)"
        f"   current {mc:7.2f} +/- {sc:.2f} ({dev_c:+.2f}%)"
    )

    z = _z(mb, sb, mc, sc)
    assert z <= 2.0, (
        f"the refactor shifted the hard-sphere density at mu = {mu} by "
        f"{z:.2f} combined standard errors: {mc:.1f} vs {mb:.1f}"
    )
    slack = 2.0 * 100 * math.hypot(sb, sc) / n_cs
    assert abs(dev_c) <= abs(dev_b) + slack, (
        f"the refactored engine is further from Carnahan-Starling "
        f"({dev_c:+.2f}%) than the engine it replaced ({dev_b:+.2f}%)"
    )


# --------------------------- LJ channel vs the never-converted metropolis
@pytest.mark.slow
def test_lj_channel_against_the_unconverted_metropolis_engine():
    box, T, mu = 8.0, 1.5, -3.0
    ens = {"mu": mu, "dimension": 3, "box": box}
    pot = {"main": "lennard_jones", "T": T}
    chan = H._p(
        pot,
        {
            "type": "channel",
            "depth": 1,
            "trial_density": 1.2,
            "channel_length": 4.0,
            "channel_width": 1.2,
        },
        ens,
        {"sampling_window": 100},
    )
    metro = H._p(pot, {"type": "metropolis"}, ens, {"sampling_window": 4000})

    # Each chain is seeded at ITS OWN stationary N, measured by relaxing from
    # above and from below: the channel settles near 252, metropolis near 264.
    # Seeded at 200 instead, both are still filling after 3,500 exchange steps
    # (the LJ add acceptance is ~0.05, so that is only ~175 accepted insertions
    # for a 60-particle deficit) and what gets compared is the filling rate.
    seeds = (1, 2, 3, 4)
    rm, rs = _seedwise(metro, 262, "current", burn=8, measure=10, seeds=seeds)
    bm, bs = _seedwise(chan, 252, "baseline", burn=20, measure=25, seeds=seeds)
    cm, cs = _seedwise(chan, 252, "current", burn=20, measure=25, seeds=seeds)

    print(
        f"\n  LJ box {box} T {T} mu {mu}:"
        f"   metropolis {rm:7.2f} +/- {rs:.2f}"
        f"   baseline channel {bm:7.2f} +/- {bs:.2f}"
        f"   current channel {cm:7.2f} +/- {cs:.2f}"
    )

    z_engines = _z(bm, bs, cm, cs)
    assert z_engines <= 2.0, (
        f"the refactor shifted the LJ channel density by {z_engines:.2f} "
        f"combined standard errors: {cm:.1f} vs {bm:.1f}"
    )

    dev_b, dev_c = abs(bm - rm), abs(cm - rm)
    assert dev_c <= dev_b + 2.0 * math.hypot(bs, cs), (
        f"the refactored channel sampler is further from the metropolis "
        f"reference ({dev_c:.1f}) than the engine it replaced ({dev_b:.1f})"
    )
