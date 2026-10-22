"""G1 of the design plan: the ladder, against closed forms and brute force.

DETERMINISTIC, NO MCMC. Every assertion here is an identity between two ways of
computing the same number, so nothing in this file can fail for statistical
reasons -- which is the point: when a physics gate later disagrees, these are
what say the ladder is not the cause.

WHAT EACH ONE CATCHES, because a test whose failure mode is not written down is
a test nobody can act on:

  brute force        the whole conditional -- P(k) AND the within-k clique
                     probabilities -- against enumeration over all 2^M subsets.
                     The shipped test checked W_k only, which is the sweep's
                     output rather than the engine's distribution.
  ideal gas          P(k) is exactly Poisson(z V_W). Catches a MISSING (M-k)!
                     (the section 2A.1 defect: the limit becomes
                     Binomial(M, zV/(1+zV)), so occupancy is capped by the trial
                     count and depends on `trial_density`, a tunable) and a
                     DOUBLE-COUNTED activity, which no self-consistency check
                     between the sampler and its own ladder can see.
  hard rods          Z_k = (L - k sigma)^k / k! in closed form. Tests the
                     exclusion DP and the combinatorial factor TOGETHER against
                     analytic numbers rather than against each other.
  current weight     for random legal configurations, the subset equal to the
                     current in-particles has strictly positive weight in its
                     own pool. This is the section 2A.3 pin: if the sweep's
                     floor were ever the stricter of the two, every subset
                     containing a legal current particle would get weight zero
                     and the move would delete that particle with probability
                     one -- silent systematic particle loss, with no reverse leg
                     to fail and no crash.
  weight range       `log_weight` against the reference `_forward` at one-body
                     weight spreads up to 1400 log units. This engine is the
                     first caller to routinely push hundreds of log units
                     through a linear-space accumulator, because outside-guest
                     energies now enter the member weights.
  tail folding       the k-dependent tail term equals the difference of directly
                     computed tail energies at N_out + k, for every k.
  V_W                the volume in the weight equals the measure the cloud is
                     actually drawn from. Nothing else checks that, and if they
                     disagree the ideal-gas test fails by a constant factor per
                     k without saying which of the two is wrong -- so this must
                     precede it in debugging order.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gcmc.forcefield.lennard_jones import lj_pair_energy_sq
from gcmc.forcefield.overlap import overlap_radius
from gcmc.proposals.channel_move import ChannelBody
from gcmc.sampling.channel import (
    ChannelSweepHardCore,
    ChannelSweepLJ,
    ladder_probs,
    window_ladder_logw,
)

BETA = 1.0 / 1.5
RC = 2.5
LOWER = 0.9


def _body(L=5.0, w=1.5, dim=3, box=12.0):
    """An axis-aligned body at the box centre, so local == box coordinates."""
    side = np.full(dim, box)
    return ChannelBody.build(
        side / 2.0,
        np.eye(dim),
        side,
        side,
        dim,
        channel_length=L,
        channel_width=w,
        trial_density=1.0,
        n_trials_base=8,
        n_trials_min=8,
    )


def _pool(rng, M, L=5.0, w=1.5, dim=3):
    """M uniform points in a body of extents L x w^(dim-1), local coordinates."""
    cols = [rng.uniform(-L / 2, L / 2, M)]
    cols += [rng.uniform(-w / 2, w / 2, M) for _ in range(dim - 1)]
    return np.column_stack(cols)


def _brute_ladder_lj(U, mw, kmax, lower, rc, beta):
    """(W_k, {clique: weight}) by enumeration over every subset."""
    n = len(U)
    d2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    over = d2 <= overlap_radius(lower) ** 2
    Wk = np.zeros(kmax + 1)
    per: dict[int, dict[tuple[int, ...], float]] = {k: {} for k in range(kmax + 1)}
    for k in range(kmax + 1):
        for c in itertools.combinations(range(n), k):
            if any(over[a, b] for a, b in itertools.combinations(c, 2)):
                continue
            wgt = float(np.prod([mw[i] for i in c])) if k else 1.0
            for a, b in itertools.combinations(c, 2):
                if d2[a, b] < rc * rc:
                    wgt *= math.exp(
                        -beta * float(lj_pair_energy_sq(d2[a, b : b + 1])[0])
                    )
            if wgt > 0.0:
                Wk[k] += wgt
                per[k][c] = wgt
    return Wk, per


@pytest.mark.parametrize("M", [8, 9, 10, 11, 12])
def test_brute_force_conditional_matches_enumeration(M):
    """G1.1: P(k) AND the within-k clique probabilities, to 1e-10.

    The shipped `test_window_logweight_matches_brute_force` checks W_k. This
    checks the DISTRIBUTION the engine draws from, which is W_k times the
    (M-k)! of section 1 -- so it fails if the combinatorial factor is wrong even
    though W_k is right, and that is exactly the 2A.1 defect.
    """
    rng = np.random.default_rng(100 + M)
    U = _pool(rng, M)
    mw = np.exp(rng.normal(0.0, 1.5, M))
    kmax = min(M, 5)
    sw = ChannelSweepLJ(U, mw, LOWER, RC, BETA, kmax)
    Wk, per = _brute_ladder_lj(U, mw, kmax, LOWER, RC, BETA)

    # the ladder: (M - k)! * W_k, normalised
    want = np.array(
        [Wk[k] * math.gamma(M - k + 1) if Wk[k] > 0 else 0.0 for k in range(kmax + 1)]
    )
    want = want / want.sum()
    got = ladder_probs(window_ladder_logw(sw, kmax))
    assert np.abs(got - want).max() < 1e-10, f"P(k) mismatch: {got} vs {want}"

    # within-k clique probabilities, from the sweep's own sampler
    for k in range(1, kmax + 1):
        if not per[k]:
            continue
        tot = sum(per[k].values())
        draws = 4000
        seen: dict[tuple[int, ...], int] = {}
        r = np.random.default_rng(7)
        for _ in range(draws):
            idx = sw.sample(k, r)
            assert idx is not None
            seen[idx] = seen.get(idx, 0) + 1
        # a 4000-draw multinomial: 5 sigma on the largest cell is generous, and
        # the sampler is exact so this is a smoke test on the CONDITIONAL, with
        # the exact statement made by the weight comparison below
        for c, wgt in per[k].items():
            p = wgt / tot
            if p < 0.02:
                continue
            f = seen.get(c, 0) / draws
            assert abs(f - p) < 5.0 * math.sqrt(p * (1 - p) / draws) + 0.01, (
                f"k={k} clique {c}: sampled {f:.4f} against exact {p:.4f}"
            )


@pytest.mark.parametrize("M", [10, 25, 40])
@pytest.mark.parametrize("zV", [0.5, 3.0, 12.0])
def test_ideal_gas_limit_is_poisson(M, zV):
    """G1.2: interactions and exclusion off -> P(k) is exactly Poisson(z V_W).

    THE CHECK THAT CATCHES BOTH COMBINATORIAL DEFECTS. W_k = C(M, k) times
    (M-k)! gives (z V_W)^k M!/k!, i.e. Poisson(z V_W) -- INDEPENDENT OF M, which
    is the property a missing (M-k)! destroys and a double-counted activity
    shifts. Exact to float, not statistical: this is an identity between two
    closed forms.
    """
    rng = np.random.default_rng(4)
    U = _pool(rng, M)
    # lower = 0 and rc = 0: no exclusion, no interaction. Every subset is legal
    # and carries only its member weights.
    sw = ChannelSweepLJ(U, np.full(M, zV), 0.0, 0.0, BETA, M)
    got = ladder_probs(window_ladder_logw(sw, M))
    want = np.array([math.exp(-zV) * zV**k / math.factorial(k) for k in range(M + 1)])
    want = want / want.sum()  # renormalise the truncation at k = M
    assert np.abs(got - want).max() < 1e-12, f"max dev {np.abs(got - want).max():.2e}"


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_hard_rods_match_the_closed_form(k):
    """G1.3: the ladder against the analytic 1D hard-rod window law.

    For k rods of diameter sigma whose CENTRES lie in a segment of length L with
    pairwise separation >= sigma, the unordered configurational volume is
    (L - (k-1) sigma)^k / k!, so the continuum window law is

        P(k)  prop  z^k (L - (k-1) sigma)^k / k!

    THE DISCRETE LADDER REPRODUCES THAT EXACTLY IN EXPECTATION, and the algebra
    is worth writing out because it is what makes this a test of the exclusion
    DP and the combinatorial factor TOGETHER rather than of either alone. With
    E[Z_k] = C(M, k) (1 - (k-1) sigma / L)^k over a uniform pool of M points,

        (M-k)! E[Z_k] (z V_W)^k = (M-k)! C(M,k) (1-(k-1)s/L)^k (zL)^k
                                = M!/k! * z^k (L - (k-1) sigma)^k

    i.e. the continuum law times a constant. So a wrong combinatorial factor or
    a wrong exclusion mask shows up as a DIFFERENT SHAPE in k, not as an overall
    scale -- which is the one thing a self-consistency check cannot see.

    Averaged over pool realisations because Z_k of a single random pool
    fluctuates; the ladder is then built from E[Z_k], which is the quantity the
    algebra above is about.
    """
    sigma, L, zV = 1.0, 8.0, 3.0
    M, reps = 400, 12
    counts = np.zeros(k + 1)
    for r in range(reps):
        rng = np.random.default_rng(11 + 100 * k + r)
        x = rng.uniform(0.0, L, M)
        sw = ChannelSweepHardCore(np.column_stack([x, np.zeros(M)]), sigma, k)
        counts += np.array([sw.count(j) for j in range(k + 1)], dtype=float)
    counts /= reps

    # the ladder from E[Z_k], with the hard-core activity (z V_W)^j
    V_W = L
    z = zV / V_W
    logw = np.array(
        [
            math.log(counts[j]) + math.lgamma(M - j + 1) + j * math.log(z * V_W)
            if counts[j] > 0
            else -math.inf
            for j in range(k + 1)
        ]
    )
    got = ladder_probs(logw)

    want = np.array(
        [
            z**j * max(0.0, L - (j - 1) * sigma) ** j / math.factorial(j)
            for j in range(k + 1)
        ]
    )
    want = want / want.sum()
    assert np.abs(got - want).max() < 0.02, (
        f"k={k}: ladder {np.round(got, 4)} against the closed form {np.round(want, 4)}"
    )


@pytest.mark.parametrize("seed", range(12))
def test_the_current_configuration_has_positive_weight(seed):
    """G1.4: the incoming in-particles are always a legal clique of their pool.

    THE 2A.3 PIN. If the sweep's overlap floor were ever the STRICTER of the two
    -- the sweep's and the engine's -- every subset containing a legal current
    particle would get weight zero and the window would delete that particle
    with probability one. There is no reverse leg to fail: the symptom is slow
    systematic particle loss with nothing in the log, which is why this is a
    test and not a comment.

    The floors agree today (`OVERLAP_TOL` is one shared constant and both sides
    call `overlap_radius`), so this passes; if it ever fires, that is news and
    belongs in docs/decisions/ before anything is touched.
    """
    rng = np.random.default_rng(500 + seed)
    sigma = 1.0
    # a legal configuration: rejection-sample k_in points at least sigma apart
    k_in = 4
    pts: list[np.ndarray] = []
    while len(pts) < k_in:
        p = np.array(
            [rng.uniform(-2.5, 2.5), rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75)]
        )
        if all(np.linalg.norm(p - q) > sigma + 1e-6 for q in pts):
            pts.append(p)
    U_in = np.array(pts)
    cloud = _pool(rng, 12)
    U = np.vstack([U_in, cloud])
    sw = ChannelSweepHardCore(U, sigma, 6)
    assert sw.count(k_in) > 0
    # and the specific subset {0..k_in-1} is reachable: its own clique weight is
    # positive, which for the hard core means no pair inside the floor
    d2 = ((U_in[:, None, :] - U_in[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(k_in, 1)
    assert (d2[iu] > overlap_radius(sigma) ** 2).all()

    swl = ChannelSweepLJ(U, np.ones(len(U)), sigma, RC, BETA, 6)
    assert math.isfinite(swl.log_weight(k_in))


@pytest.mark.parametrize("spread", [0.0, 50.0, 300.0, 700.0, 1400.0])
def test_log_weight_survives_a_huge_member_weight_spread(spread):
    """G1.5: `log_weight` against the reference `_forward` at spreads to 1400 nats.

    THIS ENGINE IS THE FIRST CALLER to routinely push hundreds of log units
    through the fast path's linear-space accumulator, because the outside-guest
    energy now enters every member weight and at high loading that is a large
    negative number. The reference `_forward` accumulates in LOG space and is
    kept in the tree precisely as this oracle.
    """
    M = 11
    rng = np.random.default_rng(int(spread) + 3)
    U = _pool(rng, M)
    logmw = rng.uniform(-spread / 2, spread / 2, M)
    mw = np.exp(np.clip(logmw, -700, 700))
    kmax = 4
    sw = ChannelSweepLJ(U, mw, LOWER, RC, BETA, kmax)
    assert sw._fast, "the fast path did not engage; the comparison is vacuous"
    ref = sw._forward()
    for k in range(kmax + 1):
        want = -math.inf
        for (_state, kappa), lw in ref[M].items():
            if kappa == k:
                want = np.logaddexp(want, lw)
        got = sw.log_weight(k)
        if math.isinf(want):
            assert math.isinf(got)
        else:
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (
                f"spread {spread}: k={k} fast {got!r} vs reference {want!r}"
            )


def test_the_tail_folds_into_the_ladder_exactly():
    """G1.6: the k-dependent tail term equals the difference of direct tail
    energies at N_out + k, for every k.

    U_tail is a function of the TOTAL occupancy, `tail * N^2`, so it is a
    function of k alone once N_out is fixed and folds into the ladder exactly.
    It also couples the window law to the global N_out, which is why the term is
    rebuilt per window and windows are not independent -- a property of the
    mean-field tail, not of this engine.
    """
    from gcmc.engine.glauber_dynamics import _ladder_extra

    tail, beta, n_out, kmax = -0.0123, 1.0 / 1.2, 137, 6
    extra = _ladder_extra(kmax, n_out=n_out, lj=True, log_zv=0.0, beta=beta, tail=tail)
    for k in range(kmax + 1):
        direct = -beta * tail * (n_out + k) ** 2
        assert abs(extra[k] - direct) < 1e-12
    # and the DIFFERENCES are what the acceptance of the Metropolis engine uses,
    # so they must agree with `beta * tail * (n_new^2 - n_old^2)` term for term
    for k in range(kmax):
        delta_ladder = extra[k] - extra[k + 1]
        delta_engine = beta * tail * ((n_out + k + 1) ** 2 - (n_out + k) ** 2)
        assert abs(delta_ladder - delta_engine) < 1e-12


def test_the_hard_core_activity_enters_the_ladder_once():
    """The hard-core convention of section 1.1: ACTIVITY_TERM(k) = (z V_W)^k.

    The hard-core sweep takes no member weights, so the activity has nowhere
    else to go; the LJ sweep carries it inside each member weight, so the ladder
    term must then be 1. Asserting which path is active is the rule, and here it
    is asserted rather than trusted: the same physical window computed both ways
    must give the same P(k).
    """
    M, zV = 14, 4.0
    rng = np.random.default_rng(19)
    U = _pool(rng, M)
    sigma = 1.0

    hc = ChannelSweepHardCore(U, sigma, M)
    ladder_hc = window_ladder_logw(hc, M, log_extra=np.arange(M + 1) * math.log(zV))
    # the same conflict graph through the LJ kernel with no interaction, the
    # activity in the member weights instead
    # rc = sigma, NOT zero: the LJ sweep's eviction window IS `rc`, so a
    # zero radius means no pair is ever in the window and no overlap mask is
    # built at all -- the sweep would then have no exclusion and the comparison
    # would be vacuous. At rc = sigma every conflicting pair is scored and the
    # only pairs left "within range" sit in (sigma - tol, sigma), where
    # u(r) = 4(r^-12 - r^-6) is zero to within a hair, so the interaction is
    # absent and the two kernels see the same conflict graph.
    lj = ChannelSweepLJ(U, np.full(M, zV), sigma, sigma, BETA, M)
    ladder_lj = window_ladder_logw(lj, M)

    p_hc, p_lj = ladder_probs(ladder_hc), ladder_probs(ladder_lj)
    assert np.abs(p_hc - p_lj).max() < 1e-11, (
        f"the two activity placements disagree: {p_hc} vs {p_lj}"
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_V_W_equals_the_measure_the_cloud_is_drawn_from(dim):
    """G1.9: Monte Carlo the body volume from `trial_cloud` and compare.

    NOTHING ELSE CHECKS THIS. `V_W` multiplies the activity, and the cloud is
    what the pool's fresh rows are drawn from; if the two are different bodies,
    the ideal-gas limit is off by a constant factor per k and G1.2 fails without
    saying which of the two is wrong. So this gate must precede it in debugging
    order, which is why it is in this file and not left to the physics.
    """
    L, w, box = 5.0, 1.5, 12.0
    body = _body(L=L, w=w, dim=dim, box=box)
    rng = np.random.default_rng(3)
    n = 400_000
    pts = body.trial_cloud(rng, n)
    # fraction of an enclosing axis-aligned box the cloud occupies. The body is
    # axis-aligned here (R = I), so the enclosing box is exactly L x w^(dim-1)
    # and the estimate is the sample's own bounding measure.
    local = pts - body.center
    lo, hi = local.min(axis=0), local.max(axis=0)
    est = float(np.prod(hi - lo))
    exact = body.volume()
    assert exact == pytest.approx(L * w ** (dim - 1), rel=1e-12)
    # 400k points bound the extents to ~L/n of the true half-width, so this is a
    # tight comparison rather than a loose one
    assert est == pytest.approx(exact, rel=2e-4), f"{est} vs {exact}"

    # and the cloud is INSIDE the body it claims: a point outside would mean the
    # measure matched by accident
    assert (np.abs(local[:, 0]) <= L / 2 + 1e-9).all()
    assert (np.abs(local[:, 1:]) <= w / 2 + 1e-9).all()


def test_the_full_span_body_reports_the_volume_the_gap_leaves():
    """The rod that closed on itself had no separable exterior and therefore no
    V_W to report. The gap is what gives it one, so the number here is the
    difference between the two bodies rather than a detail of this one."""
    side = np.full(3, 12.0)
    body = ChannelBody.build(
        side / 2.0,
        np.eye(3),
        side,
        side,
        3,
        channel_length=0.0,
        channel_width=1.5,
        trial_density=1.0,
        n_trials_base=8,
        n_trials_min=1,
        full=True,
        reach=2.5,
    )
    # an axis line closes on itself, so the gap is exactly the reach
    assert 2.0 * body.half_len == pytest.approx(12.0 - 2.5)
    assert body.volume() == pytest.approx((12.0 - 2.5) * 1.5**2)


# ---------------------------------------------------------------------------
# PRUNING: the sweep carries the placeable rows, the ladder carries M.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [8, 14, 22])
@pytest.mark.parametrize("blocked_frac", [0.0, 0.3, 0.75])
def test_dropping_unplaceable_rows_leaves_the_ladder_where_it_was(M, blocked_frac):
    """The invariance the whole pruning change rests on, asserted directly.

    A row whose weight is zero cannot be in any clique, so removing it from the
    transfer matrix cannot move W_k. What it CAN move is `(M - k)!`, because
    that factor used to be read off the sweep's row count -- which is why the
    pruned sweep is handed `pool_size=M` and the two ladders have to agree to
    float precision rather than merely be close.

    Both kernels are exercised: the hard-core sweep expresses unplaceability as
    `admissible=False`, the LJ sweep as a zero member weight, and the two take
    different paths to the same `-inf` row.
    """
    rng = np.random.default_rng(4100 + M + int(100 * blocked_frac))
    U = _pool(rng, M)
    adm = rng.random(M) >= blocked_frac
    adm[0] = True  # at least one placeable row, as the engine guarantees
    keep = np.flatnonzero(adm)
    kmax = min(M, 6)
    extra = np.arange(kmax + 1) * 0.37  # a k-only term, as the engine passes

    # hard core
    full = ChannelSweepHardCore(U, LOWER, kmax, adm)
    cut = ChannelSweepHardCore(U[keep], LOWER, kmax, adm[keep])
    a = window_ladder_logw(full, kmax, log_extra=extra)
    b = window_ladder_logw(cut, kmax, log_extra=extra, pool_size=M)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b)), f"support moved: {a} vs {b}"
    assert np.abs(a[fin] - b[fin]).max() < 1e-12, f"hard core: {a} vs {b}"

    # LJ: the same statement with weights, where W_k is a sum and not a count
    mw = np.where(adm, np.exp(rng.normal(0.0, 1.2, M)), 0.0)
    full_lj = ChannelSweepLJ(U, mw, LOWER, RC, BETA, kmax)
    cut_lj = ChannelSweepLJ(U[keep], mw[keep], LOWER, RC, BETA, kmax)
    a = window_ladder_logw(full_lj, kmax, log_extra=extra)
    b = window_ladder_logw(cut_lj, kmax, log_extra=extra, pool_size=M)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b)), f"support moved: {a} vs {b}"
    assert np.abs(a[fin] - b[fin]).max() < 1e-12, f"lj: {a} vs {b}"


def test_forgetting_pool_size_is_what_moves_the_occupancy():
    """The failure mode the `pool_size` argument exists to prevent.

    Pruning without passing M is silent: nothing raises, the ladder is still a
    distribution, and only <N> moves. This asserts that it DOES move, so that
    the previous test is checking something rather than passing vacuously.
    """
    rng = np.random.default_rng(99)
    M = 16
    U = _pool(rng, M)
    adm = rng.random(M) >= 0.5
    adm[0] = True
    keep = np.flatnonzero(adm)
    kmax = 5
    extra = np.arange(kmax + 1) * math.log(2.0)

    full = ChannelSweepHardCore(U, LOWER, kmax, adm)
    cut = ChannelSweepHardCore(U[keep], LOWER, kmax, adm[keep])
    right = ladder_probs(window_ladder_logw(full, kmax, log_extra=extra))
    wrong = ladder_probs(window_ladder_logw(cut, kmax, log_extra=extra))
    ks = np.arange(kmax + 1)
    assert float((ks * wrong).sum()) > float((ks * right).sum()) + 0.05, (
        "reading M off the sweep's row count was supposed to inflate <k>; if "
        "the two agree, this test is no longer guarding anything"
    )


def test_pool_size_below_the_row_count_is_refused():
    """M is the pool and the rows are a subset of it, so M < n is incoherent."""
    rng = np.random.default_rng(5)
    U = _pool(rng, 9)
    sw = ChannelSweepHardCore(U, LOWER, 4)
    with pytest.raises(ValueError, match="smaller than the sweep"):
        window_ladder_logw(sw, 4, pool_size=8)
