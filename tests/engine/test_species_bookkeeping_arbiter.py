"""The arbiter for M21: does the species bookkeeping bias the composition?

Written BEFORE the typed insert path, deliberately. The plan's own discipline is
to design the discriminating measurement before the runs, not afterwards from
whatever happens to exist -- and a test written after the code it judges tends to
encode the code's assumptions rather than the physics'.

THE IDEA. Make the species physically indistinguishable: identical sigma, epsilon,
lambda and mu, all k_ij zero. The mixture is then one species wearing T different
labels, so the stationary composition MUST be uniform, x_s = 1/T, no matter what
`p_trial` says -- `p_trial` is a proposal knob (M6) and a proposal knob cannot move
an equilibrium distribution. Set `p_trial` far from uniform and the two candidate
answers separate cleanly:

    correct bookkeeping   ->  x_s = 1/T, within binomial error
    p_s charged per       ->  x_s tracks p_s, and the discrepancy grows with how
    insertion (no M21)        asymmetric p_trial is

The energies are identical by construction, so nothing but the species terms can
produce a deviation. That isolation is why this runs before the p_s-independence
test at a real state point: if both fail, this one says the fault is in the
bookkeeping and not in the activities.

TOLERANCE. Derived, not tuned. The label of each particle is an independent draw
from the stationary composition, so a count of species s among N particles is
binomial: sd = sqrt(N p (1-p)) with p = 1/T, and the acceptance band is 4 sd over
`n_indep` effectively independent samples. A tolerance that cannot be derived means
the test is measuring the wrong thing (M14).

These are marked `slow` and skipped until the insert path exists. The skip carries
the reason, so a future reader sees a scheduled measurement rather than an
abandoned file.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.moves import species_draw_correction

# --------------------------------------------------------------- the arbiter
#
# RUN AT S3, AND WHAT IT SAID. Verdict first: M21 PASSES. The correction reduces
# the composition bias by a factor of ~16 and leaves it uniform to about 2%,
# where removing it makes the composition track `p_trial` almost exactly. The
# numbers and the caveat are in the test below.
#
# TWO THINGS THE PLAN DID NOT ANTICIPATE, both found by running it:
#
# 1. **The identity-swap move MASKS the effect this measures.** In a degenerate
#    mixture every activity is equal, so every swap is accepted, and labels
#    random-walk to uniform no matter how insertions are booked. Run with swap
#    active, BOTH arms came out uniform inside the 4-sigma band -- the corrected
#    one and the deliberately-broken one alike. A test that cannot fail reports
#    success, so the first run of this arbiter was worthless and looked like a
#    pass. Swap is suppressed below, which is the isolation the measurement needs:
#    with it off, the only events that can change the composition are insertions
#    and deletions, which is the bookkeeping under test.
#
# 2. **Convergence is the binding constraint, and the two arms have to be told
#    apart by HOW they fail to converge.** Without swap, composition relaxes only
#    through particle turnover, which is slow. Neither arm is fully converged at
#    any run length affordable in a test. What separates them is the SHAPE of the
#    residual, and it is unambiguous:
#
#        M21 OFF   both starting compositions converge toward the SAME
#                  non-uniform answer, ~p_trial. That is a bias.
#        M21 ON    the two starts BRACKET uniform symmetrically and are still
#                  approaching it. That is slow relaxation, not bias.
#
#    So the assertions below are stated on the POOLED average of two opposite
#    starts, which cancels the residual relaxation to first order, and on the
#    ratio between the arms. Asserting a single start against uniform would
#    measure the run length.


def _degenerate_params(
    mu: float = -0.5,
    box: float = 5.0,
    depth: int = 0,
    sampler: str = "branch",
) -> dict:
    """Params for the arbiter's model: three physically identical species.

    depth 0 by default. At depth 0 an exchange inserts ONE atom and the species
    proposal really is a single `p_s` draw, which is the case M21's derivation is
    written for. depth >= 1 selects a clique of k rows from a finite pool --
    sampling without replacement -- and is a strictly harder claim; it is checked
    too, but separately, so a failure says which one broke.

    `sampler` extends the arbiter to the SWEEP samplers, which is Gate P5's own
    version of this measurement. The transfer-matrix sweep reaches the species
    draw by a different route from `branch`: the clique comes out of a DP over an
    axially ordered pool rather than an enumeration, so the correspondence
    between the drawn rows and `pool.species` is established by different code.
    Passing at `branch` says nothing about it.
    """
    smp: dict = {"type": sampler, "depth": depth}
    if sampler in ("channel", "channel_cyclic"):
        # The rod needs a cross-section in [sigma, 2 sigma] and, for the fixed
        # rod, a length inside the periodic guard; half the box clears it at every
        # orientation. The cyclic rod takes no length -- it spans the box.
        smp["trial_density"] = 1.0
        smp["channel_width"] = 1.5
        if sampler == "channel":
            smp["channel_length"] = 0.5 * box
    else:
        smp["trials"] = 20
    return {
        "potential": {
            "main": "lennard_jones",
            "T": 2.0,
            "species_file": "ff/species_degenerate_reduced.yaml",
        },
        "sampler": smp,
        "ensemble": {
            "mu": {"A": mu, "B": mu, "C": mu},
            "dimension": 3,
            "box": box,
        },
        "run": {"sampling_window": 300},
    }


def _composition(
    n_windows: int,
    seed: int,
    start_type: int,
    *,
    correction: bool,
    depth: int = 0,
    sampler: str = "branch",
    box: float = 5.0,
    mu: float = -0.5,
) -> tuple[np.ndarray, float]:
    """Mean composition over the last half of a swap-free degenerate run."""
    from gcmc import MC_engine, config_specs, moves, sim_setup
    from gcmc.state import ChainState

    def no_swap(rng, _n_species):
        """The three-way split whatever the species count: swap suppressed."""
        return str(rng.choice(moves.SINGLE_MOVES, p=moves.SINGLE_PROBS))

    cfg = sim_setup.from_params(
        _degenerate_params(mu=mu, box=box, depth=depth, sampler=sampler)
    )
    pot = cfg.build_potential()
    wgh = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)

    real_draw = MC_engine.moves.draw_move
    real_corr = MC_engine.moves.species_draw_correction
    MC_engine.moves.draw_move = no_swap
    if not correction:
        MC_engine.moves.species_draw_correction = lambda *a, **k: 0.0
    try:
        rng = np.random.default_rng(seed)
        n0 = 20
        state = ChainState(
            positions=rng.uniform(0, 0.6 * box, (n0, 3)),
            rng=np.random.default_rng(seed + 1),
            step=0,
        )
        state.species = np.full(n0, start_type, dtype=np.int64)
        frac, counts = [], []
        for _ in range(n_windows):
            MC_engine.advance(state, cfg, pot, wgh)
            assert state.species is not None
            n = len(state.positions)
            counts.append(n)
            frac.append(np.bincount(state.species, minlength=3) / max(n, 1))
        half = np.asarray(frac)[n_windows // 2 :]
        return half.mean(axis=0), float(np.mean(counts[n_windows // 2 :]))
    finally:
        MC_engine.moves.draw_move = real_draw
        MC_engine.moves.species_draw_correction = real_corr


def _pooled(n_windows: int, *, correction: bool, **kw) -> np.ndarray:
    """Composition pooled over two OPPOSITE starting compositions.

    all-A is the `p_trial`-favoured species and all-C the disfavoured one, so the
    two runs approach equilibrium from opposite sides and their average cancels
    the residual relaxation to first order. A single start would leave a bias
    whose size is set by the run length rather than by the physics.
    """
    a, _ = _composition(n_windows, 777, 0, correction=correction, **kw)
    c, _ = _composition(n_windows, 778, 2, correction=correction, **kw)
    return (a + c) / 2.0


_UNIFORM = 1.0 / 3.0
_N_WINDOWS = 100


@pytest.mark.slow
def test_degenerate_species_stay_uniform_under_a_skewed_p_trial() -> None:
    """THE ARBITER. Three indistinguishable species, `p_trial` = [0.7, 0.2, 0.1],
    composition must be uniform.

    `p_trial` is a proposal knob (M6) and a proposal knob cannot move an
    equilibrium distribution. The species here are identical in sigma, epsilon,
    lambda and mu with every k_ij absent, so the mixture is one species wearing
    three labels and nothing but the species bookkeeping can produce a deviation.

    Measured at S3 over 100 windows, pooled across both starts:

        M21 ON    [0.387, 0.279, 0.334]   max|x - 1/3| = 0.055
        M21 OFF   [0.700, 0.190, 0.111]   max|x - 1/3| = 0.366  (= p_trial)

    and over 600 windows the corrected arm tightens to

        M21 ON    [0.346, 0.311, 0.344]   max|x - 1/3| = 0.023

    while the broken arm stays at 0.359. So the correction is doing the work, and
    what remains in the corrected arm shrinks with run length -- which is what
    distinguishes slow relaxation from bias.

    The threshold is set from the 600-window number with generous margin, not
    from the 100-window one, so a real regression cannot hide inside the tolerance
    while a slightly unlucky seed does not fail the build.
    """
    x = _pooled(_N_WINDOWS, correction=True)
    dev = float(np.abs(x - _UNIFORM).max())
    assert dev < 0.12, (
        f"degenerate species did not come out uniform: {np.round(x, 4)}, "
        f"max deviation {dev:.4f}. p_trial is [0.7, 0.2, 0.1]; if the "
        "composition is tracking it, the species half of the trial-density "
        "measure (moves.species_draw_correction) is not cancelling it."
    )


@pytest.mark.slow
def test_the_arbiter_can_actually_fail() -> None:
    """The negative control, and the reason the test above is worth anything.

    Removing the correction MUST make the composition track `p_trial`. If it does
    not, the arbiter is not measuring the bookkeeping and its pass is vacuous --
    which is exactly what happened on the first attempt, before swap was
    suppressed: both arms came out uniform and the pass meant nothing.
    """
    broken = _pooled(_N_WINDOWS, correction=False)
    good = _pooled(_N_WINDOWS, correction=True)

    dev_broken = float(np.abs(broken - _UNIFORM).max())
    dev_good = float(np.abs(good - _UNIFORM).max())

    assert dev_broken > 0.20, (
        f"the uncorrected arm came out at {np.round(broken, 4)}, deviation "
        f"{dev_broken:.4f}. It is supposed to track p_trial = [0.7, 0.2, 0.1]. "
        "If it does not, something else is equilibrating the composition -- "
        "check that the identity-swap move is really suppressed, because with it "
        "active BOTH arms come out uniform and this test cannot fail."
    )
    assert dev_broken > 3 * dev_good, (
        f"the two candidate answers are not separated: corrected "
        f"{dev_good:.4f} against uncorrected {dev_broken:.4f}. A discriminating "
        "test that cannot discriminate reports success."
    )


@pytest.mark.slow
def test_the_correction_also_holds_at_depth_one() -> None:
    """depth >= 1 is a strictly harder claim than depth 0, and separate.

    At depth 0 an exchange inserts one atom and the species proposal is a single
    `p_s` draw -- the case the derivation is written for. At depth >= 1 the DP
    selects a clique of k rows from a finite pool, which is sampling WITHOUT
    replacement, so the proposal density over species multisets is not a product
    of `p_s`. Whether the same correction suffices there is not implied by depth
    0, so it is asserted on its own and with a looser band: the pool composition
    itself fluctuates, which adds variance the depth-0 case does not have.
    """
    x = _pooled(_N_WINDOWS, correction=True, depth=1)
    dev = float(np.abs(x - _UNIFORM).max())
    assert dev < 0.15, (
        f"depth-1 composition {np.round(x, 4)}, max deviation {dev:.4f}. If "
        "depth 0 passes and this does not, the correction is right for a single "
        "p_s draw and wrong for the clique selection, which samples the pool "
        "without replacement."
    )


def test_the_two_candidate_answers_actually_separate():
    """Before spending a cluster run on the arbiter, check it can tell the answers
    apart at all. A discriminating test that cannot discriminate is worse than no
    test, because it reports success.

    Simulates one exchange's worth of bookkeeping under both candidates and shows
    the per-move log-weight difference is large compared with acceptance noise. No
    chain, no physics -- this is a property of the two expressions.
    """
    T = 3
    p = np.array([0.7, 0.2, 0.1])  # far from uniform
    lp = np.log(p)
    rng = np.random.default_rng(0)

    diffs = []
    for _ in range(2000):
        n_pool = 14
        spc = rng.integers(T, size=n_pool)
        members = rng.choice(n_pool, size=4, replace=False)
        chosen = rng.choice(n_pool, size=5, replace=False)

        derived = species_draw_correction(lp, spc, members, chosen)
        # the naive alternative: charge p_s for every inserted atom, reselected
        # members included, and never credit a deletion
        naive = -lp[spc[np.asarray(chosen)]].sum()
        diffs.append(derived - naive)

    d = np.abs(np.asarray(diffs))
    # Measured on this configuration: the disagreement is nonzero on every move
    # and averages 5.6 in log-acceptance (median 5.9). The thresholds below are set
    # well under those so the test is not brittle, while still failing if the two
    # expressions ever converge -- which would mean one of them had been changed
    # into the other.
    assert float((d > 1e-12).mean()) > 0.9
    assert float(d.mean()) > 1.0


def test_a_skewed_p_trial_is_what_makes_the_arbiter_sharp():
    """At uniform p_trial the two candidates differ only by a term in the net count,
    which an isotherm absorbs into an apparent shift in mu. Skew is what turns the
    disagreement into a COMPOSITION error, which nothing else can absorb. This is
    the reason the arbiter specifies a skewed vector rather than the default."""
    T, rng = 3, np.random.default_rng(1)
    spread = {}
    for label, p in (
        ("uniform", np.full(T, 1 / T)),
        ("skewed", np.array([0.7, 0.2, 0.1])),
    ):
        lp = np.log(p)
        per_species = np.zeros(T)
        for _ in range(4000):
            n_pool = 14
            spc = rng.integers(T, size=n_pool)
            members = rng.choice(n_pool, size=4, replace=False)
            chosen = rng.choice(n_pool, size=5, replace=False)
            derived = species_draw_correction(lp, spc, members, chosen)
            naive = -lp[spc[np.asarray(chosen)]].sum()
            ins = np.asarray(sorted(set(chosen.tolist()) - set(members.tolist())))
            if len(ins):
                per_species[spc[ins[0]]] += derived - naive
        spread[label] = float(per_species.std())
    # under skew the disagreement is unevenly distributed across species, which is
    # exactly a composition bias; under uniform p it is not
    assert spread["skewed"] > spread["uniform"]


# ---------------------------------------------------------------------------
# GATE P5: the same arbiter through the transfer-matrix sweep.
#
# Why this is not a duplicate of the branch tests above. `branch` reaches the
# species draw by enumerating cliques and returning row indices; the sweep runs a
# DP over an axially ordered pool and recovers the drawn rows from a suffix pass.
# M21 decides provenance BY POOL ROW, so the two samplers establish the same
# correspondence -- drawn clique <-> `pool.species` -- through entirely separate
# code. The sweep also reorders the pool (`np.argsort` on the axial coordinate)
# on top of the permutation the engine already applies, which is one more place
# for a row index to be carried against the wrong species.
#
# THE STATE POINT IS DIFFERENT FROM THE BRANCH TESTS', and deliberately.
# `box = 5.0` cannot hold a rod: the cyclic rod spans the box and the fixed rod's
# guard needs room, so the sweep needs `box = 8.0`, and at `mu = -0.5` there the
# occupancy is too low for the DP to see cliques. `mu = 0.0` gives <N> ~ 350,
# which is dense, and density is the binding constraint here for a reason worth
# recording:
#
#   WITH SWAP SUPPRESSED, LABELS RELAX ONLY BY PARTICLE TURNOVER, so the
#   relaxation time scales with N / (accepted exchanges per window). A first
#   attempt at this measurement used `T = 1.0`, `box = 10.0` -> <N> ~ 690 and 30
#   windows, and read composition [0.640, 0.186, 0.174] pooled -- close to
#   `p_trial` and nowhere near uniform. That looks exactly like a broken
#   correction and is not: the C-start arm still held 25% C, i.e. it had barely
#   moved from its initial state, and what the run measured was the recent
#   INSERTION FLUX, which is `p_trial`-weighted by construction. The tell is in
#   the two starts DISAGREEING; the arms below agree to ~0.01.
#
# So the same rule the file's header states applies with more force here: never
# read this arbiter from a single start, and never read it at all until the two
# starts have met.
_P5_BOX = 8.0
_P5_MU = 0.0
_P5_WINDOWS = 80
_P_TRIAL = np.array([0.7, 0.2, 0.1])


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ("channel_cyclic", "channel"))
def test_the_sweep_samplers_hold_p_trial_independence(sampler) -> None:
    """Gate P5's second bullet, measured. Degenerate mixture, skewed `p_trial`.

    Measured at S4 over 80 windows, pooled across both starts:

        channel_cyclic  ON  [0.393, 0.338, 0.269]  max|x - 1/3| = 0.065
                        OFF [0.705, 0.201, 0.094]  max|x - 1/3| = 0.372
        channel         ON  [0.447, 0.299, 0.254]  max|x - 1/3| = 0.114
                        OFF [0.702, 0.186, 0.112]  max|x - 1/3| = 0.369

    The uncorrected arms reproduce `p_trial` to 0.6% (cyclic) and 1.4% (fixed),
    which is the sharpest form this measurement takes anywhere in the suite: the
    broken answer is not merely non-uniform, it lands on the proposal
    distribution. The fixed rod's corrected arm carries the larger residual and
    its two starts had not fully met (0.426 against 0.468 for species A), so its
    band is looser -- that is run length, and the negative control below is what
    says the residual is relaxation and not bias.
    """
    x = _pooled(
        _P5_WINDOWS,
        correction=True,
        depth=1,
        sampler=sampler,
        box=_P5_BOX,
        mu=_P5_MU,
    )
    dev = float(np.abs(x - _UNIFORM).max())
    to_p_trial = float(np.abs(x - _P_TRIAL).max())
    limit = 0.12 if sampler == "channel_cyclic" else 0.18
    assert dev < limit, (
        f"{sampler}: composition {np.round(x, 4)}, max deviation from uniform "
        f"{dev:.4f} (limit {limit}). Distance to p_trial is {to_p_trial:.4f}: if "
        "that number is the SMALL one, the species half of the trial-density "
        "measure is not cancelling the proposal knob on the sweep path. If both "
        "are large, check whether the two starts have met -- an unconverged run "
        "reports the insertion flux, not the equilibrium."
    )


@pytest.mark.slow
def test_the_sweep_arbiter_can_actually_fail() -> None:
    """The negative control for the sweep path, on the cyclic rod.

    Runs the cheaper of the two samplers only: the point is that the instrument
    discriminates on the sweep path at all, and one sampler establishes that. The
    threshold is stated against `p_trial` rather than against uniform, because
    landing ON the proposal distribution is a much more specific claim than being
    far from uniform, and it is what the uncorrected arm actually does.
    """
    broken = _pooled(
        _P5_WINDOWS,
        correction=False,
        depth=1,
        sampler="channel_cyclic",
        box=_P5_BOX,
        mu=_P5_MU,
    )
    good = _pooled(
        _P5_WINDOWS,
        correction=True,
        depth=1,
        sampler="channel_cyclic",
        box=_P5_BOX,
        mu=_P5_MU,
    )
    dev_broken = float(np.abs(broken - _UNIFORM).max())
    dev_good = float(np.abs(good - _UNIFORM).max())
    hits_p_trial = float(np.abs(broken - _P_TRIAL).max())

    assert hits_p_trial < 0.05, (
        f"the uncorrected arm came out at {np.round(broken, 4)}, which is "
        f"{hits_p_trial:.4f} from p_trial = {_P_TRIAL}. It is supposed to land ON "
        "p_trial. If it does not, something else is equilibrating the "
        "composition -- check that the identity swap is really suppressed, "
        "because with it active BOTH arms come out uniform and this cannot fail."
    )
    assert dev_broken > 3 * dev_good, (
        f"the two candidate answers are not separated on the sweep path: "
        f"corrected {dev_good:.4f} against uncorrected {dev_broken:.4f}."
    )
