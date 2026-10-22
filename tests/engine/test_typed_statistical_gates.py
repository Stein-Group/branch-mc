"""Gate P3's statistical gates, measured against the typed sampler that exists.

`advance_metropolis_mof_ms` is already a complete multi-species grand-canonical
Metropolis: per-species activity, a species draw from fixed `p_s`, an identity
swap, and per-pair core floors. With an EMPTY `FrameworkField` it is a bulk
sampler, so the physics Gate P3 asks about can be measured now rather than after
new engine code is written. That ordering matters: these gates are what will say
whether the acceptance algebra of M21 is right, and a reference that predates the
change is worth more than one written alongside it.

What is being checked, and why each one is sharp:

  IDEAL GAS      at low activity <N_s> -> z_s V independently per species. N_s is
                 Poisson, so the tolerance is derived from sqrt(z_s V / n_indep)
                 rather than tuned. Catches a per-species activity that is
                 mis-scaled -- by lambda^d, say -- which no symmetric test sees.

  DEGENERATE     species identical in every parameter are one species relabelled,
                 so the composition must be UNIFORM regardless of `p_trial`. This
                 is the arbiter named in M21: a wrong species-draw correction makes
                 composition track `p_s` instead, and here the energies are
                 identical by construction so nothing else can be blamed.

  SWAP           with all species identical except activity, the stationary
                 composition is z_s / sum_t z_t. Binomial tolerance. This is the
                 only gate that exercises the swap acceptance, and the term it
                 would catch is the spurious n_s/(n_s'+1) combinatorial factor,
                 whose signature is a bias toward the minority species at
                 conserved N.

Marked `slow`: these need enough samples for the tolerances above to mean
anything, which is minutes rather than the default tier's seconds. They are
cluster-cheap and laptop-affordable, and the plan lists exactly this trade.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.engine.rigid_framework import advance_metropolis_mof_ms
from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable

pytestmark = pytest.mark.slow

BOX = 10.0
RC = 2.5

# Windows to discard before measuring. 150 rather than the 10 I first used, and the
# reason is recorded because it cost a false failure: from an empty box the system
# fills over roughly a hundred windows (<N> was still climbing, 489 at burn 20
# against 634 at burn 150), and until N has plateaued the composition has not
# either. The insertion species draw is uniform by default, so an under-equilibrated
# run sits between the uniform composition it started from and the activity ratio it
# is heading for -- a deviation with a consistent SIGN, which is exactly what a bias
# looks like. The equilibration assertion in `_run` is what stops that being
# mistaken for one again.
BURN = 40

# Inverse temperature for the composition gates. NOT 1.0, and the reason is a
# measurement: at beta = 1 with eps = 1 the reduced temperature is T* = 1.0, which
# is below the Lennard-Jones critical temperature of about 1.31, so the fluid
# condenses and N never plateaus (584 climbing to 657 after 150 burn-in windows,
# caught by the equilibration guard below). beta = 0.4 is T* = 2.5, comfortably
# supercritical, and <N> settles inside 40 windows.
#
# This costs the gates nothing. For species identical in every parameter the label
# sum factorises out of the configurational integral exactly, so x_s = z_s / sum_t
# z_t holds at ANY temperature and density; a supercritical state point tests the
# same statement and equilibrates.
BETA = 0.4

# Blocks for the block-average error. Composition is a slowly relaxing collective
# variable, so consecutive windows are correlated and the independent unit is a
# block of windows, not a window and certainly not a particle.
N_BLOCKS = 10


def _bulk(n_species: int, sigma: float = 1.0, eps: float = 1.0, c: float = 0.9):
    """A degenerate n-species bulk system: every species identical, no framework.

    Built from bare arrays rather than a yaml so the parameters are visible in the
    test that depends on them, and so "identical" is manifestly true instead of
    resting on a file being read correctly.
    """
    types = tuple(f"S{i}" for i in range(n_species))
    sig = np.full((n_species, n_species), sigma)
    e = np.full((n_species, n_species), eps)
    table = PairTable.from_reduced(types, n_species, sig, e, c)
    box = np.full(3, BOX)
    field = FrameworkField(
        np.zeros((0, 3)), np.zeros(0, dtype=np.int64), table, box, RC
    )
    return table, field, box


def _run(
    n_species: int,
    z_star: dict[int, float],
    *,
    windows: int,
    window: int = 400,
    beta: float = BETA,
    seed: int = 0,
    species_probs: dict[int, float] | None = None,
    burn: int = BURN,
):
    """Run and return the per-window N_s history after a burn-in.

    The history is per window, so a block average over windows is the natural
    independent unit and `n_indep` below is honest rather than assumed.
    """
    table, field, box = _bulk(n_species)
    rng = np.random.default_rng(seed)
    pos = np.zeros((0, 3))
    spc = np.zeros(0, dtype=np.int64)
    hist = []
    for w in range(windows + burn):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star=z_star,
            beta=beta,
            box=box,
            volume=BOX**3,
            n_cells=1.0,
            upper_cutoff=RC,
            move_factor=0.25,
            n_neigh=40,
            window=window,
            species_probs=species_probs,
        )
        pos = res.positions
        assert res.species is not None
        spc = res.species
        assert len(spc) == len(pos), "species and positions came apart"
        if w >= burn:
            assert res.n_by_type is not None
            hist.append([res.n_by_type[t] for t in sorted(z_star)])
    out = np.asarray(hist, float)

    # Equilibration guard. If N is still drifting, every composition below is
    # measuring the approach and not the stationary state, and it will do so with a
    # consistent sign. Comparing the two halves of the measured stretch against
    # their own scatter catches that without needing a target value.
    totals = out.sum(axis=1)
    h = len(totals) // 2
    first, second = totals[:h], totals[h:]
    # A BLOCKED error on each half, for the same reason the composition tests use
    # one: N is autocorrelated across windows. I first wrote var/h here, which is
    # the very mistake this guard exists to catch the consequences of, and it
    # reported drift on an equilibrated run (216.4 against 205.6 with the halves
    # fluctuating in both directions).
    se = np.hypot(_blocked_se(first, 5), _blocked_se(second, 5))
    assert abs(first.mean() - second.mean()) <= 5.0 * se, (
        f"N is still drifting: first half {first.mean():.1f}, second "
        f"{second.mean():.1f}, 5 se = {5 * se:.1f}. Raise BURN."
    )
    return out


def _blocked_se(x: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    """Standard error of the mean from block averages.

    The tolerance every composition test below uses. A binomial error over
    (particles x windows) -- which is what I reached for first -- treats every
    particle in every window as an independent draw and is 5 to 15 times too tight
    here, because a particle keeps its label across windows and the composition
    relaxes slowly. That mistake turns a 2 se fluctuation into a 7 se failure.
    """
    m = len(x) // n_blocks
    blocks = x[: m * n_blocks].reshape(n_blocks, m).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(n_blocks))


# ----------------------------------------------------------------- ideal gas


def test_the_ideal_gas_limit_holds_per_species():
    """At z V of order a few particles the gas is dilute enough that the
    interaction correction is far below the Poisson noise, so <N_s> = z_s V is the
    prediction. Two species at DIFFERENT activities, so a shared or averaged
    activity would fail rather than cancel.
    """
    z = {0: 0.002, 1: 0.006}
    hist = _run(2, z, windows=120, window=300, seed=1, beta=1.0, burn=20)
    for i, t in enumerate(sorted(z)):
        want = z[t] * BOX**3
        got = hist[:, i].mean()
        # At this activity N_s is nearly Poisson and nearly uncorrelated between
        # windows, so the Poisson prediction sqrt(mean / n) and the block error
        # agree; the larger is taken so the gate cannot be passed by an
        # underestimate of either.
        poisson = np.sqrt(want / len(hist))
        tol = 4.0 * max(poisson, _blocked_se(hist[:, i]))
        assert abs(got - want) <= tol, (
            f"species {t}: {got:.3f} vs {want:.3f} +/- {tol:.3f}"
        )


# --------------------------------------------------------------- degenerate


@pytest.mark.parametrize(
    "probs",
    [None, {0: 0.6, 1: 0.25, 2: 0.15}, {0: 0.1, 1: 0.1, 2: 0.8}],
    ids=["uniform", "skewed", "very-skewed"],
)
def test_degenerate_species_are_uniform_whatever_p_trial_is(probs):
    """The M21 arbiter. Three identical species at one activity: physically one
    species wearing three labels, so x_s = 1/3 exactly, for ANY insertion
    probability vector.

    If the species-draw probability is mishandled the composition tracks `p_s`
    instead, and the very-skewed case makes that a factor-of-eight error rather
    than a subtle one. Because every pair interaction is identical here, nothing
    but the species bookkeeping can produce a deviation.
    """
    z = dict.fromkeys(range(3), 0.2 / 3)
    hist = _run(3, z, windows=150, window=300, seed=2, species_probs=probs)
    frac = hist / hist.sum(axis=1, keepdims=True)
    for i in range(3):
        tol = 4.0 * _blocked_se(frac[:, i])
        assert abs(frac[:, i].mean() - 1 / 3) <= tol, (
            f"species {i}: x = {frac[:, i].mean():.4f} vs 1/3 +/- {tol:.4f}; "
            f"p_trial was {probs}"
        )


def test_a_degenerate_mixture_reproduces_the_single_species_density():
    """Three identical species at activity z/3 each must give the same TOTAL
    density as one species at z: relabelling cannot change the thermodynamics.

    Compared as totals with a block error from the two runs' own variance, so the
    tolerance comes from the data rather than from a guess.
    """
    z_one = {0: 0.2}
    z_three = dict.fromkeys(range(3), 0.2 / 3)
    a = _run(1, z_one, windows=120, window=300, seed=3).sum(axis=1)
    b = _run(3, z_three, windows=120, window=300, seed=3).sum(axis=1)
    se = np.hypot(_blocked_se(a), _blocked_se(b))
    assert abs(a.mean() - b.mean()) <= 4.0 * se, (
        f"one species {a.mean():.3f} vs three {b.mean():.3f}, 4 se = {4 * se:.3f}"
    )


# --------------------------------------------------------------------- swap


def test_the_stationary_composition_is_the_activity_ratio():
    """All species identical except activity, so the only thing that can set the
    composition is z_s / sum_t z_t. This is the gate that exercises the swap
    acceptance; a spurious n_s/(n_s'+1) term biases toward the minority species
    while conserving N, which is exactly what an activity-ratio check sees.
    """
    z = {0: 0.12, 1: 0.06, 2: 0.02}
    total_z = sum(z.values())
    hist = _run(3, z, windows=150, window=300, seed=4)
    frac = hist / hist.sum(axis=1, keepdims=True)
    for i, t in enumerate(sorted(z)):
        want = z[t] / total_z
        tol = 4.0 * _blocked_se(frac[:, i])
        assert abs(frac[:, i].mean() - want) <= tol, (
            f"species {t}: x = {frac[:, i].mean():.4f} vs {want:.4f} +/- {tol:.4f}"
        )


def test_the_swap_move_actually_fires():
    """A composition test passes trivially if the swap never happens and insertion
    alone sets the mixture. This confirms the move is being proposed AND accepted,
    so the gate above is testing what it claims to."""
    z = {0: 0.12, 1: 0.06, 2: 0.02}
    table, field, box = _bulk(3)
    rng = np.random.default_rng(5)
    pos, spc = np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    swap_step = swap_accept = 0
    for _ in range(20):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star=z,
            beta=BETA,
            box=box,
            volume=BOX**3,
            n_cells=1.0,
            upper_cutoff=RC,
            move_factor=0.25,
            n_neigh=40,
            window=300,
        )
        pos, spc = res.positions, res.species
        swap_step += res.counters["swap_step"]
        swap_accept += res.counters["swap_accept"]
    assert swap_step > 500, f"only {swap_step} swaps proposed"
    assert swap_accept > 50, f"only {swap_accept} swaps accepted of {swap_step}"


# ------------------------------------------------- the same claim, ENGINE path
#
# Everything above runs through `advance_metropolis_mof_ms`, which was the only
# multi-species sampler that existed when P3's gates were written. The engine's
# own typed path -- branch and the transfer-matrix sweeps, reached through
# `metropolis_dynamics.advance` -- shares none of that code: a different move split, a
# different insertion (a clique out of a trial cloud rather than one atom), a
# different activity assembly, and the species-draw correction on top. So the
# gates below are not duplicates; they are the same physical claims asked of the
# sampler the production runs actually use.


def _engine_mean_N(species_file: str, mu_per_species: dict, seed: int, windows: int):
    """<N> over the second half of a typed channel run, from a fresh state.

    From `ChainState.fresh` (via an explicit ChainState with no species column)
    on purpose: that is the start every real run gets, and a column supplied by
    the caller is what hid the S4 window-boundary defect for the whole of P3.
    """
    import numpy as _np

    from gcmc import config_specs
    from gcmc.engine import metropolis_dynamics
    from gcmc.run import sim_setup
    from gcmc.system.state import ChainState

    box, temp = 8.0, 2.0
    params = {
        "potential": {"main": "lennard_jones", "T": temp, "species_file": species_file},
        "sampler": {
            "type": "channel",
            "channel_length": 4.0,
            "trial_density": 1.0,
            "channel_width": 1.5,
        },
        "engine": {"depth": 1},
        "ensemble": {"mu": mu_per_species, "dimension": 3, "box": box},
        "run": {"sampling_window": 300},
    }
    cfg = sim_setup.from_params(params)
    pot = cfg.build_potential()
    wgh = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    rng = _np.random.default_rng(seed)
    state = ChainState(
        positions=rng.uniform(0, 0.6 * box, (20, 3)),
        rng=_np.random.default_rng(seed + 1),
        step=0,
    )
    counts = [
        metropolis_dynamics.advance(state, cfg, pot, wgh)["N"] for _ in range(windows)
    ]
    return float(_np.mean(counts[windows // 2 :])), cfg, state


def test_the_engine_reproduces_the_single_species_density_from_three_labels():
    """Gate P5, first bullet: a degenerate three-species run == the one-species run.

    THE STATE POINT IS NOT THE SAME mu, and getting that wrong is the obvious
    mistake. Three labels at one chemical potential give three times the ideal
    density. The labels are bookkeeping, so the grand partition function of a
    mixture of identical species depends only on `sum_s z_s`; matching that sum is
    the identity, which means `z_s = z / 3`, i.e. `mu_s = mu - kT ln 3`.

    THE TOLERANCE COMES FROM THE SEED SCATTER, NOT FROM WITHIN-RUN BLOCK ERRORS,
    and that is a measurement about this system rather than a convention. At this
    density N is strongly autocorrelated across windows: six one-species runs at
    110 windows gave block errors of 0.3-0.5% each while scattering 1.5% (sd)
    between seeds -- the two one-species seeds furthest apart differed by 2.8%,
    which is eight of their own quoted errors. So a gate written on the block
    error would fail on an unlucky pair of seeds and mean nothing when it passed.

    Reference measurement, 6 seeds x 110 windows:

        one species    <N> = 289.9 +- 1.2   (error over seeds)
        degenerate x3  <N> = 291.3 +- 1.3
                            +0.48%, 0.8 sigma

    with per-seed ratios 0.988, 1.036, 0.997, 1.009, 1.005, 0.995 -- scattered
    both ways. An earlier, shorter run (60 windows) gave three ratios that were
    all ABOVE one (1.003, 1.012, 1.013) and looked like a half-percent bias; at
    110 windows the sign is no longer consistent, so that was run length. Recorded
    because a consistent sign is the thing a bias looks like, and it is worth
    knowing that it appears here from nothing but a short run.

    This is a cheap gate for what it covers: the per-species activity (M10), the
    species-draw correction (M21) and the identity swap, all at once, against a
    reference that shares none of that code.
    """
    import math

    temp, mu = 2.0, -0.5
    mu_s = mu - temp * math.log(3.0)
    # 3 x 70 rather than the 6 x 110 the reference numbers came from: eight runs
    # of 110 windows is about twenty minutes, and a gate nobody can afford to run
    # is not a gate. The band below is set from the seed scatter measured at the
    # longer length, so shortening the run widens the error bar without moving
    # the threshold it has to clear.
    seeds = (901, 902, 903)
    windows = 70

    one = [
        _engine_mean_N("ff/species_one_reduced.yaml", {"A": mu}, s, windows)[0]
        for s in seeds
    ]
    three = []
    for s in seeds:
        n, cfg, state = _engine_mean_N(
            "ff/species_degenerate_reduced.yaml",
            {"A": mu_s, "B": mu_s, "C": mu_s},
            s,
            windows,
        )
        three.append(n)
        # NON-VACUITY: a run that came out single-species would pass the density
        # comparison trivially, since one label at z/3 is not the reference but a
        # collapsed mixture would still be close. Assert it really is a mixture.
        assert cfg.n_species == 3
        assert state.species is not None, "the typed run reported no species column"
        assert int(np.unique(np.asarray(state.species)).size) == 3, (
            f"degenerate run ended with "
            f"{np.unique(np.asarray(state.species)).tolist()} labels present; "
            "the three-label mixture collapsed and the comparison is vacuous"
        )

    m1, m3 = float(np.mean(one)), float(np.mean(three))
    se = float(
        np.hypot(np.std(one, ddof=1), np.std(three, ddof=1)) / np.sqrt(len(seeds))
    )
    rel = abs(m3 - m1) / m1
    assert rel < 0.03, (
        f"one species <N> = {m1:.2f}, degenerate <N> = {m3:.2f}, "
        f"{100 * rel:.2f}% apart ({abs(m3 - m1) / se:.1f} sigma over seeds). "
        f"z_s was set to z/3 (mu_s = {mu_s:.4f} against mu = {mu}); if the two "
        "differ by roughly a factor of three, the per-species activity is being "
        "summed wrongly rather than mis-scaled."
    )
