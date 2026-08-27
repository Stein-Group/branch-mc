"""The demixing order parameter reads the three arrangements it has to tell
apart, and its null is where the algebra says it is.

Every assertion here names the value it would have to see in order to fail,
because the parameter's whole job is to be believed when it comes back nonzero
on a production run:

  * SEPARATED   two slabs of one species each -> alpha near 1. Fails if the
                statistic cannot see demixing at all.
  * RANDOM      the same positions, labels shuffled -> alpha inside the
                permutation band. Fails if the expectation is mis-derived, which
                is the failure that would make every production number look like
                a result.
  * ALTERNATING a checkerboard -> alpha < 0. Fails if the statistic saturates at
                zero from below, i.e. reports "not demixed" for a structure that
                is the opposite of demixed.

The permutation band is measured rather than assumed. Its own mean is checked
against zero: that is the arithmetic gate on `_expected_unlike`, and it is the
one number in this file that comes from the algebra rather than from a shuffle.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import demixing

BOX = 12.0
DIM = 3
R_CUT = demixing.SHELL_R_CUT


N_SIDE = 10


def _lattice(n_side: int = N_SIDE) -> np.ndarray:
    """A simple cubic lattice filling the box, at a spacing that puts EXACTLY the
    six nearest neighbours inside the shell.

    The spacing has to satisfy `a <= R_CUT < a * sqrt(2)`, and the check below is
    not decoration: at `a = 1.0` the twelve face-diagonal neighbours at `1.414`
    also fall inside `R_CUT = 1.5`, and on a checkerboard those are all LIKE
    pairs, so the alternating case comes out POSITIVE. That is the statistic
    reading its shell correctly and the fixture asking the wrong question, which
    is the more dangerous of the two and the reason the geometry is pinned here
    rather than left to a round number.
    """
    a = BOX / n_side
    assert a <= R_CUT < a * np.sqrt(2), (
        f"spacing {a} puts the wrong shell inside R_CUT = {R_CUT}"
    )
    g = (np.arange(n_side) + 0.5) * a
    return np.array([[x, y, z] for x in g for y in g for z in g])


def _p_unlike(n: int) -> float:
    """Probability that a random pair of an equimolar binary set is unlike."""
    h = n // 2
    return 1.0 - 2.0 * h * (h - 1) / (n * (n - 1))


def _slab_labels(pos: np.ndarray) -> np.ndarray:
    """A in the lower half of x, B in the upper: fully separated."""
    return (pos[:, 0] >= BOX / 2).astype(np.int64)


def _checkerboard_labels(pos: np.ndarray, n_side: int = N_SIDE) -> np.ndarray:
    """Alternating labels on the lattice: unlike neighbours everywhere."""
    idx = np.round(pos / (BOX / n_side) - 0.5).astype(int)
    return (idx.sum(axis=1) % 2).astype(np.int64)


def test_separated_reads_the_value_the_geometry_predicts():
    """Two slabs, and the answer is arithmetic rather than a threshold.

    The box is periodic, so the slabs meet at TWO interfaces: of the three bond
    directions only x crosses them, and 2 of the `N_SIDE` x-planes are interface
    planes, so the unlike fraction is exactly `2 / (3 * N_SIDE)`. Comparing
    against that rather than against "big enough" is what makes the test fail if
    the counter double-counts an interface or misses one.
    """
    pos = _lattice()
    spc = _slab_labels(pos)
    a = demixing.order_parameter_of(pos, spc, BOX, DIM, 2, R_CUT)
    predicted = 1.0 - (2.0 / (3.0 * N_SIDE)) / _p_unlike(len(pos))
    assert a == pytest.approx(predicted, rel=0.01), (
        f"a fully separated configuration reads alpha = {a:.4f} against the "
        f"geometry's {predicted:.4f}"
    )


def test_random_labelling_sits_inside_its_own_band():
    pos = _lattice()
    rng = np.random.default_rng(7)
    spc = rng.permutation(_slab_labels(pos))
    mean, sd = demixing.permutation_band(
        pos, spc, BOX, DIM, 2, np.random.default_rng(11), R_CUT, n_shuffle=32
    )
    assert sd > 0, "the null has no spread, so the band cannot reject anything"
    # THE ARITHMETIC GATE. The null mean is exact by derivation, so it must sit
    # at zero within the sampling error of the shuffle mean, not merely near it.
    assert abs(mean) < 4 * sd / np.sqrt(32), (
        f"the permutation null is centred at {mean:.5f} +- {sd:.5f}, not at zero: "
        "the expected unlike-pair count and the counter disagree"
    )
    a = demixing.order_parameter_of(pos, spc, BOX, DIM, 2, R_CUT)
    assert abs(a - mean) < 4 * sd, (
        f"a randomly labelled configuration reads alpha = {a:.4f}, outside the "
        f"band {mean:.4f} +- {sd:.4f} its own relabellings occupy"
    )


def test_alternating_reads_the_floor():
    """A checkerboard has NO like contact, so it sits at the parameter's minimum,
    `1 - 1/p_unlike`, which is -1 at equal composition. The other end of the
    scale from the slabs, and equally a closed form."""
    pos = _lattice()
    spc = _checkerboard_labels(pos)
    a = demixing.order_parameter_of(pos, spc, BOX, DIM, 2, R_CUT)
    floor = 1.0 - 1.0 / _p_unlike(len(pos))
    assert a == pytest.approx(floor, rel=0.01), (
        f"a checkerboard reads alpha = {a:.4f} against the floor {floor:.4f}"
    )


def test_translation_and_label_swap_leave_it_alone():
    """PBC and label symmetry, the two ways a pair counter goes quietly wrong.

    A configuration shifted through the periodic boundary is the same
    configuration; A and B are names.
    """
    pos = _lattice()
    spc = _slab_labels(pos)
    base = demixing.order_parameter_of(pos, spc, BOX, DIM, 2, R_CUT)
    shifted = (pos + np.array([3.7, -1.1, 8.3])) % BOX
    assert demixing.order_parameter_of(
        shifted, spc, BOX, DIM, 2, R_CUT
    ) == pytest.approx(base, abs=1e-12)
    assert demixing.order_parameter_of(
        pos, 1 - spc, BOX, DIM, 2, R_CUT
    ) == pytest.approx(base, abs=1e-12)


def test_pair_counts_agree_with_a_brute_force_count():
    """The counter against the definition, on a system small enough to count by
    hand with an O(N^2) minimum-image sweep."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, BOX, size=(200, DIM))
    spc = rng.integers(0, 3, size=200)
    counts = demixing.pair_counts(pos, spc, BOX, DIM, 2.0, 3)

    dv = pos[:, None, :] - pos[None, :, :]
    dv -= BOX * np.round(dv / BOX)
    d = np.sqrt((dv**2).sum(-1))
    iu = np.triu_indices(len(pos), k=1)
    close = d[iu] <= 2.0
    a, b = spc[iu[0]][close], spc[iu[1]][close]
    brute = np.zeros((3, 3), dtype=np.int64)
    for x, y in zip(a, b):
        brute[x, y] += 1
        if x != y:
            brute[y, x] += 1
    assert counts.tolist() == brute.tolist()


def test_partial_rdfs_are_flat_and_equal_under_random_labelling():
    """Random labels cannot produce species-resolved structure.

    The strong version of the normalisation check: with labels dealt at random,
    g_AA, g_AB and g_BB must lie on top of each other, and the diagonal and the
    off-diagonal have DIFFERENT reference densities, so they only coincide if
    both are right. A single shared normalisation puts them a factor of about
    two apart while leaving both looking like a g(r).
    """
    rng = np.random.default_rng(5)
    acc = demixing.PartialRDF(BOX, DIM, 2, r_max=4.0, n_bins=40)
    for _ in range(6):
        pos = rng.uniform(0, BOX, size=(600, DIM))
        spc = rng.integers(0, 2, size=600)
        acc.accumulate(pos, spc)
    r, g = acc.finalize()
    band = (r > 1.0) & (r < 4.0)
    for a in range(2):
        for b in range(2):
            mean = g[a, b][band].mean()
            assert abs(mean - 1.0) < 0.05, (
                f"g_{a}{b} averages {mean:.3f} over the flat region; an ideal "
                "gas has g = 1 and the normalisation is what sets that"
            )


# --------------------------------------------------------- composition profile
#
# Alpha counts contacts inside `r_cut`, so it cannot tell a fluid of small like
# clusters from a box split in half: both enrich like neighbours locally. The
# profile is what separates them, and "two phases with an interface" is a claim
# about the profile rather than about alpha. These two tests are the pair that
# makes that distinction load-bearing.


def test_two_slabs_read_a_full_profile_swing():
    pos = _lattice()
    spc = _slab_labels(pos)
    amp = demixing.profile_amplitude(pos, spc, BOX, DIM, 2, n_bins=N_SIDE)
    assert amp == pytest.approx(1.0), (
        f"two pure slabs read an amplitude of {amp:.3f}; each slab is entirely "
        "one species, so the swing is the whole axis"
    )


def test_random_labels_read_almost_no_profile_swing():
    """The null, and it is not zero: with `n / n_bins` particles per slab the
    mole fraction fluctuates binomially, so the band is set by the slab
    occupancy rather than chosen."""
    pos = _lattice()
    rng = np.random.default_rng(19)
    spc = rng.permutation(_slab_labels(pos))
    amp = demixing.profile_amplitude(pos, spc, BOX, DIM, 2, n_bins=N_SIDE)
    per_slab = len(pos) / N_SIDE
    sd = np.sqrt(0.25 / per_slab)
    # the amplitude is a max-minus-min over 3 axes x N_SIDE slabs, so it sits a
    # few sd wide by construction even with no structure at all
    assert amp < 8 * sd, (
        f"randomly labelled slabs read an amplitude of {amp:.3f} against a "
        f"binomial slab sd of {sd:.3f}"
    )


def test_a_local_cluster_moves_alpha_without_moving_the_profile():
    """The case the two observables exist to tell apart.

    Labels are arranged in alternating BLOCKS one shell wide, so every slab along
    every axis holds both species in equal measure -- a flat profile -- while
    like neighbours are strongly enriched inside each block. A run reported on
    alpha alone would call this separated.
    """
    pos = _lattice()
    cell = BOX / N_SIDE
    idx = np.round(pos / cell - 0.5).astype(int)
    spc = ((idx[:, 0] // 2 + idx[:, 1] // 2 + idx[:, 2] // 2) % 2).astype(np.int64)
    alpha = demixing.order_parameter_of(pos, spc, BOX, DIM, 2, R_CUT)
    amp = demixing.profile_amplitude(pos, spc, BOX, DIM, 2, n_bins=N_SIDE)
    # "Clustered" is measured against the null this configuration itself
    # provides, not against a number picked to pass: the same positions and the
    # same label multiset, shuffled.
    _mean, sd = demixing.permutation_band(
        pos, spc, BOX, DIM, 2, np.random.default_rng(23), R_CUT, n_shuffle=24
    )
    assert alpha > 5 * sd, (
        f"blocked labels read alpha = {alpha:.3f} against a null spread of "
        f"{sd:.4f}; the blocks are not registering as local clustering at all"
    )
    assert amp < 0.55, (
        f"blocked labels read a profile amplitude of {amp:.3f}; the blocks are "
        "small compared with the box, so no slab is dominated by one species"
    )
