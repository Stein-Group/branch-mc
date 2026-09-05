"""Species-resolved structure: the demixing order parameter and the partial
radial distribution functions.

WHAT THIS IS FOR. The demixing runs, and the claim
being tested is that a mixture with a strongly weakened cross term
separates, and that the channel sampler reaches the separated state from a mixed
start in fewer cycles than metropolis does. Both halves of that need an
observable that is a NUMBER PER CONFIGURATION, so it can be plotted against
cycles; a picture of the box cannot be a convergence curve.

THE ORDER PARAMETER, and why this one.

    alpha(r_cut) = 1 - n_unlike / E[n_unlike]

over the pairs closer than `r_cut`, where `E[n_unlike]` is the count expected if
the same labels were dealt out at random over the same configuration:

    E[n_unlike] = n_pairs * (1 - sum_s n_s (n_s - 1) / (N (N - 1)))

That expectation is EXACT rather than approximate -- it is the marginal
probability that a random permutation of the label multiset gives one pair two
different labels, and linearity does the rest -- so the null is `alpha = 0` by
construction and not by assumption. The value the parameter would have to see in
order to fail its null is therefore reachable and known before any run happens,
which is the property this project asks of a gate.

    alpha = 0     labels are arranged as randomly as the positions allow
    alpha -> 1    no unlike pair is within r_cut: the species have separated
    alpha < 0     unlike neighbours are enriched: alternation, not demixing

It is label-symmetric (swapping A and B leaves it alone), defined for any number
of species, and unaffected by the packing: the positions enter only through
which pairs exist, and both the count and its expectation are taken over that
same pair set. A parameter built on the count of like neighbours instead would
move with density even at fixed composition.

`r_cut` is an argument with no default that is right for every model. The first
coordination shell is the sensitive choice -- 1.5 sigma for a Lennard-Jones
fluid, being the first minimum of g(r) at liquid density -- and the potential
cutoff is the conservative one, which dilutes the signal by averaging the second
and third shells into it. Whichever is used belongs in the reported result,
which is why `SHELL_R_CUT` is a named constant rather than a bare literal.

THE FINITE-SAMPLE BAND. The null MEAN is exact; its SPREAD is not analytic,
because pairs sharing a particle are not independent. `permutation_band` gets it
by relabelling the actual configuration: same positions, same label multiset,
shuffled assignment. That is the exact null distribution of this statistic on
this configuration, so a tolerance derived from it is a measurement rather than
a guess -- and it collapses correctly at small N, where a hand-written binomial
error bar would quietly be wrong.

WHY NOT `rdf.py`. That module computes the quenched-annealed benchmark's two
correlations, fluid-fluid and fluid-matrix, for a single-component fluid in a
cubic box, with a cKDTree and a density bookkeeping written around a frozen
obstacle set. Its `g11` is the one-species case of `partial_rdf` here, so the two
DO overlap and the overlap is real duplication -- recorded here rather than
pretended away. It is not merged because merging means
moving the matrix half onto a species-resolved base, which is a semantic merge
with its own reduction gate, and what was needed was the mixture observable
rather than a rewrite of the benchmark. Neighbours here come from `CellList`, not from a
cKDTree, so the merge is toward this module rather than away from it.
"""

from __future__ import annotations

import numpy as np

from .._types import RNG, FloatArray, IntArray
from ..space.neighbors import CellList

#: First-shell cutoff for a Lennard-Jones fluid in reduced units: the first
#: minimum of g(r) sits near 1.5 sigma at liquid density, so a neighbour inside
#: this radius is in contact rather than merely nearby. Stated once here so a
#: run, a test and a figure cannot each pick their own.
SHELL_R_CUT = 1.5


def pair_counts(
    positions: FloatArray,
    species: IntArray,
    box: float | FloatArray,
    dim: int,
    r_cut: float,
    n_species: int,
) -> IntArray:
    """Unordered pair counts by species pair, within `r_cut`, minimum image.

    Returns a symmetric `(T, T)` integer array whose `[a][b]` entry is the number
    of unordered pairs with one particle of type `a` and one of type `b`; the
    diagonal counts like-like pairs once each, not twice.

    Uses the package's own `CellList`, which is what the engine reads, so the
    neighbourhood this measures is the neighbourhood the sampler saw.
    """
    positions = np.asarray(positions, float).reshape(-1, dim)
    species = np.asarray(species, dtype=np.int64).reshape(-1)
    if len(species) != len(positions):
        raise ValueError(
            f"{len(positions)} positions against {len(species)} species labels"
        )
    counts = np.zeros((n_species, n_species), dtype=np.int64)
    if len(positions) < 2:
        return counts

    cl = CellList.from_positions(positions, box, dim, r_cut, species=species)
    slots = cl.active_slots()
    types = cl.species_at(slots)
    coords = cl.coords
    for i, slot in enumerate(slots):
        nb, _d = cl.neighbours_within(coords[slot], r_cut)
        nb = nb[nb != slot]
        if nb.size == 0:
            continue
        other = cl.species_at(nb)
        mine = int(types[i])
        binned = np.bincount(other, minlength=n_species)
        counts[mine, :] += binned
    # Every pair was seen from both ends, and the two ends land in DIFFERENT
    # entries when the types differ: an (a, b) pair increments [a][b] once and
    # [b][a] once, so the off-diagonal is already the unordered count. A like
    # pair increments [a][a] twice. Only the diagonal is halved, and the halving
    # is exact because that count is even by construction.
    diag = np.diag_indices(n_species)
    assert not (counts[diag] % 2).any(), (
        "a like-like count is odd, so a pair was seen from one end only"
    )
    counts[diag] //= 2
    assert (counts == counts.T).all()
    return counts


def _expected_unlike(n_by_species: IntArray | list[int], n_pairs: int) -> float:
    """Unlike-pair count expected when the same labels are dealt at random over
    the same pair set.

    Exact, not asymptotic: for one pair the probability of two different labels
    under a uniform random permutation of the label multiset is
    `1 - sum_s n_s (n_s - 1) / (N (N - 1))`, and expectations add regardless of
    the dependence between pairs.
    """
    n_s = np.asarray(n_by_species, dtype=float)
    n = float(n_s.sum())
    if n < 2:
        return 0.0
    p_like = float((n_s * (n_s - 1.0)).sum() / (n * (n - 1.0)))
    return n_pairs * (1.0 - p_like)


def order_parameter(counts: IntArray, n_by_species: IntArray | list[int]) -> float:
    """The demixing order parameter from a pair-count matrix.

    Zero for random labelling, one when no unlike pair is in contact, negative
    when unlike contacts are enriched. Returns 0.0 when the expectation is zero
    -- a single-species configuration, or one with no pairs at all -- because
    there is nothing there to be non-random.
    """
    counts = np.asarray(counts)
    total = int(np.triu(counts).sum())
    unlike = total - int(np.trace(counts))
    expected = _expected_unlike(n_by_species, total)
    if expected <= 0.0:
        return 0.0
    return 1.0 - unlike / expected


def order_parameter_of(
    positions: FloatArray,
    species: IntArray,
    box: float | FloatArray,
    dim: int,
    n_species: int,
    r_cut: float = SHELL_R_CUT,
) -> float:
    """`order_parameter` straight from a configuration."""
    counts = pair_counts(positions, species, box, dim, r_cut, n_species)
    n_by = np.bincount(np.asarray(species, dtype=np.int64), minlength=n_species)
    return order_parameter(counts, n_by)


def permutation_band(
    positions: FloatArray,
    species: IntArray,
    box: float | FloatArray,
    dim: int,
    n_species: int,
    rng: RNG,
    r_cut: float = SHELL_R_CUT,
    n_shuffle: int = 64,
) -> tuple[float, float]:
    """(mean, sd) of the order parameter under random relabelling of THIS
    configuration.

    The positions and the label multiset are held fixed and only the assignment
    is shuffled, so this is the exact null distribution of the statistic for the
    system in hand. The mean returned should sit at zero to within its own sd --
    if it does not, `_expected_unlike` and the counter disagree about what a pair
    is, and the band is the thing that would catch it.

    Costs one pair pass per shuffle, so it is an analysis-time measurement, not
    something to call per window.
    """
    species = np.asarray(species, dtype=np.int64).reshape(-1)
    n_by = np.bincount(species, minlength=n_species)
    vals = np.empty(n_shuffle)
    for i in range(n_shuffle):
        shuffled = rng.permutation(species)
        counts = pair_counts(positions, shuffled, box, dim, r_cut, n_species)
        vals[i] = order_parameter(counts, n_by)
    return float(vals.mean()), float(vals.std(ddof=1))


def composition_profile(
    positions: FloatArray,
    species: IntArray,
    box: float | FloatArray,
    dim: int,
    n_species: int,
    axis: int,
    n_bins: int = 16,
) -> FloatArray:
    """Mole fraction of each species in slabs along one axis: `(n_bins, T)`.

    The SECOND observable a demixing run needs, and it answers a different
    question from `order_parameter`. Alpha is local -- it counts contacts inside
    `r_cut` -- so a fluid of small like-clusters and a fluid split into two halves
    can read the same alpha. Only a profile separates them, and "two phases with
    an interface" is a claim about the profile.

    Empty bins come back as NaN rather than as zero: a slab with no particles has
    no composition, and zero would read as "all of the other species".
    """
    positions = np.asarray(positions, float).reshape(-1, dim)
    species = np.asarray(species, dtype=np.int64).reshape(-1)
    side = float(np.asarray(box, float).reshape(-1)[axis % np.size(box)])
    idx = np.floor((positions[:, axis] % side) / side * n_bins).astype(int) % n_bins
    total = np.bincount(idx, minlength=n_bins).astype(float)
    out = np.full((n_bins, n_species), np.nan)
    for s in range(n_species):
        here = np.bincount(idx[species == s], minlength=n_bins).astype(float)
        np.divide(here, total, out=out[:, s], where=total > 0)
    return out


def profile_amplitude(
    positions: FloatArray,
    species: IntArray,
    box: float | FloatArray,
    dim: int,
    n_species: int,
    n_bins: int = 16,
) -> float:
    """Largest slab-to-slab swing in the first species' mole fraction, over all
    axes: 0 for a uniform mixture, 1 for two pure slabs.

    Reported per frame so "did it separate macroscopically" is a number that can
    be plotted against cycles, next to the local one. Taking the maximum over
    axes rather than a chosen axis is deliberate: nothing in a periodic cubic box
    picks out a direction, so which axis the interface forms along is an outcome
    of the chain and not an input.

    ONE ARTEFACT TO KNOW ABOUT, because it shows up in the first frame of every
    run started from a lattice: when `n_bins` does not divide the lattice's site
    count along an axis, the bins alias and some collect fewer sites than others,
    which reads as an amplitude of a few tenths on a configuration that is
    perfectly mixed. It decays within a window or two as the lattice melts, and
    it is a property of the binning rather than of the fluid -- compare against
    `order_parameter`, which sees the same frame as random (alpha ~ 0).
    """
    best = 0.0
    for axis in range(dim):
        prof = composition_profile(
            positions, species, box, dim, n_species, axis, n_bins
        )[:, 0]
        if np.isnan(prof).all():
            continue
        best = max(best, float(np.nanmax(prof) - np.nanmin(prof)))
    return best


class PartialRDF:
    """Streaming species-resolved g_ab(r) over a trajectory.

    One `(T, T, n_bins)` histogram, accumulated frame by frame and normalised
    once at the end against the ideal-mixture shell count, so memory is fixed
    whatever the trajectory length -- the same arrangement `rdf.RDFAccumulator`
    uses for the single-component case.

    The normalisation differs between the diagonal and the off-diagonal and
    getting it wrong is invisible in the shape: `g_ab` for `a != b` divides by
    `N_a * rho_b`, while `g_aa` divides by `N_a * rho_a * (N_a - 1) / N_a`,
    since a particle is not its own neighbour. At equal composition the two
    differ by a factor near two, so a single normalisation would put the
    diagonal and the off-diagonal on different vertical scales while both still
    tended to a plateau, which reads as structure.
    """

    def __init__(
        self,
        box: float | FloatArray,
        dim: int,
        n_species: int,
        r_max: float | None = None,
        n_bins: int = 120,
    ) -> None:
        """Fixed binning `[0, r_max)`; `r_max` defaults to half the shortest box
        side, the largest radius the minimum image resolves."""
        self.box = np.asarray(box, float) * np.ones(dim)
        self.dim = int(dim)
        self.n_species = int(n_species)
        self.r_max = float(r_max) if r_max is not None else float(self.box.min()) / 2.0
        self.n_bins = int(n_bins)
        self.edges = np.linspace(0.0, self.r_max, self.n_bins + 1)
        self.r = 0.5 * (self.edges[:-1] + self.edges[1:])
        if self.dim == 3:
            self.shell_vol = (
                (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
            )
        else:
            self.shell_vol = np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        self.volume = float(np.prod(self.box))
        self._h = np.zeros((n_species, n_species, self.n_bins))
        self._n_frames = 0
        self._sum_n = np.zeros(n_species)

    def accumulate(self, positions: FloatArray, species: IntArray) -> None:
        """Add one configuration to the histograms."""
        positions = np.asarray(positions, float).reshape(-1, self.dim)
        species = np.asarray(species, dtype=np.int64).reshape(-1)
        n_by = np.bincount(species, minlength=self.n_species)
        self._sum_n += n_by
        self._n_frames += 1
        if len(positions) < 2:
            return
        cl = CellList.from_positions(
            positions, self.box, self.dim, self.r_max, species=species
        )
        slots = cl.active_slots()
        coords = cl.coords
        types = cl.species_at(slots)
        for i, slot in enumerate(slots):
            nb, d = cl.neighbours_within(coords[slot], self.r_max)
            keep = nb != slot
            nb, d = nb[keep], d[keep]
            if nb.size == 0:
                continue
            other = cl.species_at(nb)
            mine = int(types[i])
            for b in range(self.n_species):
                sel = d[other == b]
                if sel.size:
                    hist, _ = np.histogram(sel, bins=self.edges)
                    self._h[mine, b] += hist

    def finalize(self) -> tuple[FloatArray, FloatArray]:
        """Return `(r, g)` with `g` of shape `(T, T, n_bins)`, zero where no data.

        Each entry counts ORDERED neighbours (every pair was seen from both
        ends), which is why the diagonal's reference uses `N_a - 1` rather than
        halving anything.
        """
        g = np.zeros_like(self._h)
        if not self._n_frames:
            return self.r, g
        mean_n = self._sum_n / self._n_frames
        rho = mean_n / self.volume
        for a in range(self.n_species):
            for b in range(self.n_species):
                if mean_n[a] <= 0 or mean_n[b] <= 0:
                    continue
                # neighbours of type b seen from each a; a particle is not its
                # own neighbour, so the like-like reference loses one
                rho_b = rho[b] * ((mean_n[b] - 1.0) / mean_n[b] if a == b else 1.0)
                norm = self._n_frames * mean_n[a] * rho_b * self.shell_vol
                g[a, b] = np.divide(
                    self._h[a, b], norm, out=np.zeros(self.n_bins), where=norm > 0
                )
        return self.r, g
