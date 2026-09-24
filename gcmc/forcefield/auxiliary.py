"""Auxiliary potential used for enhanced sampling.

Two distinct flavours, kept as two classes because they share nothing but a
name:

  HardCoreAux2D : an analytic pair g(r) built from a hard-disk equation of state. Used
          for 2D hard-disk aux runs. Weight of a configuration is the product
          of g(r) over pairs, optionally raised to gamma.

  HardCoreAux3D : an angular three-body weight g3(r, cos_theta) read from a tabulated
          grid and interpolated. Used for 3D hard-sphere aux runs. The first
          neighbour fixes a reference direction; every other neighbour
          contributes g3 of its distance and the angle it makes with that
          reference. The reference neighbour's coordinates are carried forward
          so the per-pair recompute can reuse the same reference.

The constant derivations and both weight expressions are transcribed verbatim
from the original so they can be characterised against it.

Named `auxiliary.py`, not `aux.py`: `AUX` is a reserved DOS device name that
Windows still enforces, so a file called `aux.py` cannot exist in a checkout on
that platform. Registry names (`"aux"`, `"aux2d"`, `"ljaux"`) and every config
key are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import BoolArray, FloatArray, IntArray
from .base import Potential, register


def aux_constants_2d(sigma: float = 1.0, eta: float = 0.69) -> dict[str, float]:
    """Coefficients of the analytic hard-disk pair correlation g(r) at packing
    fraction `eta`: the depletion branch A,B,mu,alpha (sigma <= r < r_m), the
    oscillatory structural branch C,omega,delta,kappa (r_m <= r < rc), and the
    matching point r_m. Transcribed verbatim from the original code (an
    empirical EOS-based fit); do not "simplify" the algebra, the tests
    characterise these exact floating-point values."""
    eta_2, eta_3, eta_4, eta_5 = eta**2, eta**3, eta**4, eta**5
    omega = (4.07688 + 3.53995 * eta) / sigma
    kappa = (1.19878 - 1.54059 * eta - 0.567808 * eta_2) / sigma
    alpha = (-1.69377 + 13.50042 * eta - 29.8772 * eta_2) / sigma
    r_star = 2.14098 - 1.03267 * eta + 0.320164 * eta_2 + 0.0102421 * eta_3
    g_m = 1.97652 - 6.34821 * eta + 11.53962 * eta_2 - 5.96960 * eta_3 - 3.92125 * eta_4
    mu = (
        -116.2837
        + 1218.065 * eta
        - 4784.826 * eta_2
        + 9229.068 * eta_3
        - 8862.922 * eta_4
        + 3422.542 * eta_5
    ) / sigma
    delta = -omega * sigma * r_star - np.arctan(
        ((kappa * sigma * r_star) + 1) / (omega * sigma * r_star)
    )
    C = (
        (r_star * (g_m - 1) * np.exp(kappa * sigma * r_star))
        / (np.cos((omega * sigma * r_star) + delta))
    ) / sigma
    r_m = r_star * sigma
    b_1, b_2, m_1, m_2 = -1.04191e8, 2.66813e8, 53, 56
    c = 1 / (0.75 + np.finfo(float).eps)
    g_sigma = (1 - (7 / 16) * eta + (1 / 36) * eta_2 - (2 / 21) * eta_3) / (
        (1 - eta) ** 2
    ) + (b_1 * (eta**m_1) + b_2 * (eta**m_2)) / (1 - c * eta)
    B = (
        ((r_star * g_m) - g_sigma * np.exp(mu * sigma * (r_star - 1)))
        / (np.exp(alpha * sigma * (r_star - 1)) - np.exp(mu * sigma * (r_star - 1)))
    ) / sigma
    A = (g_sigma - B) * sigma
    return {
        "A": A,
        "B": B,
        "C": C,
        "mu": mu,
        "alpha": alpha,
        "omega": omega,
        "delta": delta,
        "kappa": kappa,
        "sigma": sigma,
        "r_m": r_m,
    }


def resolve_min_g3(min_g3: float | None, n_trials: int) -> float:
    """The floor applied to g3, as a number.

    `None` means the historical default 1/n_trials, kept so every config written
    before this key existed reproduces bit-for-bit. An explicit value is used as
    given and is INDEPENDENT OF k, which is the point of the key: the default
    floor moves whenever `n_trials` moves, so the weight distribution -- and any
    gamma tuned against it -- is silently a function of the trial count. Pinning
    the floor is the first half of decoupling gamma from k (see
    docs/decisions/adaptive_gamma_plan.md, stage 1).

    One function rather than two expressions because the floor is applied in two
    places that are NOT equivalent -- `build` floors the tabulated grid before
    interpolation, `weight_bulk` / `weight_pair` floor the interpolated value --
    and the two must never be able to drift onto different constants.
    """
    if min_g3 is None:
        return 1.0 / int(n_trials)
    floor = float(min_g3)
    if not (0.0 < floor <= 1.0):
        raise ValueError(
            f"auxiliary.min_g3 must satisfy 0 < min_g3 <= 1, got {floor!r}. "
            f"Zero or negative annihilates a trial outright (and its log, which "
            f"the adaptive-gamma solver reads, is -inf); above 1 clamps almost "
            f"the whole table to one constant, which is a bias that does not "
            f"bias. The historical default is 1/n_trials."
        )
    return floor


@register("aux2d")
class HardCoreAux2D(Potential):
    """Analytic hard-disk g(r) auxiliary (2D). The per-pair weight is g(r)
    itself (optionally ^gamma), so trials are biased toward the liquid's own
    pair structure; the acceptance divides the bias back out, leaving the
    hard-disk ensemble unchanged while the proposals become smarter."""

    # the fitted g(r) constants installed by __init__ (see aux_constants_2d)
    A: float
    B: float
    C: float
    mu: float
    alpha: float
    omega: float
    delta: float
    kappa: float
    sigma: float
    r_m: float

    def __init__(
        self, diameter: float, upper_cutoff: float, params: dict[str, float]
    ) -> None:
        """Store the contact diameter, the cutoff, and every fitted constant
        from `aux_constants_2d` as attributes (A, B, C, mu, alpha, omega,
        delta, kappa, sigma, r_m)."""
        self.diameter = float(diameter)
        self.upper_cutoff = float(upper_cutoff)
        for k, v in params.items():
            setattr(self, k, v)

    @classmethod
    def build(  # type: ignore[override]  # registry factory, narrowing intended
        cls,
        diameter: float,
        upper_cutoff: float,
        eta: float = 0.69,
        sigma: float = 1.0,
        **_ignored: Any,
    ) -> HardCoreAux2D:
        """Registry constructor: fit the g(r) constants at packing fraction
        `eta` and wrap them; unknown config keys are ignored on purpose."""
        return cls(diameter, upper_cutoff, aux_constants_2d(sigma=sigma, eta=eta))

    def weight_pair(  # kwargs-absorbing variant of the base signature
        self, r: FloatArray | float, gamma: float = 1.0, **_ignored: Any
    ) -> FloatArray:
        """g(r)^gamma per pair: 0 inside the core, the depletion branch up to
        r_m, the damped oscillation up to rc, and exactly 1 beyond. Padded inf
        entries are clamped to rc first so they land on the neutral branch
        (weight 1); a 2-D input is reduced to one weight per row by product."""
        r = np.asarray(r, dtype=float)
        if r.ndim == 2:
            r = r.copy()
            r[np.isinf(r[:, 0]), 0] = self.upper_cutoff
        elif r.ndim == 1:
            r = np.minimum(r, self.upper_cutoff)

        g_r = np.ones_like(r)
        overlap = r < self.diameter
        g_r[overlap] = 0.0

        dep = (r >= self.diameter) & (r < self.r_m)
        if np.any(dep):
            r_dep = r[dep]
            g_r[dep] = (self.A / r_dep) * np.exp(self.mu * (r_dep - self.sigma)) + (
                self.B / r_dep
            ) * np.exp(self.alpha * (r_dep - self.sigma))

        stru = (r >= self.r_m) & (r < self.upper_cutoff)
        if np.any(stru):
            r_str = r[stru]
            g_r[stru] = 1 + (self.C / r_str) * np.cos(
                self.omega * r_str + self.delta
            ) * np.exp(-self.kappa * r_str)

        if gamma != 1.0:
            g_r = g_r**gamma

        return np.prod(g_r, axis=1) if g_r.ndim == 2 else g_r

    def weight_bulk(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        n_particles: int | None = None,
        choice: str | None = None,
        gamma: float = 1.0,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """First-level trial weights: identical to `weight_pair` on the padded
        neighbour rows (2D aux needs no geometry beyond the distances)."""
        return self.weight_pair(dists, gamma)


#: A marker for the g3 reader, carried into the run identity by
#: `PotentialSpec.g3_reader`. Not a config key: there is one reader and it is
#: `_NearestG3`. It exists because the reader CHANGED -- runs before it
#: interpolated the table, runs after it read the nearest tabulated point, and
#: the two give different g3 and therefore select different trials. Without a
#: marker in the identity those two would share a run directory and resume each
#: other. Bump the string if the reader ever changes again.
G3_READER = "nearest"


class _NearestG3:
    """The g3 table read at the nearest tabulated point, by integer arithmetic.

    WHY THIS EXISTS. The table lookup was the auxiliary's largest single cost.
    It used to be `scipy.interpolate.RegularGridInterpolator`, which validates
    its input, locates each query with a binary search per axis and blends four
    corners; measured on the query the first Rosenbluth level actually issues
    (about 2400 points, one per trial per neighbour) that was 271 microseconds a
    call. The grid is REGULAR -- r in steps of 0.05 from 0.025, cos theta in
    steps of 0.02 from -0.99 -- so locating a point needs no search at all:
    subtract the origin, divide by the step, round. This does that, and the same
    query costs 31 microseconds. End to end it took 14% off the cost of a move
    and added 19% to ESJD per second, on the fitted and the adaptive arm alike.

    WHY IT IS ALLOWED TO BE COARSER, which is the part that matters. g3 is a
    SELECTION BIAS and it is divided back out of the acceptance ratio. Any
    strictly positive function of the geometry leaves the sampled ensemble
    exactly unchanged, as long as the forward and reverse passes read the same
    one. So the resolution of g3 is a variance knob, on the same footing as
    `min_g3` and `pr_target`, and not an accuracy claim about the three-body
    correlation. What is NOT negotiable is that the function be deterministic
    and identical in both passes, which integer arithmetic on a fixed table is
    about as strongly as anything can be.

    The cost of the coarseness is a median relative departure from the
    interpolated value of 0.3% over the tabulated region, because the grid is
    fine (52 x 100) and g3 is smooth across a cell. Note that a COARSER TABLE
    would not be any faster: the lookup does the same arithmetic whatever the
    table's shape, so there is nothing to gain by throwing values away.

    THE DOMAIN IS UNCHANGED from the interpolator this replaced: outside the
    tabulated range the answer is 1.0, an unbiased pair, over exactly the same
    range, bounded on the node values themselves. So the change was one of
    resolution and not of WHICH pairs are biased at all, the second being a much
    larger change wearing the same clothes. A point exactly on the boundary is
    inside. `tests/unit/test_g3_lookup.py` builds the interpolator as a
    reference oracle and pins both claims.
    """

    __slots__ = ("_c0", "_dc", "_dr", "_fill", "_flat", "_nc", "_nr", "_r0")

    def __init__(
        self,
        r_nodes: FloatArray,
        cos_nodes: FloatArray,
        grid: FloatArray,
        fill_value: float = 1.0,
    ) -> None:
        if r_nodes.size < 2 or cos_nodes.size < 2:
            raise ValueError(
                f"the g3 table needs at least two nodes per axis to have a "
                f"step, got {r_nodes.size} r values and {cos_nodes.size} cos "
                f"values."
            )
        for name, nodes in (("r", r_nodes), ("cos theta", cos_nodes)):
            steps = np.diff(nodes)
            if not np.allclose(steps, steps[0], rtol=0.0, atol=1e-9 * abs(steps[0])):
                raise ValueError(
                    f"the {name} axis of the g3 table is not evenly spaced "
                    f"(steps from {steps.min()!r} to {steps.max()!r}), so the "
                    f"nearest-point lookup cannot index it by arithmetic. "
                    f"Regrid the table onto a constant step per axis, or "
                    f"restore an interpolating reader, which searches instead."
                )
        self._flat = np.ascontiguousarray(grid, dtype=float).ravel()
        self._r0 = float(r_nodes[0])
        self._dr = float(r_nodes[1] - r_nodes[0])
        self._nr = int(r_nodes.size)
        self._c0 = float(cos_nodes[0])
        self._dc = float(cos_nodes[1] - cos_nodes[0])
        self._nc = int(cos_nodes.size)
        self._fill = float(fill_value)

    def __call__(self, pts: FloatArray) -> FloatArray:
        """g3 at each (r, cos theta) row of `pts`, same shape contract as the
        interpolator this replaces: one value per row, `fill_value` where the
        row is outside the tabulated range. Kept because it is the interface the
        interpolator had and the tests compare against; the kernels call `at`.
        """
        q = np.asarray(pts, dtype=float)
        return self.at(q[:, 0], q[:, 1])

    def at(self, r: FloatArray, cos_t: FloatArray) -> FloatArray:
        """The same lookup from two SEPARATE arrays, of any matching shape.

        This is what the geometry kernels call, and the reason it exists is that
        the (r, cos theta) pair is never naturally a two-column array: the bulk
        kernel holds an (N, k-1) matrix of each and the pair kernel two vectors.
        Going through `__call__` meant a `column_stack` and a `flatten` on the
        way in and a `reshape` on the way out, three array copies per level to
        satisfy a shape contract that only the interpolator ever wanted.

        Returns an array shaped like its inputs, so the caller keeps whatever
        shape it had.
        """
        # Axis coordinates in units of the step, so the node index is just this
        # rounded. Held as floats until the very last moment, because the cast
        # is the one operation here that cannot represent an out-of-range row.
        x = (r - self._r0) / self._dr
        y = (cos_t - self._c0) / self._dc
        # The node range, inclusive, exactly as the interpolator bounds it, so
        # switching the key changes the resolution and not which pairs are
        # biased at all. A NaN coordinate compares false everywhere and lands on
        # the fill value, which is what `weight_pair` relies on for a trial with
        # no reference direction.
        inside = (x >= 0.0) & (x <= self._nr - 1) & (y >= 0.0) & (y <= self._nc - 1)
        # `+ 0.5` then floor, rather than `rint`, so the tie at a cell boundary
        # goes the same way at every index instead of to the nearest even one.
        # An outside row indexes nothing and is sent to 0 BEFORE the cast: a NaN
        # survives the arithmetic, and casting NaN to an integer is undefined
        # (in practice a huge negative index, which then raises). Inside the
        # range no clip is needed -- x <= nr - 1 floors to at most nr - 1 -- so
        # this one `where` replaces two clips as well.
        flat = np.where(inside, np.floor(x + 0.5) * self._nc + np.floor(y + 0.5), 0.0)
        return np.where(inside, self._flat[flat.astype(np.intp)], self._fill)


@register("aux3d")
class HardCoreAux3D(Potential):
    """Tabulated three-body angular auxiliary (3D). The bias of a trial is the
    product over its neighbours j >= 1 of g3(r_j, cos theta_j), where theta_j
    is the angle neighbour j makes with the NEAREST neighbour (j = 0), taken
    as the reference direction. g3 comes from a CSV grid, linearly
    interpolated; values are floored at `min_g3` so a single terrible angle
    cannot annihilate a trial outright and stall the chain. The floor defaults
    to 1/n_trials and can be pinned to a k-independent constant from the config
    (`auxiliary.min_g3`); see `resolve_min_g3`."""

    needs_geometry = True

    def __init__(
        self,
        diameter: float,
        upper_cutoff: float,
        side: float,
        n_trials: int,
        interpolator: Any,
        min_g3: float | None = None,
        g3_r_cut: float | None = None,
    ) -> None:
        """Wrap a prebuilt reader of the g3 table over (r, cos theta), which is
        a `_NearestG3`; `side`
        is needed because the angular weight minimum-images its own vectors.
        `min_g3` is the g3 floor; `None` falls back to the historical 1/n_trials
        (see `resolve_min_g3`)."""
        self.diameter = float(diameter)
        self.upper_cutoff = float(upper_cutoff)
        self.side = float(side)
        self.min_g3 = resolve_min_g3(min_g3, n_trials)
        #: Radius beyond which a neighbour contributes NOTHING to the bias, or
        #: None for the shipped behaviour, where every neighbour inside the
        #: potential's own cutoff contributes a factor.
        #
        #: WHY THIS IS A KNOB. The bias is a product over neighbours, so every
        #: pair multiplies in whether or not it carries information. Measured on
        #: `data/g3_analysis_data.csv`, the angular structure of log g3 dies out
        #: by r = 1.9: the spread over cos theta falls from 2.8 at contact to
        #: 0.26 by r = 1.975, and pairs beyond r = 1.8 hold about 4% of the
        #: total variance while being roughly two thirds of the pairs. Those
        #: pairs therefore add cost, and they add a dependence on how MANY
        #: neighbours a trial has, without adding ranking. Cutting them is the
        #: one change that made the bias rank positions measurably better in
        #: `scripts/aux_variants.py`, which is what a value here should be
        #: chosen with rather than by argument.
        self.r_cut = None if g3_r_cut is None else float(g3_r_cut)
        self._interp = interpolator

    @classmethod
    def build(  # type: ignore[override]  # registry factory, narrowing intended
        cls,
        diameter: float,
        upper_cutoff: float,
        side: float,
        n_trials: int,
        g3_csv: str,
        min_g3: float | None = None,
        g3_r_cut: float | None = None,
        **_ignored: Any,
    ) -> HardCoreAux3D:
        """Load the g3 grid from `g3_csv` (columns: r, _, cos theta, _, g3),
        floor it at `min_g3` (default 1/n_trials), and wrap it in the
        nearest-point reader. Weight is 1 outside the tabulated range, so
        out-of-range pairs are unbiased."""
        floor = resolve_min_g3(min_g3, n_trials)
        data = np.loadtxt(g3_csv, delimiter=",", skiprows=1)
        r_vals, cos_vals, g3_vals = data[:, 0], data[:, 2], data[:, 4]
        r_u, cos_u = np.unique(r_vals), np.unique(cos_vals)
        order = np.lexsort((cos_vals, r_vals))
        grid = g3_vals[order].reshape(len(r_u), len(cos_u))
        grid = np.where(grid < floor, floor, grid)
        reader = _NearestG3(r_u, cos_u, grid, fill_value=1.0)
        return cls(diameter, upper_cutoff, side, n_trials, reader, floor, g3_r_cut)

    def _g3_grid_bulk(
        self,
        dists: FloatArray,
        coords: FloatArray | None,
        idxs: IntArray | None,
        kdtree: Any,
    ) -> tuple[FloatArray | None, FloatArray]:
        """(floored g3 per (trial, neighbour j >= 1), ref_coords), with NO
        exponent and no product taken.

        This is the whole three-body geometry of the first Rosenbluth level: the
        nearest neighbour fixes the reference direction, every further neighbour
        contributes g3 of its distance and its angle to that reference, and the
        table value is floored at `min_g3`. `weight_bulk` finishes the job by
        raising to gamma and multiplying along the neighbour axis;
        `weight_bulk_split` finishes it by taking logs instead.

        A `None` first element is the EMPTY ENVIRONMENT: no neighbours, so no
        triplets, so the product below would run over an empty set. The two
        callers say what an empty product means in their own currency (1, and
        log 1 = 0) rather than having this function invent a shape for it.

        Factored out rather than copied because the two callers must agree
        bit-for-bit on the floor and on the angle convention; a copy is a place
        for them to drift.
        """
        # needs_geometry contract: these are only optional in the shared base
        # signature; every geometry-aware call site supplies them.
        assert coords is not None and idxs is not None
        # `k` is no longer needed: nothing reshapes by it any more.
        N = dists.shape[0]
        if kdtree is None:
            # EMPTY ENVIRONMENT. `metropolis_dynamics.advance` passes `tree = cl if
            # cl.n_active else None`, so an empty exclusion set -- no fluid left
            # and no frozen matrix -- arrives here as None with fully padded
            # neighbour rows. Every other weigher handles that already: an
            # infinite distance contributes zero energy and unit weight. This one
            # asserted instead, which is why a QA or low-density pmf run died on
            # its first window rather than on anything to do with the auxiliary.
            #
            # The no-neighbour limit is not a special case of the physics, it IS
            # the physics: this bias is a THREE-BODY factor over (r, cos theta)
            # triplets, and with no neighbours there is no triplet. The product
            # below runs over an empty set, and an empty product is 1.
            #
            # The reference neighbour is undefined, and NaN says so rather than a
            # placeholder saying something false. Zero would be worse than
            # undefined: `weight_pair` builds `vec_BA = ref - trial`, so a zero
            # reference is not neutral -- it is a real direction pointing at the
            # origin, and the per-pair recompute would apply a definite bias
            # derived from nothing. `weight_pair` reads the NaN and takes the
            # same no-bias branch.
            return None, np.full((N, coords.shape[1]), np.nan)
        # FUSED, and deliberately so. The arrays here are small -- a level pool
        # is about ten surviving trials against fifteen neighbour columns -- so
        # the cost of this function is numpy DISPATCH and not arithmetic: the
        # same pairs, batched into one large call, cost a sixth as much per
        # trial. Every array operation removed below is therefore worth about as
        # much as every other one, whatever its arithmetic weight, and the
        # rewrite is bit-identical to the version it replaced over 400 captured
        # argument sets from a live run. It keeps `np.round(x / side)` rather
        # than a multiply by a precomputed reciprocal for exactly that reason:
        # 1/20 is not representable, so the faster form would move the last bit
        # of every minimum-imaged component.
        n_data = kdtree.data.shape[0]
        neigh = kdtree.data[np.minimum(idxs, n_data - 1)]  # (N, k, 3)
        deltas = neigh - coords[:, None, :]
        deltas -= self.side * np.round(deltas / self.side)
        vec_BA = deltas[:, 0, :]
        vec_BC = deltas[:, 1:, :]
        # `einsum` in place of a broadcast product summed over the last axis:
        # the product materialised an (N, k-1, 3) temporary to throw it away.
        # `norm_BA` likewise, rather than `np.linalg.norm`, which is a Python
        # wrapper around several of these.
        dot = np.einsum("ij,ikj->ik", vec_BA, vec_BC)
        norm_BA = np.sqrt(np.einsum("ij,ij->i", vec_BA, vec_BA))
        norm_BC = dists[:, 1:]
        cos_t = np.clip(dot / (norm_BA[:, None] * norm_BC), -1.0, 1.0)
        # Straight to the reader in the shape it already has: no column_stack,
        # no flatten, no reshape.
        g3 = self._interp.at(norm_BC, cos_t)
        np.maximum(g3, self.min_g3, out=g3)
        if self.r_cut is not None:
            # A neutralised pair contributes exactly 1, which is the identity of
            # the product and of the sum of logs alike, so the two callers need
            # no branch of their own. Written as a `where` rather than as a
            # compressed array because the row alignment with `dists` is what
            # every caller downstream indexes by.
            g3 = np.where(norm_BC < self.r_cut, g3, 1.0)
        return g3, neigh[:, 0, :]

    def weight_bulk(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        n_particles: int | None = None,
        choice: str | None = None,
        gamma: float = 1.0,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Returns (weights, ref_coords). The caller stores ref_coords for the
        matching per-pair recompute. needs_geometry is True, so callers expect
        the tuple here rather than a bare array."""
        g3, ref_coords = self._g3_grid_bulk(dists, coords, idxs, kdtree)
        if g3 is None:
            # empty product over an empty neighbour set (see _g3_grid_bulk)
            return np.ones(dists.shape[0]), ref_coords
        if gamma != 1.0:
            g3 = g3**gamma
        return np.prod(g3, axis=1), ref_coords

    def weight_bulk_split(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        **_ignored: Any,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(base, log_g3, ref_coords) for the first level, gamma-free.

        `weight_bulk` returns one number per trial and there is no way to get
        gamma back out of it. The adaptive solver needs the two factors apart:
        the part gamma does not touch, and log of the part it does, so that a
        candidate gamma costs one `exp` rather than a re-walk of the geometry.
        Here `base` is all ones -- this potential IS the bias, it has no
        Boltzmann factor of its own -- and it is returned anyway so the pair has
        the same shape as `LJAux3D`'s and the caller never branches on which
        potential it holds.

        `log_g3` is the log of the TRIAL's whole bias, i.e. summed over that
        trial's neighbours, because the bias is their product. The sum runs
        along the neighbour axis, whose order is fixed by the neighbour query;
        the pool permutation the engine applies reorders trials (rows), which a
        row-wise reduction cannot see. So this is permutation-invariant without
        needing the sorted reduction the solver itself uses.
        """
        g3, ref_coords = self._g3_grid_bulk(dists, coords, idxs, kdtree)
        n = dists.shape[0]
        if g3 is None:
            # empty product = 1, so its log is 0. These rows STAY: an unbiased
            # trial is still a trial and still carries its share of the
            # participation ratio.
            return np.ones(n), np.zeros(n), ref_coords
        return np.ones(n), np.sum(np.log(g3), axis=1), ref_coords

    def _g3_pair(
        self,
        r: FloatArray | float,
        trial_coords: FloatArray | None,
        ref_coords: FloatArray | None,
        chosen_coord: FloatArray | None,
    ) -> tuple[FloatArray | None, BoolArray, int]:
        """(floored g3 per survivor, no-reference mask, row count) for the
        deeper-level refresh against a just-chosen trial, with no exponent
        applied. `r` is SQUARED distances, as at every depth > 0 call site.

        A `None` first element means EVERY row lacks a reference neighbour, so
        no row carries a bias at all; the callers turn that into unit weights or
        zero logs. When only SOME rows lack one, the geometry below still runs
        for all of them -- their NaN direction reaches the interpolator and comes
        back NaN -- and `no_ref` is what masks the result afterwards. That is the
        pre-existing behaviour, preserved deliberately: the mask, not the
        arithmetic, is what makes those rows unbiased.

        Same factoring argument as `_g3_grid_bulk`: one angle convention, one
        floor, two callers.
        """
        # needs_geometry contract (see weight_bulk)
        assert trial_coords is not None and ref_coords is not None
        assert chosen_coord is not None
        sq = np.asarray(r, dtype=float)
        # No reference neighbour: `weight_bulk` marks the empty-environment case
        # with NaN. Same reasoning as there -- no triplet, so no three-body bias,
        # so unit weight. The ALL-rows case short-circuits, because with nothing
        # to interpolate there is no point building a NaN (r, cos) grid at all.
        no_ref = np.asarray(
            ~np.isfinite(np.atleast_2d(ref_coords)).all(axis=1), dtype=bool
        )
        n = int(np.atleast_1d(sq).shape[0])
        if no_ref.all():
            return None, no_ref, n
        # Fused for the reason given in `_g3_grid_bulk`, and bit-identical to
        # the version it replaced on the same captured argument sets. This is
        # the hotter of the two: it runs once per survivor per level, and a
        # level pool is small enough that dispatch dominates.
        vec_BA = ref_coords - trial_coords
        vec_BA -= self.side * np.round(vec_BA / self.side)
        vec_BC = chosen_coord - trial_coords
        vec_BC -= self.side * np.round(vec_BC / self.side)
        dot = np.einsum("ij,ij->i", vec_BA, vec_BC)
        norm_BA = np.sqrt(np.einsum("ij,ij->i", vec_BA, vec_BA))
        norm_BC = np.sqrt(sq)
        cos_t = np.clip(dot / (norm_BA * norm_BC), -1.0, 1.0)
        g3 = self._interp.at(norm_BC, cos_t)
        np.maximum(g3, self.min_g3, out=g3)
        if self.r_cut is not None:
            g3 = np.where(norm_BC < self.r_cut, g3, 1.0)
        return g3, no_ref, n

    def weight_pair(  # kwargs-absorbing variant of the base signature
        self,
        r: FloatArray | float,
        gamma: float = 1.0,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        box: FloatArray | None = None,
        **_ignored: Any,
    ) -> FloatArray:
        """`r` here is the squared distances to the newly chosen trial."""
        g3, no_ref, n = self._g3_pair(r, trial_coords, ref_coords, chosen_coord)
        if g3 is None:  # no reference neighbour anywhere, so no bias anywhere
            return np.ones(n)
        if gamma != 1.0:
            g3 = g3**gamma
        # rows whose reference neighbour was undefined carry no bias
        return np.where(no_ref, 1.0, g3)

    def pairwise_split(
        self,
        sq: FloatArray,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        **_ignored: Any,
    ) -> tuple[FloatArray, FloatArray]:
        """(base, log_g3) for one deeper-level refresh, gamma-free. The split
        twin of `pairwise`; see `weight_bulk_split` for why `base` is ones here.

        The no-reference rows are the thing to be careful about. They carry bias
        1, so their log is 0, and they must arrive in the returned array as
        zeros rather than being filtered out: the participation ratio is computed
        over the LEVEL POOL, and a trial with no reference neighbour is a member
        of that pool with an honest weight. Drop it and the solver reads a
        smaller pool than the sampler selects from -- silent at low density,
        wrong at high, which is the worst possible failure shape.
        """
        g3, no_ref, n = self._g3_pair(sq, trial_coords, ref_coords, chosen_coord)
        if g3 is None:
            return np.ones(n), np.zeros(n)
        # `where` before `log`, not after: a no-reference row's g3 is NaN (its
        # direction was NaN), and log(NaN) is NaN, which would poison the sum.
        # Substituting the bias it actually has (1) makes its log exactly 0.0.
        return np.ones(n), np.log(np.where(no_ref, 1.0, g3))

    def pairwise(
        self,
        sq: FloatArray,
        gamma: float = 1.0,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        **_ignored: Any,
    ) -> FloatArray:
        """Deeper-level refresh: unlike the scalar-distance base class this
        needs the survivors' coordinates and their carried reference
        neighbours, so it forwards the squared distances straight to
        `weight_pair` instead of square-rooting them first."""
        return self.weight_pair(
            sq,
            gamma,
            trial_coords=trial_coords,
            ref_coords=ref_coords,
            chosen_coord=chosen_coord,
        )


@register("ljaux3d")
class LJAux3D(Potential):
    """Lennard-Jones main potential with the 3D PMF auxiliary as a selection
    bias: the per-trial weight is exp(-beta * u_LJ) * g3(r, cos)^gamma.

    The acceptance must divide out only the *bias* of the chosen/removed
    molecule, never its Boltzmann factor (CBMC cancels that against the
    selection probability). With combined weights the engine achieves this by
    using delta_E = (log combined chosen) - (log combined removed) plus an
    explicit +beta*(U_inserted - U_removed) pair-energy term: the Boltzmann
    parts cancel exactly, leaving log(bias_new) - log(bias_old). At one trial
    (metropolis) the bias cancels identically and the mode reduces to plain
    LJ. The long-range LJ tail stays a move-level term, as for pure LJ.
    """

    needs_geometry = True

    def __init__(self, lj: Any, aux: HardCoreAux3D) -> None:
        """Compose a plain LJ (no tail: the engine applies the tail at move
        level) with an HardCoreAux3D bias; core diameter and cutoff follow the LJ."""
        self.lj = lj
        self.aux = aux
        self.diameter = lj.diameter
        self.upper_cutoff = lj.upper_cutoff

    @classmethod
    def build(  # type: ignore[override]  # registry factory, narrowing intended
        cls,
        beta: float,
        upper_cutoff: float,
        diameter: float,
        side: float,
        n_trials: int,
        g3_csv: str,
        min_g3: float | None = None,
        g3_r_cut: float | None = None,
        **_ignored: Any,
    ) -> LJAux3D:
        """Registry constructor: build both halves from the shared config keys
        (the LJ deliberately without `volume`, see __init__). `min_g3` belongs
        to the aux half only; the LJ has no floor."""
        from .lennard_jones import LennardJones

        lj = LennardJones(
            beta=beta, upper_cutoff=upper_cutoff, diameter=diameter, volume=None
        )
        aux = HardCoreAux3D.build(
            diameter=diameter,
            upper_cutoff=upper_cutoff,
            side=side,
            n_trials=n_trials,
            g3_csv=g3_csv,
            min_g3=min_g3,
            g3_r_cut=g3_r_cut,
        )
        return cls(lj, aux)

    def weight_bulk(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        n_particles: int | None = None,
        choice: str | None = None,
        gamma: float = 1.0,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """(combined weights, ref_coords): the per-trial LJ Boltzmann factor
        times the g3 bias, plus the reference neighbours the recompute needs.
        The tuple return mirrors HardCoreAux3D (needs_geometry is True)."""
        lj_w = self.lj.weight_pair(dists)
        aux_w, ref = self.aux.weight_bulk(
            dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=gamma
        )
        return lj_w * aux_w, ref

    def weight_bulk_split(
        self,
        dists: FloatArray,
        *,
        coords: FloatArray | None = None,
        idxs: IntArray | None = None,
        kdtree: Any = None,
        **_ignored: Any,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(lj_w, log_g3, ref_coords): the combined weight of `weight_bulk`,
        taken apart at the seam gamma sits on.

        This is the split that matters, because here the two factors are
        genuinely different physics: `lj_w` is the Boltzmann factor, which the
        acceptance cancels against the selection probability and gamma must
        never touch, and `g3` is the bias, which gamma tunes and the acceptance
        divides back out. `weight_bulk` multiplies them and the seam is gone.

        Composing them back -- `lj_w * exp(gamma * log_g3)` -- reproduces
        `weight_bulk(gamma)` to round-off, which is gate S2.
        """
        lj_w = self.lj.weight_pair(dists)
        _base, log_g3, ref = self.aux.weight_bulk_split(
            dists, coords=coords, idxs=idxs, kdtree=kdtree
        )
        # `_base` is the aux's own ones-array; the LJ Boltzmann factor is the
        # base on this path and replaces it.
        return lj_w, log_g3, ref

    def weight_pair(  # kwargs-absorbing variant of the base signature
        self,
        r: FloatArray | float,
        gamma: float = 1.0,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        box: FloatArray | None = None,
        **_ignored: Any,
    ) -> FloatArray:
        """Combined deeper-level factor for each survivor against the chosen
        trial. NOTE the convention shift: `r` is SQUARED distances here (this
        is only called from the depth > 0 refresh chain, which carries d^2)."""
        # r is squared distances (depth > 0 chain refresh); product of the LJ
        # Boltzmann factor and the g3 bias of the new pair
        lj_w = self.lj.weight_pair(np.sqrt(np.asarray(r, dtype=float)))
        g3 = self.aux.weight_pair(
            r,
            gamma,
            trial_coords=trial_coords,
            ref_coords=ref_coords,
            chosen_coord=chosen_coord,
        )
        return lj_w * g3

    def pairwise(self, sq: FloatArray, gamma: float = 1.0, **kw: Any) -> FloatArray:
        """Forward the squared distances unchanged (see weight_pair note)."""
        return self.weight_pair(sq, gamma, **kw)

    def pairwise_split(
        self,
        sq: FloatArray,
        *,
        trial_coords: FloatArray | None = None,
        ref_coords: FloatArray | None = None,
        chosen_coord: FloatArray | None = None,
        **_ignored: Any,
    ) -> tuple[FloatArray, FloatArray]:
        """(lj_w, log_g3) for one deeper-level refresh; the split twin of
        `pairwise`. The LJ factor is formed from the same
        `weight_pair(sqrt(sq))` expression `weight_pair` uses, so composition
        differs from the combined path only by the exp/log round-trip."""
        lj_w = self.lj.weight_pair(np.sqrt(np.asarray(sq, dtype=float)))
        _base, log_g3 = self.aux.pairwise_split(
            sq,
            trial_coords=trial_coords,
            ref_coords=ref_coords,
            chosen_coord=chosen_coord,
        )
        return lj_w, log_g3
