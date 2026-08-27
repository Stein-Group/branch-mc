"""Auxiliary potential used for enhanced sampling.

Two distinct flavours, kept as two classes because they share nothing but a
name:

  Aux2D : an analytic pair g(r) built from a hard-disk equation of state. Used
          for 2D hard-disk aux runs. Weight of a configuration is the product
          of g(r) over pairs, optionally raised to gamma.

  Aux3D : an angular three-body weight g3(r, cos_theta) read from a tabulated
          grid and interpolated. Used for 3D hard-sphere aux runs. The first
          neighbour fixes a reference direction; every other neighbour
          contributes g3 of its distance and the angle it makes with that
          reference. The reference neighbour's coordinates are carried forward
          so the per-pair recompute can reuse the same reference.

The constant derivations and both weight expressions are transcribed verbatim
from the original so they can be characterised against it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .._types import FloatArray, IntArray
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


@register("aux2d")
class Aux2D(Potential):
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
    ) -> Aux2D:
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


@register("aux3d")
class Aux3D(Potential):
    """Tabulated three-body angular auxiliary (3D). The bias of a trial is the
    product over its neighbours j >= 1 of g3(r_j, cos theta_j), where theta_j
    is the angle neighbour j makes with the NEAREST neighbour (j = 0), taken
    as the reference direction. g3 comes from a CSV grid, linearly
    interpolated; values are floored at 1/n_trials so a single terrible angle
    cannot annihilate a trial outright and stall the chain."""

    needs_geometry = True

    def __init__(
        self,
        diameter: float,
        upper_cutoff: float,
        side: float,
        n_trials: int,
        interpolator: Any,
    ) -> None:
        """Wrap a prebuilt RegularGridInterpolator over (r, cos theta); `side`
        is needed because the angular weight minimum-images its own vectors,
        and `n_trials` sets the g3 floor 1/n_trials."""
        self.diameter = float(diameter)
        self.upper_cutoff = float(upper_cutoff)
        self.side = float(side)
        self.min_g3 = 1.0 / int(n_trials)
        self._interp = interpolator

    @classmethod
    def build(  # type: ignore[override]  # registry factory, narrowing intended
        cls,
        diameter: float,
        upper_cutoff: float,
        side: float,
        n_trials: int,
        g3_csv: str,
        **_ignored: Any,
    ) -> Aux3D:
        """Load the g3 grid from `g3_csv` (columns: r, _, cos theta, _, g3),
        floor it at 1/n_trials, and build the linear interpolator with weight 1
        outside the tabulated range (out-of-range pairs are unbiased)."""
        data = np.loadtxt(g3_csv, delimiter=",", skiprows=1)
        r_vals, cos_vals, g3_vals = data[:, 0], data[:, 2], data[:, 4]
        r_u, cos_u = np.unique(r_vals), np.unique(cos_vals)
        order = np.lexsort((cos_vals, r_vals))
        grid = g3_vals[order].reshape(len(r_u), len(cos_u))
        grid = np.where(grid < 1.0 / int(n_trials), 1.0 / int(n_trials), grid)
        interp = RegularGridInterpolator(
            (r_u, cos_u), grid, method="linear", bounds_error=False, fill_value=1.0
        )
        return cls(diameter, upper_cutoff, side, n_trials, interp)

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
        # needs_geometry contract: these are only optional in the shared base
        # signature; every geometry-aware call site supplies them.
        assert coords is not None and idxs is not None
        N, k = dists.shape
        if kdtree is None:
            # EMPTY ENVIRONMENT. `MC_engine.advance` passes `tree = cl if
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
            return np.ones(N), np.full((N, coords.shape[1]), np.nan)
        box = np.array([self.side, self.side, self.side])
        n_data = kdtree.data.shape[0]
        safe = np.minimum(idxs, n_data - 1)
        neigh = kdtree.data[safe]  # (N, k, 3)
        deltas = neigh - coords[:, None, :]
        deltas -= box * np.round(deltas / box)
        vec_BA = deltas[:, 0, :]
        ref_coords = neigh[:, 0, :]
        norm_BA = np.linalg.norm(vec_BA, axis=1, keepdims=True)
        vec_BC = deltas[:, 1:, :]
        dot = np.sum(vec_BA[:, None, :] * vec_BC, axis=2)
        norm_BC = dists[:, 1:]
        cos_t = np.clip(dot / (norm_BA * norm_BC), -1.0, 1.0)
        pts = np.column_stack([dists[:, 1:].flatten(), cos_t.flatten()])
        g3 = self._interp(pts).reshape(N, k - 1)
        g3 = np.where(g3 < self.min_g3, self.min_g3, g3)
        if gamma != 1.0:
            g3 = g3**gamma
        return np.prod(g3, axis=1), ref_coords

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
        # needs_geometry contract (see weight_bulk)
        assert trial_coords is not None and ref_coords is not None
        assert chosen_coord is not None
        sq = np.asarray(r, dtype=float)
        # No reference neighbour: `weight_bulk` marks the empty-environment case
        # with NaN. Same reasoning as there -- no triplet, so no three-body bias,
        # so unit weight. Checked before the arithmetic because a NaN direction
        # would otherwise reach `_interp` as a NaN (r, cos) pair.
        no_ref = ~np.isfinite(np.atleast_2d(ref_coords)).all(axis=1)
        if no_ref.all():
            return np.ones(np.atleast_1d(sq).shape[0])
        vec_BA = ref_coords - trial_coords
        vec_BA -= self.side * np.round(vec_BA / self.side)
        vec_BC = chosen_coord - trial_coords
        vec_BC -= self.side * np.round(vec_BC / self.side)
        dot = np.sum(vec_BA * vec_BC, axis=1)
        norm_BA = np.linalg.norm(vec_BA, axis=1)
        norm_BC = np.sqrt(sq)
        cos_t = np.clip(dot / (norm_BA * norm_BC), -1.0, 1.0)
        pts = np.column_stack([norm_BC, cos_t])
        g3 = self._interp(pts)
        g3 = np.where(g3 < self.min_g3, self.min_g3, g3)
        if gamma != 1.0:
            g3 = g3**gamma
        # rows whose reference neighbour was undefined carry no bias
        return np.where(no_ref, 1.0, g3)

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

    def __init__(self, lj: Any, aux: Aux3D) -> None:
        """Compose a plain LJ (no tail: the engine applies the tail at move
        level) with an Aux3D bias; core diameter and cutoff follow the LJ."""
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
        **_ignored: Any,
    ) -> LJAux3D:
        """Registry constructor: build both halves from the shared config keys
        (the LJ deliberately without `volume`, see __init__)."""
        from .lennard_jones import LennardJones

        lj = LennardJones(
            beta=beta, upper_cutoff=upper_cutoff, diameter=diameter, volume=None
        )
        aux = Aux3D.build(
            diameter=diameter,
            upper_cutoff=upper_cutoff,
            side=side,
            n_trials=n_trials,
            g3_csv=g3_csv,
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
        The tuple return mirrors Aux3D (needs_geometry is True)."""
        lj_w = self.lj.weight_pair(dists)
        aux_w, ref = self.aux.weight_bulk(
            dists, coords=coords, idxs=idxs, kdtree=kdtree, gamma=gamma
        )
        return lj_w * aux_w, ref

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
