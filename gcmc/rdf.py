"""Radial distribution functions for the quenched-annealed system.

Two pair correlations matter for the benchmark:
  g_11(r) : fluid-fluid   (mobile-mobile)
  g_01(r) : fluid-matrix   (mobile against the frozen obstacles)

Both are computed with the minimum-image convention in a cubic periodic box.
Normalisation is the standard ideal-gas reference: the number of pairs in a
shell [r, r+dr] divided by the number expected for a uniform fluid of the same
mean density. For g_01 the reference uses the fluid density seen from each matrix
centre (or equivalently the matrix density seen from each fluid particle -- the
two give the same g_01 by symmetry of the pair count).

These are accumulated over many independent fluid configurations (and, when you
aggregate yourself, over matrix realisations): call accumulate() per snapshot,
then finalize() once at the end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._types import FloatArray


class RDFAccumulator:
    """Streaming radial distribution function g(r): pair-distance histograms
    are accumulated frame by frame and normalised at the end against the
    ideal-gas shell counts (2 pi r dr rho in 2D, 4 pi r^2 dr rho in 3D), so
    memory stays one histogram regardless of trajectory length."""

    def __init__(
        self,
        box: float,
        dimension: int = 3,
        r_max: float | None = None,
        n_bins: int = 200,
    ) -> None:
        """Fixed binning [0, r_max) in `n_bins` equal bins; r_max defaults to
        the half box, the largest radius the minimum image resolves."""
        self.box = float(box)
        self.dim = int(dimension)
        self.r_max = float(r_max) if r_max is not None else self.box / 2.0
        self.n_bins = int(n_bins)
        self.edges = np.linspace(0.0, self.r_max, self.n_bins + 1)
        self.dr = self.edges[1] - self.edges[0]
        self.r = 0.5 * (self.edges[:-1] + self.edges[1:])
        # shell volumes (3D spherical / 2D annular)
        if self.dim == 3:
            self.shell_vol = (
                (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
            )
        else:
            self.shell_vol = np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        self.box_vol = self.box**self.dim
        # 11 (fluid-fluid) accumulators
        self._h11 = np.zeros(self.n_bins)
        self._n11_frames = 0
        self._sum_Nf = 0.0  # for mean fluid density
        self._sum_Nf2_pairs = 0.0
        # 01 (fluid-matrix) accumulators
        self._h01 = np.zeros(self.n_bins)
        self._sum_Nf_for01 = 0.0
        self._n01_frames = 0
        self.matrix: FloatArray | None = None
        self._matrix_tree: Any = None
        self.n_matrix = 0

    def set_matrix(self, matrix: FloatArray) -> None:
        """Register the frozen matrix once (constant across all snapshots)."""
        self.matrix = np.asarray(matrix, float) % self.box
        self.n_matrix = len(self.matrix)
        self._matrix_tree = (
            cKDTree(self.matrix, boxsize=[self.box] * self.dim)
            if self.n_matrix
            else None
        )

    def accumulate(self, fluid: FloatArray) -> None:
        """Add one fluid configuration to both histograms."""
        fluid = np.asarray(fluid, float) % self.box
        nf = len(fluid)
        # --- fluid-fluid pairs ---
        if nf >= 2:
            tf = cKDTree(fluid, boxsize=[self.box] * self.dim)
            pairs = tf.query_pairs(self.r_max, output_type="ndarray")
            if len(pairs):
                d = self._mic_dist(fluid[pairs[:, 0]], fluid[pairs[:, 1]])
                hist, _ = np.histogram(d, bins=self.edges)
                self._h11 += 2.0 * hist  # each pair counted once -> x2 for ordered
            self._sum_Nf += nf
            self._n11_frames += 1
        # --- fluid-matrix pairs ---
        if nf >= 1 and self.n_matrix:
            # for each fluid particle, matrix neighbours within r_max
            nbr = self._matrix_tree.query_ball_point(fluid, self.r_max)
            assert self.matrix is not None  # n_matrix > 0 implies set_matrix ran
            alld = []
            for i, idxs in enumerate(nbr):
                if idxs:
                    alld.append(
                        self._mic_dist(
                            np.repeat(fluid[i : i + 1], len(idxs), axis=0),
                            self.matrix[idxs],
                        )
                    )
            if alld:
                d = np.concatenate(alld)
                hist, _ = np.histogram(d, bins=self.edges)
                self._h01 += hist
            self._sum_Nf_for01 += nf
            self._n01_frames += 1

    def _mic_dist(self, a: FloatArray, b: FloatArray) -> FloatArray:
        """Minimum-image distances between paired rows of `a` and `b` in the
        cubic box."""
        dv = a - b
        dv -= self.box * np.round(dv / self.box)
        return np.sqrt((dv**2).sum(axis=1))

    def finalize(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return (r, g11, g01). g=NaN-safe; zero where no data."""
        g11 = np.zeros(self.n_bins)
        if self._n11_frames and self._sum_Nf > 0:
            mean_Nf = self._sum_Nf / self._n11_frames
            rho_f = mean_Nf / self.box_vol
            # expected ordered pairs in shell = Nf * rho_f * shell_vol, per frame
            norm = self._n11_frames * mean_Nf * rho_f * self.shell_vol
            g11 = np.divide(
                self._h11, norm, out=np.zeros_like(self._h11), where=norm > 0
            )
        g01 = np.zeros(self.n_bins)
        if self._n01_frames and self.n_matrix and self._sum_Nf_for01 > 0:
            mean_Nf = self._sum_Nf_for01 / self._n01_frames
            rho_m = self.n_matrix / self.box_vol
            # expected fluid-matrix pairs in shell = Nf * rho_m * shell_vol, per frame
            norm = self._n01_frames * mean_Nf * rho_m * self.shell_vol
            g01 = np.divide(
                self._h01, norm, out=np.zeros_like(self._h01), where=norm > 0
            )
        return self.r, g11, g01
