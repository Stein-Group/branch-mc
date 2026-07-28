"""Weight policies for the Rosenbluth selection.

The selection loop is one algorithm; what differs between modes is how a trial's
selection weight is formed and how surviving weights are refreshed after a pick.
Each policy answers exactly that, so the loop in rosenbluth.py never branches on
mode.

  selection(pool) -> (sum_weight, probability)
  recompute(pool, chosen_coord, sq_filtered, gamma) -> None   (mutates pool)

EnergyWeigher covers LJ and both aux flavours (the potential supplies the
pairwise refresh; 3D-aux uses the carried reference neighbour). UniformWeigher
covers plain hard spheres, where every survivor is equally likely and there is
no refresh. The reweighting (psi6/q6) policy lives with that sampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._types import FloatArray

if TYPE_CHECKING:
    from ..potentials.base import Potential
    from ..trials import TrialSet


class EnergyWeigher:
    """Boltzmann-weighted selection for soft potentials (LJ, aux, ljaux).

    Physics: in configurational-bias sampling a trial is picked with
    probability proportional to its Rosenbluth weight w_i = exp(-beta u_i),
    where u_i is the trial's interaction energy with the frozen system plus
    every previously chosen trial. The system part is precomputed into
    pool.energy_weight by the neighbour build; the previously-chosen part is
    multiplied in incrementally by `recompute` after each pick, so w_i is
    always current without ever re-summing from scratch.
    """

    def __init__(self, potential: Potential) -> None:
        """Bind the potential whose `pairwise` supplies the refresh factors."""
        self.pot = potential

    def selection(self, pool: TrialSet) -> tuple[float, FloatArray | None]:
        """(sum of weights, normalised pick probabilities) over the pool. The
        sum is the Rosenbluth normalisation W at this level; a non-positive sum
        means every trial is annihilated (e.g. deep overlap underflow) and the
        caller must reject the move."""
        w = pool.energy_weight
        s = np.sum(w)
        if not (s > 0.0):
            return s, None  # exhausted pool; caller rejects the move
        return s, w / s

    def recompute(
        self,
        pool: TrialSet,
        chosen_coord: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """Fold the just-chosen trial's pair Boltzmann factors into every
        surviving trial's weight (in place). `sq_filtered` are the squared
        distances survivor -> chosen, already minimum-imaged and overlap-
        filtered by the selection loop."""
        g = self.pot.pairwise(
            sq_filtered,
            gamma,
            trial_coords=pool.coords,
            ref_coords=pool.ref_neighbor_coord,
            chosen_coord=chosen_coord,
        )
        pool.energy_weight = pool.energy_weight * g

    # ---- reverse direction (deletion) ----
    # The reverse pass reconstructs the Rosenbluth weight the CURRENT
    # configuration would have received had it been proposed: the removed
    # members are "re-selected" in order, so at step i the available weight is
    # the sum over members i.. plus any surviving trials. Detailed balance
    # requires these sums to be formed with exactly the same weight definition
    # as the forward pass, which is why the same policy object serves both.

    def reverse_no_trial(
        self, mol_in_cav: TrialSet, n_atoms: int
    ) -> tuple[FloatArray, FloatArray]:
        """Reverse-pass level sums when NO trial cloud survives: level i sees
        only the not-yet-re-selected members, so the sum is over w[i:] and the
        per-level chosen weight is w[i]. Returns (sum_array, weight_array)."""
        w = mol_in_cav.energy_weight
        sum_arr = np.array([np.sum(w[i:]) for i in range(n_atoms)], dtype=float)
        e_arr = w[:n_atoms].astype(float).copy()
        return sum_arr, e_arr

    def base_terms(self, mol_in_cav: TrialSet, i: int) -> tuple[float, float]:
        """(remaining member weight sum, weight of member i) at reverse level
        i; the trial-cloud contribution is added by the caller."""
        w = mol_in_cav.energy_weight
        return float(np.sum(w[i:])), float(w[i])

    def trial_sum(self, trial_set: TrialSet) -> float:
        """Total selection weight the surviving trial cloud adds to a reverse
        level's normalisation."""
        return float(np.sum(trial_set.energy_weight))

    def recompute_trial(
        self,
        trial_set: TrialSet,
        chosen: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """Reverse-pass twin of `recompute`: fold the re-selected member's pair
        factors into the surviving trials' weights (in place)."""
        g = self.pot.pairwise(
            sq_filtered,
            gamma,
            trial_coords=trial_set.coords,
            ref_coords=trial_set.ref_neighbor_coord,
            chosen_coord=chosen,
        )
        trial_set.energy_weight = trial_set.energy_weight * g

    def recompute_cav(
        self,
        mol_in_cav: TrialSet,
        i: int,
        chosen: FloatArray,
        gamma: float,
        side: float | FloatArray,
    ) -> None:
        """Fold the re-selected member i's pair factors into the LATER members
        (rows i+1..) of the removed set, minimum-imaging in the box; the
        reverse pass needs member-member interactions counted once, in
        selection order, mirroring how the forward pass accumulated them."""
        coords = mol_in_cav.coords[i + 1 :]
        diff = coords - chosen
        diff -= side * np.round(diff / side)
        sq = np.sum(diff**2, axis=1)
        g = self.pot.pairwise(
            sq,
            gamma,
            trial_coords=coords,
            ref_coords=mol_in_cav.ref_neighbor_coord[i + 1 :],
            chosen_coord=chosen,
        )
        mol_in_cav.energy_weight[i + 1 :] *= g


class UniformWeigher:
    """Hard-sphere selection: every admissible trial has Boltzmann weight
    exp(0) = 1, so selection is uniform, every sum is a count, and nothing
    needs refreshing after a pick (a pick can only DELETE overlapping
    survivors, which the loop does via the overlap filter, not the weigher).
    Each method is the w = 1 specialisation of its EnergyWeigher twin."""

    def selection(self, pool: TrialSet) -> tuple[float, FloatArray | None]:
        """(count, uniform probabilities): the W of a hard-core level is just
        the number of surviving trials."""
        n = len(pool)
        return n, np.ones(n) / n

    def recompute(
        self,
        pool: TrialSet,
        chosen_coord: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """No-op: unit weights never change."""
        pass

    # ---- reverse direction (deletion) ----
    def reverse_no_trial(
        self, mol_in_cav: TrialSet, n_atoms: int
    ) -> tuple[FloatArray, FloatArray]:
        """Level sums degenerate to the remaining member counts n, n-1, .., 1,
        each chosen member contributing weight 1."""
        return np.arange(n_atoms, 0, -1).astype(float), np.ones(n_atoms)

    def base_terms(self, mol_in_cav: TrialSet, i: int) -> tuple[float, float]:
        """(remaining member count, 1.0) at reverse level i."""
        return float(len(mol_in_cav.energy_weight[i:])), 1.0

    def trial_sum(self, trial_set: TrialSet) -> float:
        """The trial cloud contributes its size."""
        return float(len(trial_set.energy_weight))

    def recompute_trial(
        self,
        trial_set: TrialSet,
        chosen: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """No-op: unit weights never change."""
        pass

    def recompute_cav(
        self,
        mol_in_cav: TrialSet,
        i: int,
        chosen: FloatArray,
        gamma: float,
        side: float | FloatArray,
    ) -> None:
        """No-op: unit weights never change."""
        pass


class ReweightWeigher:
    """2D hexatic reweighting: the selection weight and the per-molecule term
    are the psi6 phasor magnitude carried on the trial. Like UniformWeigher
    there is no refresh after a pick (psi6 is fixed by the frozen system
    neighbours), but the weights are psi6, not 1."""

    @staticmethod
    def _psi6(s: TrialSet) -> FloatArray:
        """|psi6| per trial from the phasor accumulators the neighbour build
        stored on the TrialSet (lazy import avoids a cycle at module load)."""
        from .reweighting import psi6_magnitude

        return psi6_magnitude(s.cossum, s.sinsum, s.counts)

    def selection(self, pool: TrialSet) -> tuple[float, FloatArray | None]:
        """(sum of |psi6|, normalised probabilities); a zero sum (no bonded
        neighbours anywhere) rejects the move, as with EnergyWeigher."""
        w = self._psi6(pool)
        s = np.sum(w)
        if not (s > 0.0):
            return s, None
        return s, w / s

    def recompute(
        self,
        pool: TrialSet,
        chosen_coord: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """No-op: psi6 is computed against the frozen system neighbours only,
        so a pick does not change the survivors' weights."""
        pass

    # ---- reverse direction (deletion) ----
    def reverse_no_trial(
        self, mol_in_cav: TrialSet, n_atoms: int
    ) -> tuple[FloatArray, FloatArray]:
        """As EnergyWeigher.reverse_no_trial with w = |psi6| per member."""
        w = self._psi6(mol_in_cav)
        sum_arr = np.array([np.sum(w[i:]) for i in range(n_atoms)], dtype=float)
        e_arr = w[:n_atoms].astype(float).copy()
        return sum_arr, e_arr

    def base_terms(self, mol_in_cav: TrialSet, i: int) -> tuple[float, float]:
        """(remaining |psi6| sum, |psi6| of member i) at reverse level i."""
        w = self._psi6(mol_in_cav)
        return float(np.sum(w[i:])), float(w[i])

    def trial_sum(self, trial_set: TrialSet) -> float:
        """Total |psi6| the surviving trial cloud adds to a reverse level."""
        return float(np.sum(self._psi6(trial_set)))

    def recompute_trial(
        self,
        trial_set: TrialSet,
        chosen: FloatArray,
        sq_filtered: FloatArray,
        gamma: float,
    ) -> None:
        """No-op: see `recompute`."""
        pass

    def recompute_cav(
        self,
        mol_in_cav: TrialSet,
        i: int,
        chosen: FloatArray,
        gamma: float,
        side: float | FloatArray,
    ) -> None:
        """No-op: see `recompute`."""
        pass
