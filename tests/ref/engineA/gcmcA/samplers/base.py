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
import numpy as np


class EnergyWeigher:
    def __init__(self, potential):
        self.pot = potential

    def selection(self, pool):
        w = pool.energy_weight
        s = np.sum(w)
        if not (s > 0.0):
            return s, None       # exhausted pool; caller rejects the move
        return s, w / s

    def recompute(self, pool, chosen_coord, sq_filtered, gamma):
        g = self.pot.pairwise(sq_filtered, gamma,
                              trial_coords=pool.coords,
                              ref_coords=pool.ref_neighbor_coord,
                              chosen_coord=chosen_coord)
        pool.energy_weight = pool.energy_weight * g

    # ---- reverse direction (deletion) ----
    def reverse_no_trial(self, mol_in_cav, n_atoms):
        w = mol_in_cav.energy_weight
        sum_arr = np.array([np.sum(w[i:]) for i in range(n_atoms)], dtype=float)
        e_arr = w[:n_atoms].astype(float).copy()
        return sum_arr, e_arr

    def base_terms(self, mol_in_cav, i):
        w = mol_in_cav.energy_weight
        return float(np.sum(w[i:])), float(w[i])

    def trial_sum(self, trial_set):
        return float(np.sum(trial_set.energy_weight))

    def recompute_trial(self, trial_set, chosen, sq_filtered, gamma):
        g = self.pot.pairwise(sq_filtered, gamma,
                              trial_coords=trial_set.coords,
                              ref_coords=trial_set.ref_neighbor_coord,
                              chosen_coord=chosen)
        trial_set.energy_weight = trial_set.energy_weight * g

    def recompute_cav(self, mol_in_cav, i, chosen, gamma, side):
        coords = mol_in_cav.coords[i + 1:]
        diff = coords - chosen
        diff -= side * np.round(diff / side)
        sq = np.sum(diff ** 2, axis=1)
        g = self.pot.pairwise(sq, gamma,
                              trial_coords=coords,
                              ref_coords=mol_in_cav.ref_neighbor_coord[i + 1:],
                              chosen_coord=chosen)
        mol_in_cav.energy_weight[i + 1:] *= g


class UniformWeigher:
    def selection(self, pool):
        n = len(pool)
        return n, np.ones(n) / n

    def recompute(self, pool, chosen_coord, sq_filtered, gamma):
        pass

    # ---- reverse direction (deletion) ----
    def reverse_no_trial(self, mol_in_cav, n_atoms):
        return np.arange(n_atoms, 0, -1).astype(float), np.ones(n_atoms)

    def base_terms(self, mol_in_cav, i):
        return float(len(mol_in_cav.energy_weight[i:])), 1.0

    def trial_sum(self, trial_set):
        return float(len(trial_set.energy_weight))

    def recompute_trial(self, trial_set, chosen, sq_filtered, gamma):
        pass

    def recompute_cav(self, mol_in_cav, i, chosen, gamma, side):
        pass


class ReweightWeigher:
    """2D hexatic reweighting: the selection weight and the per-molecule term
    are the psi6 phasor magnitude carried on the trial. Like UniformWeigher
    there is no refresh after a pick (psi6 is fixed by the frozen system
    neighbours), but the weights are psi6, not 1."""

    @staticmethod
    def _psi6(s):
        from .reweighting import psi6_magnitude
        return psi6_magnitude(s.cossum, s.sinsum, s.counts)

    def selection(self, pool):
        w = self._psi6(pool)
        s = np.sum(w)
        if not (s > 0.0):
            return s, None
        return s, w / s

    def recompute(self, pool, chosen_coord, sq_filtered, gamma):
        pass

    # ---- reverse direction (deletion) ----
    def reverse_no_trial(self, mol_in_cav, n_atoms):
        w = self._psi6(mol_in_cav)
        sum_arr = np.array([np.sum(w[i:]) for i in range(n_atoms)], dtype=float)
        e_arr = w[:n_atoms].astype(float).copy()
        return sum_arr, e_arr

    def base_terms(self, mol_in_cav, i):
        w = self._psi6(mol_in_cav)
        return float(np.sum(w[i:])), float(w[i])

    def trial_sum(self, trial_set):
        return float(np.sum(self._psi6(trial_set)))

    def recompute_trial(self, trial_set, chosen, sq_filtered, gamma):
        pass

    def recompute_cav(self, mol_in_cav, i, chosen, gamma, side):
        pass
