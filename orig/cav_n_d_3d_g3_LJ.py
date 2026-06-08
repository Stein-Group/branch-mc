#!/usr/bin/env python3
"""
This module implements a Grand Canonical Monte Carlo (GCMC) simulation for hard disks 
with an auxiliary potential. It simulates moves (add, remove, and move) in a 2D square box 
with periodic boundary conditions and uses a KD-tree for fast neighbor search.

There are two files saved during the simulation:
  1. The final configuration of the system (a .xyz file in the save_file_folder).
  2. An output file that contains the number of MC cycles and the various move counters.

Author: Dipesh Somvanshi
Date: Today
"""

import numpy as np
from scipy.spatial import cKDTree, KDTree
import time
import os
import glob
import re
import argparse
from scipy.spatial.distance import pdist, cdist
#from scipy.special import sph_harm
import timeit
from ase import Atoms
from ase.io import read, write
from scipy.interpolate import RegularGridInterpolator
import logging

import parallelization
import concurrent.futures

# ============================================================================
# ALGORITHM OVERVIEW
# ============================================================================
# This code implements Grand Canonical Monte Carlo (GCMC) with nested sampling:
#
# 1. GCMC operates in the (μ,V,T) ensemble where:
#    - μ (chemical potential) is fixed
#    - V (volume) is fixed
#    - T (temperature) is fixed
#    - N (number of particles) fluctuates
#
# 2. Three types of MC moves maintain detailed balance:
#    a) Insertion: Add a particle at random position
#    b) Deletion: Remove a random particle
#    c) Translation: Move a particle by small random displacement
#
# 3. Nested sampling enhances insertion/deletion acceptance:
#    - Define a spherical "cavity" in the system
#    - Generate many trial positions within cavity
#    - Sequentially select positions with Boltzmann-weighted probability
#    - This biasing is exactly cancelled in the acceptance criterion
#
# 4. Auxiliary potential (optional) further improves sampling:
#    - Apply artificial potential U_aux during sampling
#    - Reweight configurations to recover correct ensemble
#    - Gamma parameter controls auxiliary potential strength
#
# 5. Periodic boundary conditions eliminate surface effects:
#    - Particles leaving one side re-enter from opposite side
#    - Minimum image convention for distance calculations
#
# 6. KD-tree accelerates neighbor searches from O(N²) to O(N log N)
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

# -------------------- Command Line Argument Block (Commented) --------------------
# Uncomment the following lines to enable command line arguments for chem_pot and restart_mode.
#

parser = argparse.ArgumentParser(description='GCMC for hard disks.')
parser.add_argument('--restart',  action='store_true', help='Restart from last saved file')
parser.add_argument('--mu',       type=float, required=True, help='Chemical potential')
parser.add_argument('--r',        type=float, required=True, help='Cavity radius')
parser.add_argument('--d',        type=float, required=False, help='Depth')
parser.add_argument('--2d',       action='store_true', dest='two_d', help='Run in 2D mode')
parser.add_argument('--aux_pot',  action='store_true', help='Enable auxiliary potential')
parser.add_argument('--LJ_pot',  action='store_true', help='Enable Lennard jones potential')
parser.add_argument('--gamma_max',        type=float, required=False, default=0.2, help='Initial gamma value')
parser.add_argument('--delta_eta',        type=float, required=False, default=0.0, help='Diff packing fraction of aux pot')
parser.add_argument('--T', type=float, required=False, default=1.5, help='Reduced temperature T*')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--k',         type=float, help='Number of trial molecules')
group.add_argument('--rho_k',  type=float, help='Some alternative parameter')

args = parser.parse_args()

sampling_window = 1000

restart_mode      = args.restart
chem_pot           = args.mu
cavity_radius      = args.r
depth                 = int(args.d)
two_dim_flag     = args.two_d
aux_pot_flag      = args.aux_pot
gamma_max      = args.gamma_max
delta_eta            = args.delta_eta
Temp                 = args.T
lj_pot_flag          = args.LJ_pot

if args.k is not None:
    no_of_trial_molecules = int(args.k)
    trial_density = round((no_of_trial_molecules * (np.pi * 0.5 * 0.5))/((2*cavity_radius)**2),3)
else:
    trial_density = float(args.rho_k)
    no_of_trial_molecules = int((trial_density * (2*cavity_radius)**2)/(np.pi * 0.5 * 0.5))


logging.basicConfig(level=logging.CRITICAL)

print(f"chem_pot: {chem_pot}, no_of_trial_molecules: {no_of_trial_molecules}, cavity_radius: {cavity_radius}")
print(f"restart mode: {restart_mode}")
print("Aux pot", aux_pot_flag)
print(f"LJ {lj_pot_flag} T {Temp}")

# -----------------------------------------------------------------------------------


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================


# -------------------- Simulation Parameters --------------------

# logging.basicConfig(level=logging.DEBUG)
# #logging.basicConfig(level=logging.CRITICAL)
# chem_pot = -4
# restart_mode = True
# sampling_window = 5000

# cavity_radius = 1.5
# trial_density = 400
# gamma_max = 0.2
# delta_eta = 0.0
# aux_pot_flag = False #Flag to enable auxiliary potential
# no_of_trial_molecules = int((trial_density * (2*cavity_radius)**2)/(np.pi * 0.5 * 0.5))
# print("no_of_trial_molecules", no_of_trial_molecules)
# logging.debug(f'Initial no_of_trial_molecules:  {no_of_trial_molecules}') 
# depth = 3
# two_dim_flag = False  # Flag to indicate if the simulation is in 2D
# Temp            = 1.5
# lj_pot_flag  = True

#mu_-4_r_1.5_rhok_400_k_4583_2d_False_

# --- globals (set by the host code; defaults here only so the module imports) -

BETA = 1/Temp

if lj_pot_flag: assert two_dim_flag == False
if not lj_pot_flag: assert Temp == 1.0

BRANCH_flag = False
USE_DEPTH = True
PARALLEL = False

traj_freq = 1
log_buffer = []
LOG_FLUSH_EVERY = 1   # tune this; bigger = fewer syscalls
TRAJ_FLUSH_EVERY = LOG_FLUSH_EVERY * traj_freq  # how often to actually commit traj frames

#Number of parallel processes
n_workers = 4
cell_length_global = 30
buffer=2
x_spacing=4
y_spacing=3

#No of cells assigned to each processer
#If each processor has equal load, one processor will not lag behind
#If chunk_size = 1, everytime a processor finishes a cell,
#it will have to pickle and copy data to another processor
#By setting chunk_size >= 1, all pickling is done once and copied once.
chunk_size = 1

reweighting_flag = False  # Flag to enable reweighting of the configuration
use_LUT = False #Flag to use LUT for speedup

# Gamma controls auxiliary potential strength at each nesting depth
# Lower values = weaker potential = better sampling in that region
if aux_pot_flag == True:

    if two_dim_flag:
        #gamma_dict = {1:0.5402, 2:0.1765, 3:0.1194, 4:0.1037, 5:0.0951, 6:0.0631, 7:0.0504, 8:0.0, 9:0.0}
        gamma_dict = {1:0.193548, 2:0.049948, 3:0.022557, 4:0.010187, 5:0.007558, 6:0.002438, 7:0.001180, 8:0.000495, 9:0.000176}

    else:
        #gamma_dict = {1:0.4516, 2:0.2914, 3:0.0658, 4:0.0127, 5:0.0062, 6:0.0042, 7:0.0, 8:0.0, 9:0.0}
        gamma_dict = {1:0.2903, 2:0.1030, 3:0.0332, 4:0.0193, 5:0.0037, 6:0.0042, 7:0.0, 8:0.0, 9:0.0}

elif  reweighting_flag == True:
    # Full potential (gamma=1.0) at all depths for reweighting analysis
    gamma_dict = {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.0, 9:1.0}
else:
    # Standard GCMC without auxiliary potential (gamma=1.0)
    gamma_dict = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}

if aux_pot_flag: assert not reweighting_flag
if reweighting_flag: assert not aux_pot_flag

# if two_dim_flag and not PARALLEL:
#     side_of_box = 75
# else:
#     side_of_box = 20

match (two_dim_flag, PARALLEL):
    case (True,  True ): side_of_box = 200
    case (True,  False): side_of_box = 75
    case (False, True ): side_of_box = 50
    case (False, False): side_of_box = 20

logging.debug(f'Side of box:  {side_of_box}') 

if not lj_pot_flag: assert chem_pot >= 0

# Side of the box. The simulation is 2D, so the box is a square.
load_initial_config = False  # Flag to load initial configuration from a file

assert cavity_radius <= side_of_box, "Cavity radius must be less or equal than the side of the box."

# ============================================================================
# THERMODYNAMIC SETUP
# ============================================================================

logging.debug(f"chem_pot: {chem_pot}")
# Fugacity z = exp(βμ) controls particle insertion/deletion rates in GCMC
fugacity = np.exp(BETA * chem_pot)

# Directory for saving simulation files (configuration and counters)
save_file_folder = "run-files"
if not os.path.exists(save_file_folder):
    os.makedirs(save_file_folder)
    logging.debug(f"Directory '{save_file_folder}' created.")
else:
    logging.debug(f"Directory '{save_file_folder}' already exists.")

# Name of the output file; note that two files will be saved:
# 1. The final configuration (.xyz file)
# 2. The output file containing MC step counters and other data.
#traj_file_path = os.path.join(save_file_folder, f"mu_{chem_pot}_r_{cavity_radius}_k_{no_of_trial_molecules}_depth_{depth}_2d_{str(two_dim_flag)}_aux_pot_{str(aux_pot_flag)}_.extxyz")
#log_file_path = f"mu_{chem_pot}_r_{cavity_radius}_k_{no_of_trial_molecules}_depth_{depth}_2d_{str(two_dim_flag)}_aux_pot_{str(aux_pot_flag)}_.log"
    
if aux_pot_flag:
    traj_file_path = os.path.join(
        save_file_folder,
        f"mu_{chem_pot}_r_{cavity_radius}_rhok_{trial_density}_k_{no_of_trial_molecules}"
        f"_2d_{two_dim_flag}_aux_pot_{aux_pot_flag}_gammamax_{gamma_max}_deltaeta_{delta_eta}_.extxyz"
    )
    log_file_path = (
        f"mu_{chem_pot}_r_{cavity_radius}_rhok_{trial_density}_k_{no_of_trial_molecules}"
        f"_2d_{two_dim_flag}_aux_pot_{aux_pot_flag}_gammamax_{gamma_max}_deltaeta_{delta_eta}_.log"
    )
else:
        traj_file_path = os.path.join(
        save_file_folder,
        f"mu_{chem_pot}_r_{cavity_radius}_rhok_{trial_density}_k_{no_of_trial_molecules}_2d_{two_dim_flag}_.extxyz"
        )
    
        log_file_path = (f"mu_{chem_pot}_rhok_{trial_density}_r_{cavity_radius}_k_{no_of_trial_molecules}_2d_{two_dim_flag}_.log")

# ============================================================================
# MONTE CARLO PARAMETERS
# ============================================================================


burn_in = 0  # Number of initial MC cycles to discard (equilibration period)
mc_cycles = 1.0025e9  # Total number of Monte Carlo cycles to execute

move_factor = 0.25  # Maximum displacement = 0.25 * diameter
                       # Smaller values = higher acceptance, slower equilibration
                       # Larger values = lower acceptance, faster exploration
radius_of_disk = 0.5  # Hard disk radius
diameter_of_disk = 2*radius_of_disk
r_squared = (diameter_of_disk)**2  # Squared overlap distance for hard disk/sphere interaction

epsilon = np.finfo(float).eps

if two_dim_flag:
    no_neigh = 40  # Maximum neighbors to query per particle (KD-tree parameter)
                   # Should exceed typical coordination number for accurate energy calculations
else:
    no_neigh = 75

#upper_cutoff = 1.5  # Upper cutoff distance for pairwise interactions; above this, interaction energy is zero.

# Cutoff distance for auxiliary potential interactions
# Beyond this distance, interaction energy is zero
if reweighting_flag:
    upper_cutoff = 3*radius_of_disk #1.5σ = 1.5
elif lj_pot_flag:
    upper_cutoff = 2.5      # rc for the LJ branch
else:
    if two_dim_flag:
        upper_cutoff = 3.7545
    else:
        upper_cutoff = 2.0

# ============================================================================
# VOLUME/AREA CALCULATIONS FOR packing_fraction
# ============================================================================
if two_dim_flag:
    volume_of_molecule = np.pi * radius_of_disk**2
    Total_volume = side_of_box**2
else:
    volume_of_molecule = (4/3) * np.pi * radius_of_disk**3
    Total_volume = side_of_box**3

# LJ long-range tail prefactor (3D, sigma=eps=1). U_tail = (8/3) pi eps sigma^3 (N^2/V) X,
# X = (1/3) rc^-9 - rc^-3, so the energy change on N -> N+1 is lj_tail_pref*(2N+1).
# Applied exactly once per insertion/deletion in the acceptance criterion.
if lj_pot_flag:
    _lj_X = upper_cutoff ** -9 / 3.0 - upper_cutoff ** -3
    lj_tail_pref = (8.0 / 3.0) * np.pi / Total_volume * _lj_X

# ============================================================================
# INITIAL CONFIGURATION
# ============================================================================

if load_initial_config:
    if two_dim_flag:
        molecules_xyz = np.loadtxt(f'configs/initial_config_{int(chem_pot)}.xyz', usecols=(0, 1), delimiter=' ')
        #molecules_xyz = np.loadtxt(f'configs/xyz_chem_pot_12_parallel.xyz', usecols=(0, 1), delimiter=' ')
    else:
        if lj_pot_flag:
            molecules_xyz = np.loadtxt(f'configs_3d/mu_1.0.xyz', usecols=(0, 1, 2), delimiter=' ')
        else:
            molecules_xyz = np.loadtxt(f'configs_3d/mu_{int(chem_pot)}.0.xyz', usecols=(0, 1, 2), delimiter=' ')
else:
    num_mol = max(int(0.0 * (Total_volume/volume_of_molecule)), 1)
    if two_dim_flag:
        molecules_xyz = np.random.uniform(low=0, high=side_of_box, size=(num_mol, 2))
    else:
        molecules_xyz = np.random.uniform(low=0, high=side_of_box, size=(num_mol, 3))

packing_fraction = (len(molecules_xyz)*volume_of_molecule)/Total_volume  # Packing fraction η = (N × v_particle) / V_box  
logging.debug(f'Initial packing_fraction:  {packing_fraction}') 

if two_dim_flag: box_dims = [side_of_box, side_of_box]
else:
    box_dims = [side_of_box, side_of_box, side_of_box]

# ============================================================================
# AUXILIARY POTENTIAL PARAMETERS (2D and 3D)
# ============================================================================
# These parameters define the auxiliary potential used for enhanced sampling.
# For 2D: Uses fitted polynomial potential based on hard disk equation of state
# For 3D: Uses hard sphere potential based on Carnahan-Starling EOS

# -------------------- Auxiliary Potential Parameters --------------------
if two_dim_flag:
    sigma = 1.0 
    eta = 0.69 
    eta_2 = eta**2
    eta_3 = eta**3
    eta_4 = eta**4
    eta_5 = eta**5
    omega = (4.07688 + 3.53995 * eta) / sigma
    kappa = (1.19878 - 1.54059 * eta - 0.567808 * eta_2) / sigma
    alpha = (-1.69377 + 13.50042 * eta - 29.8772 * eta_2) / sigma
    r_star = 2.14098 - 1.03267 * eta + 0.320164 * eta_2 + 0.0102421 * eta_3
    g_m = 1.97652 - 6.34821 * eta + 11.53962 * eta_2 - 5.96960 * eta_3 - 3.92125 * eta_4
    mu = (-116.2837 + 1218.065 * eta - 4784.826 * eta_2 + 9229.068 * eta_3 - 8862.922 * eta_4 + 3422.542 * eta_5) / sigma
    delta = -omega * sigma * r_star - np.arctan(((kappa * sigma * r_star) + 1) / (omega * sigma * r_star))
    C = ((r_star * (g_m - 1) * np.exp(kappa * sigma * r_star)) / (np.cos((omega * sigma * r_star) + delta))) / sigma
    r_m = r_star * sigma
    b_1 = -1.04191e8
    b_2 = 2.66813e8
    m_1 = 53
    m_2 = 56
    c = 1 / (0.75 + np.finfo(float).eps)
    g_sigma = ((1 - (7/16) * eta + (1/36) * eta_2 - (2/21) * eta_3) / ((1 - eta)**2) + (b_1 * (eta**m_1) + b_2 * (eta**m_2)) / (1 - c * eta))
    B = (((r_star * g_m) - g_sigma * np.exp(mu * sigma * (r_star - 1))) /
        (np.exp(alpha * sigma * (r_star - 1)) - np.exp(mu * sigma * (r_star - 1)))) / sigma
    A = (g_sigma - B) * sigma
else:
    # 3D Hard Sphere parameters
    sigma = 1.0
    eta = 0.45  # Adjust this for 3D: eta = π * rho * sigma^3 / 6
    eta_2 = eta**2
    eta_3 = eta**3
    eta_4 = eta**4
    omega = (6.56745 + 0.721795 * eta) / sigma
    kappa = (1.75879 - 3.23977 * eta + 0.449588 * eta_2 - 0.148890 * eta_3) / sigma
    alpha = (0.595157 - 2.469402 * eta - 31.99603 * eta_2 - 0.897749 * eta_3) / sigma
    r_star = 1.960492 - 0.240131 * eta - 3.26102 * eta_2 + 4.08014 * eta_3
    g_m = 1.072954 - 1.13263 * eta + 3.05087 * eta_2 - 5.94853 * eta_3
    mu = (0.58236 + 8.43296 * eta - 34.52268 * eta_2 + 58.63440 * eta_3 - 35.13525 * eta_4) / sigma
    delta = -omega * sigma * r_star - np.arctan(((kappa * sigma * r_star) + 1) / (omega * sigma * r_star))
    C = ((r_star * (g_m - 1) * np.exp(kappa * sigma * r_star)) / (np.cos((omega * sigma * r_star) + delta))) / sigma
    r_m = r_star * sigma
    # 3D EoS for g(sigma)
    g_sigma = (1 - 0.5 * eta + (5/52) * eta_2 - 0.25 * eta_3 + 0.125 * eta_4) / ((1 - eta)**3)
    B = (((r_star * g_m) - g_sigma * np.exp(mu * sigma * (r_star - 1))) /
        (np.exp(alpha * sigma * (r_star - 1)) - np.exp(mu * sigma * (r_star - 1)))) / sigma
    A = (g_sigma - B) * sigma
    

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

# ----------------------------------------------------------------------

class TrialMoleculeSet:
    """
    Container for managing sets of trial molecules with their properties.
    
    This class stores trial molecule coordinates along with their associated
    energy weights and structural order parameters (ψ₆). It provides convenient
    methods for filtering, copying, and manipulating sets of trial molecules
    during GCMC insertion/deletion moves.
    
    A class to manage a set of trial molecules for the GCMC simulation.
    
    Attributes:
        trial_molecules (np.ndarray): Array of trial molecule coordinates.
        no_of_trial_molecules (int): Number of trial molecules.
        energy_weight (np.ndarray): Energy weights (g_r) for each molecule.
        psi_6 (np.ndarray): Hexatic order parameter for each molecule.
    """
    
    def __init__(self, coords, energy_weight, cossum=None, sinsum=None, counts=None, ref_neighbor_coord=None):
        # expects data either None, a list-of-lists, or an ndarray
        self._coords = coords
        self._energy_weight = energy_weight #if energy_weight is not None else np.ones(len(arr), dtype=float)
        self._cossum = cossum if cossum is not None else np.ones(len(coords), dtype=float)
        self._sinsum = sinsum if sinsum is not None else np.ones(len(coords), dtype=float)
        self._counts = counts if counts is not None else np.ones(len(coords), dtype=float)
        # Store reference neighbor COORDINATES (not indices) to handle kdtree changes
        if ref_neighbor_coord is None:
            self._ref_neighbor_coord = np.zeros((len(coords), 3) if len(coords) > 0 and coords.shape[1] == 3 else (len(coords), 2), dtype=float)
        else:
            self._ref_neighbor_coord = ref_neighbor_coord

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, index):
        if isinstance(index, (np.ndarray, list, slice)):
            return TrialMoleculeSet(self._coords[index], self._energy_weight[index], self._cossum[index], self._sinsum[index], self._counts[index], self._ref_neighbor_coord[index])
        
        return self._coords[index], self._energy_weight[index], self._cossum[index], self._sinsum[index], self._counts[index]

    def __repr__(self):
            n = len(self)
            preview_n = min(3, n-1)
            lines = [f"<TrialMolSet: {n} molecules>"]
            for i in range(preview_n):
                lines.append(f"  #{i+1}: coord={self._coords[i]}, energy={self._energy_weight[i]:.3f}, cos_sum={self._cossum[i]:.3f}, sin_sum={self._sinsum[i]:.3f}, counts={self._counts[i]:.3f}")
            if n > preview_n:
                lines.append("  ...")
            lines.append(f"  #{n}: coord={self._coords[-1]}, energy={self._energy_weight[-1]:.3f}, cos_sum={self._cossum[-1]:.3f}, sin_sum={self._sinsum[-1]:.3f}, counts={self._counts[-1]:.3f}")
            return "\n".join(lines)

    @property
    def coords(self):
        return self._coords

    @property
    def energy_weight(self):
        return self._energy_weight

    @energy_weight.setter
    def energy_weight(self, value):
        self._energy_weight = value

    @property
    def sinsum(self):
        return self._sinsum
    
    @sinsum.setter
    def sinsum(self, value):
        self._sinsum = value

    @property
    def cossum(self):
        return self._cossum

    @cossum.setter
    def cossum(self, value):
        self._cossum = value

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, value):
        self._counts = value
    
    @property
    def ref_neighbor_coord(self):
        return self._ref_neighbor_coord

    @ref_neighbor_coord.setter
    def ref_neighbor_coord(self, value):
        self._ref_neighbor_coord = value
     
    def mask(self, boolean_mask):
        """
        Create a new TrialMoleculeSet with molecules filtered by the boolean mask.
        
        Args:
            boolean_mask (np.ndarray): Boolean array indicating which trial molecules to keep.
        
        Returns:
            TrialMoleculeSet: A new instance with filtered trial molecules.
        """
        return TrialMoleculeSet(self._coords[boolean_mask], self._energy_weight[boolean_mask], self._cossum[boolean_mask], self._sinsum[boolean_mask], self._counts[boolean_mask], self._ref_neighbor_coord[boolean_mask])

    def first(self):
        """
        Get the first trial molecule in the set.
        
        Returns:
            tuple: Coordinates, energy weight, and psi_6 of the first trial molecule.
        """
        if len(self) == 0:
            raise IndexError("TrialMoleculeSet is empty.")
        return self._coords[0], self._energy_weight[0], self._cossum[0], self._sinsum[0], self._counts[0]
    
    def delete(self, index):
        """
        Delete the trial molecule at the specified index.
        
        Args:
            index (int): Index of the trial molecule to delete.
        """
        self._coords = np.delete(self._coords, index, axis=0)
        self._energy_weight = np.delete(self._energy_weight, index, axis=0)
        self._cossum = np.delete(self._cossum, index, axis=0)
        self._sinsum = np.delete(self._sinsum, index, axis=0)
        self._counts = np.delete(self._counts, index, axis=0)
        self._ref_neighbor_coord = np.delete(self._ref_neighbor_coord, index, axis=0)

        return self

    def copy(self):
        """
        Create a copy of the TrialMoleculeSet.
        
        Returns:
            TrialMoleculeSet: A new instance with the same coordinates, energy weights, and psi_6.
        """
        return TrialMoleculeSet(self._coords.copy(), self._energy_weight.copy(), self._cossum.copy(), self._sinsum.copy(), self._counts.copy(), self._ref_neighbor_coord.copy())
    
    def merge(self, other:'TrialMoleculeSet'):
        merged_coords = np.vstack([self._coords, other._coords])
        merged_energy_weight = np.hstack([self._energy_weight, other._energy_weight])
        merged_cossum = np.hstack([self._cossum, other._cossum])
        merged_sinsum = np.hstack([self._sinsum, other._sinsum])
        merged_counts = np.hstack([self._counts, other._counts])
        merged_ref_neighbor_coord = np.vstack([self._ref_neighbor_coord, other._ref_neighbor_coord])

        return TrialMoleculeSet(merged_coords, merged_energy_weight, merged_cossum, merged_sinsum, merged_counts, merged_ref_neighbor_coord)


class InteractionEnergyLUT:
    """Lookup table for fast interaction energy calculations."""
    
    def __init__(self, A, B, C, mu, alpha, omega, delta, kappa, sigma,
                 diameter_of_disk, r_m, upper_cutoff, 
                 n_points=512 #1024 #2048  #8192
                 ):
        """
        Initialize lookup tables for interaction energy.
        
        Args:
            A, B, C, mu, alpha, omega, delta, kappa, sigma: Potential parameters
            diameter_of_disk: Overlap threshold (1.0)
            r_m: Transition point between depletion and structural regions (1.5639)
            upper_cutoff: Maximum interaction distance (2.7245)
            n_points: Total number of points in lookup tables (8192 = ~64KB)
        """
        self.diameter_of_disk = diameter_of_disk
        self.r_m = r_m
        self.upper_cutoff = upper_cutoff
        
        # Calculate ranges and allocate points proportionally
        dep_range = r_m - diameter_of_disk
        str_range = upper_cutoff - r_m
        total_range = dep_range + str_range
        
        n_dep = int(n_points * dep_range / total_range)
        n_str = n_points - n_dep
        
        # Depletion region lookup table
        self.r_dep = np.linspace(diameter_of_disk, r_m, n_dep)
        self.g_dep = (A / self.r_dep) * np.exp(mu * (self.r_dep - sigma)) + \
                     (B / self.r_dep) * np.exp(alpha * (self.r_dep - sigma))
        
        # Structural region lookup table
        self.r_str = np.linspace(r_m, upper_cutoff, n_str)
        self.g_str = 1 + (C / self.r_str) * np.cos(omega * self.r_str + delta) * \
                     np.exp(-kappa * self.r_str)
    
    def __call__(self, r, gamma):
        """
        Calculate Boltzmann weight using lookup tables.
        
        Args:
            r (np.ndarray): 1D or 2D array of distances
            gamma (float): Exponent to scale the potential (0.0 to 1.0)
        
        Returns:
            np.ndarray: Boltzmann weight for each interaction
        """
        # Handle inf values
        if r.ndim == 2:
            r = r.copy()
            r[np.isinf(r[:, 0]), 0] = self.upper_cutoff
        elif r.ndim == 1:
            r = np.minimum(r, self.upper_cutoff)
        
        # Initialize output (default to 1.0 for no-interaction region)
        g_r = np.ones_like(r)
        
        # Overlap region: r < diameter_of_disk => g = 0
        overlap_mask = r < self.diameter_of_disk
        g_r[overlap_mask] = 0.0
        
        # Depletion region: interpolate from lookup table
        dep_mask = (r >= self.diameter_of_disk) & (r < self.r_m)
        if np.any(dep_mask):
            g_r[dep_mask] = np.interp(r[dep_mask], self.r_dep, self.g_dep)
        
        # Structural region: interpolate from lookup table
        str_mask = (r >= self.r_m) & (r < self.upper_cutoff)
        if np.any(str_mask):
            g_r[str_mask] = np.interp(r[str_mask], self.r_str, self.g_str)
        
        # Apply gamma scaling
        if gamma != 1.0:
            g_r = g_r ** gamma
        
        # Aggregate interactions
        if g_r.ndim == 2:
            e_minus_beta_u_r = np.prod(g_r, axis=1)
        else:
            e_minus_beta_u_r = g_r
        
        return e_minus_beta_u_r


if aux_pot_flag:
    if two_dim_flag:
        interaction_lut = InteractionEnergyLUT(
            A, B, C, mu, alpha, omega, delta, kappa, sigma,
            diameter_of_disk, r_m, upper_cutoff
        )
    else:
        # Load 3D g3 data from CSV
        
        csv_data = np.loadtxt('g3_analysis_data.csv', delimiter=',', skiprows=1)
        r_vals = csv_data[:, 0]
        cos_theta_vals = csv_data[:, 2]
        g3_vals = csv_data[:, 4]
        
        r_unique = np.unique(r_vals)
        cos_theta_unique = np.unique(cos_theta_vals)
        
        n_r = len(r_unique)
        n_cos = len(cos_theta_unique)
        
        # Reshape to grid: first sort by r, then by cos_theta
        sorted_idx = np.lexsort((cos_theta_vals, r_vals))
        g3_grid = g3_vals[sorted_idx].reshape(n_r, n_cos)
        
        # Replace zeros with minimum value to avoid zero weights
        min_g3 = 1.0 / no_of_trial_molecules
        g3_grid = np.where(g3_grid < min_g3, min_g3, g3_grid)
        
        g3_interpolator = RegularGridInterpolator(
            (r_unique, cos_theta_unique),
            g3_grid,
            method='linear',
            bounds_error=False,
            fill_value=1.0
        )
        
        print(f"Loaded g3 interpolator: {n_r} r-points × {n_cos} cos_theta-points")
        print(f"g3 range: [{np.min(g3_grid):.6f}, {np.max(g3_grid):.6f}]")


# ============================================================================
# INTERACTION ENERGY CALCULATIONS
# ============================================================================

# -------------------- Interaction Energy Calculation --------------------

"""
calculate_interaction_energy with a new full-LJ branch.

Full LJ is reproduced as: truncated-but-NOT-shifted explicit pair sum (r < rc)
plus the standard 3D long-range correction, folded straight into the returned
Boltzmann weight so nothing downstream touches energies.

Reduced units: sigma = epsilon = 1, beta = 1/T, lambda = 1.
Globals expected when lj_pot_flag is on:  T (temperature), V (box volume),
upper_cutoff (= rc).  N and choice ("add"/"remove") come in as arguments.
"""

def calculate_interaction_energy(r, gamma, N=None, choice=None):
    
    if aux_pot_flag:

        # Handle inf values
        if r.ndim == 2:
            r = r.copy()
            r[np.isinf(r[:, 0]), 0] = upper_cutoff
        elif r.ndim == 1:
            r = np.minimum(r, upper_cutoff)
        

        assert np.amin(r) >= 1.0, np.amin(r)
        inf_mask = r != np.inf
        assert np.amax(r[inf_mask]) <= upper_cutoff, np.amax(r)
        assert not (r < diameter_of_disk).any(), r[r < diameter_of_disk]

        # Initialize output (default to 1.0 for no-interaction region)
        g_r = np.ones_like(r)
        
        # Overlap region: r < 2*radius_of_disk => g = 0
        overlap_mask = r < diameter_of_disk
        g_r[overlap_mask] = 0.0
        
        # Depletion region: 2*radius_of_disk <= r < r_m
        dep_mask = (r >= diameter_of_disk) & (r < r_m)
        if np.any(dep_mask):
            r_dep = r[dep_mask]
            g_r[dep_mask] = (A / r_dep) * np.exp(mu * (r_dep - sigma)) + (B / r_dep) * np.exp(alpha * (r_dep - sigma))
        
        # Structural region: r_m <= r < upper_cutoff
        str_mask = (r >= r_m) & (r < upper_cutoff)
        if np.any(str_mask):
            r_str = r[str_mask]
            g_r[str_mask] = 1 + (C / r_str) * np.cos(omega * r_str + delta) * np.exp(-kappa * r_str)
        
        # No-interaction region already set to 1.0

        #g_r = g_r ** gamma
        if gamma != 1.0:
            g_r = g_r ** gamma

        #     non_trivial = g_r != 1.0  # Find elements that aren't 1.0
        #     if np.any(non_trivial):
        #         g_r[non_trivial] **= gamma  
                        
        # Aggregate interactions
        if g_r.ndim == 2:
            e_minus_beta_u_r = np.prod(g_r, axis=1)
        else:
            e_minus_beta_u_r = g_r
        

        assert not np.isnan(e_minus_beta_u_r).any(), e_minus_beta_u_r
        assert not np.isinf(e_minus_beta_u_r).any(), e_minus_beta_u_r
        
        return e_minus_beta_u_r
    
    elif lj_pot_flag:

        # Explicit truncated-unshifted 12-6 pair Boltzmann weights ONLY.
        # The long-range tail correction is a system-level, per-MOVE quantity and
        # must NOT live here: this function is called once per pair at every level
        # of the nested Rosenbluth recursion, so any tail term folded in gets
        # multiplied back O(depth^2) times. The tail is applied exactly once in the
        # add/remove acceptance criterion instead (see lj_tail_pref usage below).
        rc   = upper_cutoff

        # r >= rc (incl. inf padding) -> weight 1
        g_r = np.ones_like(r)
        inside = r < rc
        if np.any(inside):
            inv_r6 = (1.0 / r[inside]) ** 6
            u = 4.0 * inv_r6 * (inv_r6 - 1.0)
            with np.errstate(over="ignore"):     # tiny r -> u huge -> weight 0
                g_r[inside] = np.exp(-BETA * u)

        w = np.prod(g_r, axis=1) if g_r.ndim == 2 else g_r

        assert not np.isnan(w).any()
        assert not np.isinf(w).any()
        return w

    else:
        # hard-disk / hard-sphere branch unchanged
        if r.ndim == 1:
            return np.where(r <= 1.0, 0, 1)
        else:
            return np.sum(r >= 1.0, axis=1)

# ============================================================================
# ORDER PARAMETER CALCULATIONS
# ============================================================================


def calculate_psi_6_2d(deltas2d, finite_mask, gamma):
    """
    Calculate the 2D hexatic (ψ₆) order parameter.
    
    Measures 6-fold rotational symmetry in the local neighborhood.
    Perfect hexagonal order gives ψ₆ = 1, random gives ψ₆ ≈ 0.
    
    Args:
        deltas2d (np.ndarray): 2D displacement vectors to neighbors, shape (N, k, 2)
        finite_mask (np.ndarray): Boolean mask for valid neighbors, shape (N, k)
        gamma (float): Bias exponent to dampen the order parameter effect
    
    Returns:
        np.ndarray: Biased ψ₆ values for each position, shape (N,)
    """
    # Bond angles θ from displacement vectors
    thetas = np.arctan2(deltas2d[..., 1], deltas2d[..., 0])  # (N, k)
    
    # 6-fold complex phasors: exp(i*6θ) = cos(6θ) + i*sin(6θ)
    cos_6theta = np.cos(6 * thetas)  # (N, k)
    sin_6theta = np.sin(6 * thetas)  # (N, k)
    
    # Apply mask to exclude invalid neighbors
    cos_6theta_masked = cos_6theta * finite_mask
    sin_6theta_masked = sin_6theta * finite_mask
    
    # Sum over neighbors
    cos_sum = np.sum(cos_6theta_masked, axis=1)  # (N,)
    sin_sum = np.sum(sin_6theta_masked, axis=1)  # (N,)
    
    # Count valid neighbors
    counts = finite_mask.sum(axis=1)  # (N,)
    
    # # Handle zero neighbors
    # has_neighbors = counts > 0
    
    # # Average the complex phasors, then take magnitude
    # psi_6 = np.ones(deltas2d.shape[0]) * 0.3 #Start with psi6 of 0.3 which is the psi6 of random arrangement

    # psi_6[has_neighbors] = np.sqrt(
    #     (cos_sum[has_neighbors] / counts[has_neighbors])**2 + 
    #     (sin_sum[has_neighbors] / counts[has_neighbors])**2
    # )
    
    # print(counts[has_neighbors])

    # Apply bias exponent
    return cos_sum, sin_sum, counts

def calculate_q6_3d(deltas3d, finite_mask, gamma):
    """
    Calculate the 3D Steinhardt bond order parameter Q₆ (local version).
    
    Measures local crystalline order using spherical harmonics.
    Different crystal structures have characteristic Q₆ values:
    - FCC: ~0.575
    - BCC: ~0.511  
    - HCP: ~0.485
    - Random: ~0.0
    
    Args:
        deltas3d (np.ndarray): 3D displacement vectors to neighbors, shape (N, k, 3)
        finite_mask (np.ndarray): Boolean mask for valid neighbors, shape (N, k)
        gamma (float): Bias exponent to dampen the order parameter effect
    
    Returns:
        np.ndarray: Biased Q₆ values for each position, shape (N,)
    """
    # Convert Cartesian to spherical coordinates
    x, y, z = deltas3d[..., 0], deltas3d[..., 1], deltas3d[..., 2]
    
    # r = sqrt(x² + y² + z²)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # θ (polar angle from z-axis): [0, π]
    # Avoid division by zero
    theta = np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0))
    
    # φ (azimuthal angle in xy-plane): [0, 2π]
    phi = np.arctan2(y, x)
    
    # Calculate spherical harmonics Y₆ₘ for m = -6 to +6
    # sph_harm(m, l, phi, theta) computes Y_l^m
    # Returns complex values
    
    N = deltas3d.shape[0]
    #k = deltas3d.shape[1]
    
    # Storage for averaged q₆ₘ components
    q6m = np.zeros((N, 13), dtype=complex)  # 13 values for m = -6 to +6
    
    # Count valid neighbors for each position
    counts = finite_mask.sum(axis=1)  # (N,)
    has_neighbors = counts > 0
    
    # Calculate each m component
    for i, m in enumerate(range(-6, 7)):  # m = -6, -5, ..., 5, 6
        # Y₆ₘ for all positions and neighbors
        Y6m = sph_harm(m, 6, phi, theta)  # (N, k) complex array
        
        # Mask invalid neighbors
        Y6m_masked = Y6m * finite_mask
        
        # Average over neighbors
        q6m_sum = np.sum(Y6m_masked, axis=1)  # (N,) complex
        q6m[has_neighbors, i] = q6m_sum[has_neighbors] / counts[has_neighbors]
    
    # Initialize Q6 with liquid-like default (parallel to the 2D change)
    Q6 = np.ones(N) * 0.3

    # Calculate Q₆ = sqrt(4π/13 * Σₘ |q₆ₘ|²)
    q6_squared_sum = np.sum(np.abs(q6m)**2, axis=1)
    Q6[has_neighbors] = np.sqrt(4 * np.pi / 13 * q6_squared_sum[has_neighbors])

    #q6_squared_sum = np.sum(np.abs(q6m)**2, axis=1)  # (N,)
    #Q6 = np.sqrt(4 * np.pi / 13 * q6_squared_sum)
    
    # Apply bias exponent
    return Q6 ** gamma


def calculate_order_parameter(coords, dists, idxs, kdtree, side_of_box, gamma):
    """
    Bridge function to calculate order parameters with proper PBC handling.
    
    Automatically detects dimensionality and calls the appropriate function:
    - 2D coordinates → hexatic ψ₆
    - 3D coordinates → Steinhardt Q₆
    
    Args:
        coords (np.ndarray): Trial molecule coordinates, shape (N, 2) or (N, 3)
        dists (np.ndarray): Distances to neighbors, shape (N, k)
        idxs (np.ndarray): Neighbor indices, shape (N, k)
        kdtree: KDTree object with .data attribute containing all coordinates
        side_of_box (float or np.ndarray): Box side length(s). 
                                           Scalar for cubic, array [Lx, Ly, Lz] for orthorhombic
        gamma (float): Bias exponent (from gamma_dict)
    
    Returns:
        np.ndarray: Order parameter values (ψ₆ or Q₆), shape (N,)
    """
    N_data = kdtree.data.shape[0]
    #N_coords = coords.shape[0]
    #k = idxs.shape[1]
    ndim = coords.shape[1]
    
    # Ensure side_of_box is an array for consistent handling
    if np.isscalar(side_of_box):
        box_array = np.array([side_of_box] * ndim)
    else:
        box_array = np.asarray(side_of_box)
    
    # Clamp indices to prevent out-of-bounds access
    safe_idxs = np.minimum(idxs, N_data - 1)  # (N, k)
    
    # Retrieve all neighbor coordinates in one operation
    all_neigh_coords = kdtree.data[safe_idxs]  # (N, k, ndim)
    
    # Create mask for valid (finite) neighbors
    finite_mask = np.isfinite(dists)  # (N, k)
    
    # Calculate displacement vectors with PBC
    if ndim == 2:
        # 2D case
        deltas = all_neigh_coords[..., :2] - coords[:, None, :2]  # (N, k, 2)
        deltas -= box_array[:2] * np.round(deltas / box_array[:2])
        return calculate_psi_6_2d(deltas, finite_mask, gamma)
    
    elif ndim == 3:
        # 3D case
        deltas = all_neigh_coords - coords[:, None, :]  # (N, k, 3)
        deltas -= box_array * np.round(deltas / box_array)
        return calculate_q6_3d(deltas, finite_mask, gamma)
    
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}. Expected 2 or 3.")
    
# ============================================================================
# NEIGHBOR SEARCH AND FILTERING
# ============================================================================

# ============================================================================
# 3D G3 AUXILIARY POTENTIAL FUNCTIONS
# ============================================================================

def calculate_g3_3d(coords, dists, idxs, kdtree, gamma):
    """
    Vectorized g3 calculation for 3D using nearest neighbor as reference.
    
    Args:
        coords: Trial molecule coordinates (N, 3)
        dists: Distances to neighbors (N, k)
        idxs: Neighbor indices (N, k)
        kdtree: KDTree of existing molecules
        gamma: Gamma parameter for scaling
    
    Returns:
        g_r: Product of g3 values for each trial molecule (N,)
        ref_coords: Coordinates of reference neighbors (N, 3)
    """
    N, k = dists.shape
    box_array = np.array([side_of_box, side_of_box, side_of_box])
    N_data = kdtree.data.shape[0]
    safe_idxs = np.minimum(idxs, N_data - 1)
    all_neigh_coords = kdtree.data[safe_idxs]  # (N, k, 3)
    
    # Calculate displacement vectors with PBC
    deltas = all_neigh_coords - coords[:, None, :]
    deltas -= box_array * np.round(deltas / box_array)
    
    # Reference: B->A (nearest neighbor, index 0)
    vec_BA = deltas[:, 0, :]  # (N, 3)
    ref_coords = all_neigh_coords[:, 0, :]  # Store actual coordinates
    norm_BA = np.linalg.norm(vec_BA, axis=1, keepdims=True)  # (N, 1)
    
    # All other: B->C (indices 1 to k-1)
    vec_BC = deltas[:, 1:, :]  # (N, k-1, 3)
    
    # cos(θ) = (BA·BC) / (|BA||BC|)
    dot_prod = np.sum(vec_BA[:, None, :] * vec_BC, axis=2)  # (N, k-1)
    norm_BC = dists[:, 1:]  # (N, k-1)
    cos_theta = dot_prod / (norm_BA * norm_BC)  # (N, k-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    
    # Interpolate g3(rBC, cos_theta) - vectorized
    rBC = dists[:, 1:].flatten()
    cos_theta_flat = cos_theta.flatten()
    points = np.column_stack([rBC, cos_theta_flat])  # (N*(k-1), 2)
    
    g3_flat = g3_interpolator(points)  # (N*(k-1),)
    g3_values = g3_flat.reshape(N, k-1)  # (N, k-1)
    
    # Ensure no zeros for non-overlapping molecules
    min_g3 = 1.0 / no_of_trial_molecules
    g3_values = np.where(g3_values < min_g3, min_g3, g3_values)
    
    # Apply gamma and product
    if gamma != 1.0:
        g3_values = g3_values ** gamma
    
    g_r = np.prod(g3_values, axis=1)  # (N,) - product over k-1 neighbors
    
    return g_r, ref_coords

def calculate_g3_pairwise(trial_mol_set, new_neighbor_coord, sq_distances, gamma):
    """
    Calculate g3 for adding a new neighbor to existing trial molecules.
    
    Args:
        trial_mol_set: TrialMoleculeSet with ref_neighbor_coord
        new_neighbor_coord: Coordinate of new neighbor being added
        sq_distances: Squared distances to new neighbor (after masking)
        gamma: Gamma parameter for scaling
    
    Returns:
        g3_values: g3 value for each trial molecule (N,)
    """
    ref_coords = trial_mol_set.ref_neighbor_coord  # (N, 3) - actual coordinates
    trial_coords = trial_mol_set.coords  # (N, 3)
    
    # Vector B->A (reference)
    vec_BA = ref_coords - trial_coords
    vec_BA -= side_of_box * np.round(vec_BA / side_of_box)  # PBC
    
    # Vector B->C (new neighbor)
    vec_BC = new_neighbor_coord - trial_coords
    vec_BC -= side_of_box * np.round(vec_BC / side_of_box)  # PBC
    
    # cos(θ) = (BA·BC) / (|BA||BC|)
    dot_prod = np.sum(vec_BA * vec_BC, axis=1)
    norm_BA = np.linalg.norm(vec_BA, axis=1)
    norm_BC = np.sqrt(sq_distances)
    cos_theta = dot_prod / (norm_BA * norm_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    
    # Lookup g3(rBC, cos_theta)
    points = np.column_stack([norm_BC, cos_theta])
    g3_values = g3_interpolator(points)
    
    # Ensure no zeros for non-overlapping molecules
    min_g3 = 1.0 / no_of_trial_molecules
    g3_values = np.where(g3_values < min_g3, min_g3, g3_values)
    
    if gamma != 1.0:
        g3_values = g3_values ** gamma
    
    return g3_values


# ============================================================================
# BACKTRACKING CALCULATIONS
# ============================================================================
 
"""
Optimized k-clique counting for periodic disk/sphere systems.
Vectorized fast paths for k<=3, kClist with optimized ordering for k>=4.

References
----------
Danisch, Blin, Sagot. "Listing k-cliques in Sparse Real-World Graphs." WWW 2018.
Chiba, Nishizeki. "Arboricity and Subgraph Listing Algorithms." SIAM J. Comput. 1985.
"""

"""
Optimized k-clique counting for periodic disk/sphere systems.
Vectorized fast paths for k<=3, kClist with optimized ordering for k>=4.

References
----------
Danisch, Blin, Sagot. "Listing k-cliques in Sparse Real-World Graphs." WWW 2018.
Chiba, Nishizeki. "Arboricity and Subgraph Listing Algorithms." SIAM J. Comput. 1985.
"""

# ── adjacency construction ──────────────────────────────────────────

def _build_adj(centers, radius, box):
    n = len(centers)
    shifted = np.mod(centers, box)
    tree = KDTree(shifted, boxsize=box)
    pairs = tree.query_pairs(r=2.0 * radius - 1e-12, output_type="ndarray")
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    if len(pairs) > 0:
        adj[pairs[:, 0], pairs[:, 1]] = False
        adj[pairs[:, 1], pairs[:, 0]] = False
    return adj, shifted


# ── ordering ────────────────────────────────────────────────────────

def _degeneracy_rank(adj, n):
    """Bucket-queue degeneracy ordering."""
    nbrs = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = [len(nb) for nb in nbrs]
    max_d = max(deg) if n > 0 else 0
    buckets = [set() for _ in range(max_d + 1)]
    for v in range(n):
        buckets[deg[v]].add(v)
    rank = np.empty(n, dtype=np.int64)
    removed = bytearray(n)
    min_d = 0
    for i in range(n):
        while not buckets[min_d]:
            min_d += 1
        v = buckets[min_d].pop()
        rank[v] = i
        removed[v] = 1
        for w in nbrs[v]:
            if not removed[w]:
                old = deg[w]
                buckets[old].discard(w)
                deg[w] = old - 1
                buckets[old - 1].add(w)
                if old - 1 < min_d:
                    min_d = old - 1
    return rank


def _degree_rank(adj, n):
    """Simple degree sort. One vectorized call."""
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(adj.sum(axis=1))] = np.arange(n)
    return rank


# ── small subgraph: count + find ────────────────────────────────────

def _count_sub(A, k):
    """Count k-cliques in a small bool adjacency matrix."""
    d = A.shape[0]
    if d < k or k <= 0:
        return 0
    if k == 1:
        return d
    if k == 2:
        return int(np.count_nonzero(A)) // 2
    if k == 3:
        Af = A.astype(np.float64) if A.dtype != np.float64 else A
        return int(np.sum(Af * (Af @ Af))) // 6
    total = 0
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1:])[0] + (i + 1)
        if fwd.size >= k - 1:
            total += _count_sub(A[np.ix_(fwd, fwd)], k - 1)
    return total


def _find_sub(A, k):
    """Find one k-clique in a small adjacency. Returns index list or None."""
    d = A.shape[0]
    if d < k or k <= 0:
        return None
    if k == 1:
        return [0]
    if k == 2:
        for i in range(d - 1):
            c = np.argmax(A[i, i + 1:])
            if A[i, i + 1 + c]:
                return [i, i + 1 + c]
        return None
    if k == 3:
        Af = A.astype(np.float64) if A.dtype != np.float64 else A
        M = Af * (Af @ Af)
        for i in range(d - 1):
            row = M[i, i + 1:]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(A[i, :j] & A[j, :j])[0]
            if len(cn) == 0:
                cn = np.where(A[i, j + 1:] & A[j, j + 1:])[0]
                if len(cn) > 0:
                    return [i, j, int(cn[0]) + j + 1]
            else:
                return [int(cn[0]), i, j]
        return None
    for i in range(d - k + 1):
        fwd = np.where(A[i, i + 1:])[0] + (i + 1)
        if fwd.size >= k - 1:
            inner = _find_sub(A[np.ix_(fwd, fwd)], k - 1)
            if inner is not None:
                return [i] + [int(fwd[x]) for x in inner]
    return None


# ── main kClist (k >= 4) ───────────────────────────────────────────

def _kClist_count_and_find(adj, k):
    n = adj.shape[0]
    if n < k:
        return 0, None

    rank = _degree_rank(adj, n) if k == 4 else _degeneracy_rank(adj, n)

    total = 0
    sample = None
    for v in range(n):
        fwd = np.where(adj[v] & (rank > rank[v]))[0]
        if fwd.size < k - 1:
            continue
        sub = adj[np.ix_(fwd, fwd)]
        c = _count_sub(sub, k - 1)
        total += c
        if sample is None and c > 0:
            inner = _find_sub(sub, k - 1)
            if inner is not None:
                sample = tuple(sorted([v] + [int(fwd[i]) for i in inner]))
    return total, sample


# ── dispatch ────────────────────────────────────────────────────────

def _count_from_adj(adj, k):
    n = adj.shape[0]
    if k <= 0:
        return 1, None
    if n < k:
        return 0, None
    if k == 1:
        return (n, (0,)) if n > 0 else (0, None)
    if k == 2:
        count = int(np.count_nonzero(adj)) // 2
        if count == 0:
            return 0, None
        r0 = int(np.argmax(adj.any(axis=1)))
        c0 = int(np.argmax(adj[r0]))
        return count, tuple(sorted((r0, c0)))
    if k == 3:
        Af = adj.astype(np.float64)
        M = Af * (Af @ Af)
        count = int(np.sum(M)) // 6
        if count == 0:
            return 0, None
        # Find one triangle directly from M (reuse the matmul)
        for i in range(n - 1):
            row = M[i, i + 1:]
            j = int(np.argmax(row)) + i + 1
            if M[i, j] < 0.5:
                continue
            cn = np.where(adj[i, :j] & adj[j, :j])[0]
            if len(cn) > 0:
                return count, tuple(sorted((int(cn[0]), i, j)))
            cn = np.where(adj[i, j + 1:] & adj[j, j + 1:])[0]
            if len(cn) > 0:
                return count, tuple(sorted((i, j, int(cn[0]) + j + 1)))
        return count, None
    return _kClist_count_and_find(adj, k)


# ── public API ──────────────────────────────────────────────────────

def _parse_box(box_lengths, dim):
    box = np.atleast_1d(np.asarray(box_lengths, dtype=np.float64))
    if box.size == 1:
        box = np.full(dim, float(box[0]))
    return box


def count_and_sample(centers, k, radius=0.5, *, box_lengths):
    """Count non-overlapping k-disk configurations and return one example.

    Returns
    -------
    (count, sample) where sample is a sorted tuple of indices, or None.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    box = _parse_box(box_lengths, centers.shape[1] if n > 0 else 1)

    if k <= 0 or n < k:
        return 0, None
    if k == 1:
        return n, ((0,) if n > 0 else None)

    adj, _ = _build_adj(centers, radius, box)
    return _count_from_adj(adj, k)


def count_and_sample_branch(centers, k, radius=0.5, *, box_lengths):
    """Count k-cliques and (k+1)-cliques over the same centers with one
    adjacency build.

    Returns
    -------
    (count_k, count_k1, sample_k, sample_k1)
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    dim = centers.shape[1] if n > 0 else 1
    box = _parse_box(box_lengths, dim)

    if k <= 0:
        return 0, 0, None, None

    adj, _ = _build_adj(centers, radius, box)

    count_k, sample_k = _count_from_adj(adj, k)
    count_k1, sample_k1 = _count_from_adj(adj, k + 1)

    return count_k, count_k1, sample_k, sample_k1

# ============================================================================
# NEIGHBOR SEARCH AND ENERGY WEIGHT CALCULATIONS
# ============================================================================

def find_neighbors_fast(
    kdtree: cKDTree,
    trial_mols: np.ndarray,
    lower_cutoff: float,
    upper_cutoff: float,
    no_neigh: int,
    keep_mask = None,
    num_mols = None,
    choice = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns:
      - out (Nx4): x, y, z, g(r)
      - psi6 (N,) : local ψ₆ order parameter for each point
    """

    # 1) Query k nearest *with* an upper bound = 1.5σ
    distances, indices = kdtree.query(trial_mols, k=no_neigh, distance_upper_bound=upper_cutoff)

    # 2) Remove any that overlap the system
    good_mask = np.all(distances >= lower_cutoff, axis=1)
    if not np.any(good_mask):
        #logging.debug(f'{len(trial_mols)} trial molecules  have {np.sum(good_mask)} non-overlapping neighbors.')
        return None, None
    # else:
    #     logging.debug(f'{len(trial_mols)} trial molecules have {np.sum(good_mask)} non-overlapping neighbors.')

    #print('disdt_before', distances)
    coords  = trial_mols[good_mask]     # (N,3)
    dists   = distances[good_mask]      # (N,k)
    idxs    = indices[good_mask]        # (N,k)

    if keep_mask is not None: 
        keep_mask = keep_mask[good_mask]

        assert len(keep_mask) == len(coords)

        #print("keep mask", np.sum(good_mask)-np.sum(keep_mask), 12)

    # 3) Bulk g(r)
    if aux_pot_flag:

        if two_dim_flag:
            
            if use_LUT:
                g_r = interaction_lut(dists, gamma_dict[1])
            else:
                g_r = calculate_interaction_energy(dists, gamma_dict[1], len(molecules_xyz), choice)

            trial_mol_set = TrialMoleculeSet(coords=coords, energy_weight=g_r)
        else:
            g_r, ref_coords = calculate_g3_3d(coords, dists, idxs, kdtree, gamma_dict[1])
            trial_mol_set = TrialMoleculeSet(coords=coords, energy_weight=g_r, ref_neighbor_coord=ref_coords)
        
        return trial_mol_set, keep_mask

    elif reweighting_flag:

        if two_dim_flag:
            
            cos_sum, sin_sum, psi_counts = calculate_order_parameter(coords, dists, idxs, kdtree, side_of_box, gamma_dict[1])
            g_r = np.ones(coords.shape[0], float)
            
            assert coords.shape[0] == g_r.shape[0] == cos_sum.shape[0] == sin_sum.shape[0] == psi_counts.shape[0], f"coords: {coords.shape}, g_r: {cos_sum.shape}"

            return TrialMoleculeSet(coords=coords, energy_weight=g_r, cossum=cos_sum, sinsum=sin_sum, counts=psi_counts), keep_mask

        else:

            g_r = calculate_order_parameter(coords, dists, idxs, kdtree, side_of_box, gamma_dict[1])

        pass

        #assert coords.shape[0] == psi_6.shape[0], f"coords: {coords.shape}, g_r: {psi_6.shape}"

    elif lj_pot_flag:

        g_r = calculate_interaction_energy(dists, gamma_dict[1], len(molecules_xyz), choice)

        # Tail for N -> N-1. Using the current count N this is
        # -lj_tail_pref*(2N-1), the exact inverse of the insertion term
        # for the M-1 <-> M pair, preserving detailed balance.
        # N_cur  = len(molecules_xyz)
        # delta_E = -lj_tail_pref * (2.0 * N_cur - 1.0) / Temp
        # g_r *= np.exp(-BETA * delta_E)

        N_cur = len(molecules_xyz)
        if choice == "add":
            dU_tail = lj_tail_pref * (2.0 * N_cur + 1.0)   # N -> N+1
        else:                                              # remove
            dU_tail = -lj_tail_pref * (2.0 * N_cur - 1.0)  # N -> N-1
        g_r = g_r * np.exp(-BETA * dU_tail)

        trial_mol_set = TrialMoleculeSet(coords=coords, energy_weight=g_r)
    
    else:
        #psi6 = np.ones_like(g_r, float)  # If ψ₆ is not used, return an array of ones
        g_r = np.ones(coords.shape[0], float)  # If ψ₆ is not used, return an array of ones

    # —— assemble your old 4-col result and return together —— #

    assert coords.shape[0] == g_r.shape[0], f"coords: {coords.shape}, g_r: {g_r.shape}"

    trial_mol_set = TrialMoleculeSet(coords=coords, energy_weight=g_r)

    return trial_mol_set, keep_mask

# 2d and 3d centers are different. implement
def create_cavity(cell_border_dims_array, box_dims, two_dim_flag):

    side_of_sheet = box_dims[0]
    
    [x_lower, x_upper, y_lower, y_upper] = cell_border_dims_array[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_border_dims_array[1]
    
    cavity_center_x = None
    cavity_center_x_left = None
    cavity_center_x_right = None

    if x_split_left == x_split_right:
        cavity_center_x = np.random.default_rng().uniform(low=x_lower,
                                                          high=x_upper,
                                                          size=1)[0]

    elif x_split_left == 0 and x_split_right == side_of_sheet:
               
        cavity_center_x_left = np.random.default_rng().uniform(low=x_lower,
                                                            high=x_split_right,
                                                            size=1)[0]
                 
        cavity_center_x_right = np.random.default_rng().uniform(low=x_split_left,
                                                            high=x_upper,
                                                            size=1,
                                                            )[0]
    else: raise ValueError(x_split_left, x_split_right)

    if cavity_center_x_left != None and cavity_center_x_right != None:

        #If not done like this, the small sliver of space created by cutting off the cell will be sampled more

        cell_side_1 = x_upper-x_split_left
        cell_side_2 = x_split_right-x_lower

        right_pecentage = cell_side_1/(cell_side_1+cell_side_2)
        left_pecentage = cell_side_2/(cell_side_1+cell_side_2)

        #assert right_pecentage + left_pecentage == 1, str(right_pecentage) + " " + str(left_pecentage)
        
        cavity_center_x = np.random.choice([cavity_center_x_left, cavity_center_x_right],
                                           p=[left_pecentage, right_pecentage])
    
    #assert cavity_center_x !=  None

    cavity_center_y = None
    cavity_center_y_left = None
    cavity_center_y_right = None

    if y_split_left == y_split_right:
            
        cavity_center_y = np.random.default_rng().uniform(low=y_lower,
                                                        high=y_upper,
                                                        size=1)[0]

    elif y_split_left ==0 and y_split_right == side_of_sheet:

        cavity_center_y_left = np.random.default_rng().uniform(low=y_lower,
                                                            high=y_split_right,
                                                            size=1)[0]
                 
        cavity_center_y_right = np.random.default_rng().uniform(low=y_split_left,
                                                            high=y_upper,
                                                            size=1)[0]
        
    else: raise ValueError(y_split_left, y_split_right)

    if cavity_center_y_left != None and cavity_center_y_right != None:
        
        cell_side_1 = y_upper-y_split_left
        cell_side_2 = y_split_right-y_lower

        right_pecentage = cell_side_1/(cell_side_1+cell_side_2)
        left_pecentage = cell_side_2/(cell_side_1+cell_side_2)

        #assert right_pecentage + left_pecentage == 1, str(right_pecentage) + " " + str(left_pecentage)
    
        cavity_center_y = np.random.choice([cavity_center_y_left, cavity_center_y_right],
                                           p=[left_pecentage, right_pecentage])

    #assert cavity_center_y !=  None

    cavity_center = np.array([cavity_center_x, cavity_center_y])

    return cavity_center


def neighbor_indices_in_cavity(molecules_xyz, cavity_center, cavity_radius, side_of_box):
    # X mask
    cx = cavity_center[0]
    if cx - cavity_radius < 0:
        lo1, hi1 = 0, cx + cavity_radius
        lo2, hi2 = cx - cavity_radius + side_of_box, side_of_box
        mask_x = ((molecules_xyz[:,0] >= lo1) & (molecules_xyz[:,0] <= hi1))
        mask_x |= ((molecules_xyz[:,0] >= lo2) & (molecules_xyz[:,0] <= hi2))
    elif cx + cavity_radius > side_of_box:
        lo1, hi1 = cx - cavity_radius, side_of_box
        lo2, hi2 = 0, cx + cavity_radius - side_of_box
        mask_x = ((molecules_xyz[:,0] >= lo1) & (molecules_xyz[:,0] <= hi1))
        mask_x |= ((molecules_xyz[:,0] >= lo2) & (molecules_xyz[:,0] <= hi2))
    else:
        lo, hi = cx - cavity_radius, cx + cavity_radius
        mask_x = (molecules_xyz[:,0] >= lo) & (molecules_xyz[:,0] <= hi)

    # Y mask
    cy = cavity_center[1]
    if cy - cavity_radius < 0:
        lo1, hi1 = 0, cy + cavity_radius
        lo2, hi2 = cy - cavity_radius + side_of_box, side_of_box
        mask_y = ((molecules_xyz[:,1] >= lo1) & (molecules_xyz[:,1] <= hi1))
        mask_y |= ((molecules_xyz[:,1] >= lo2) & (molecules_xyz[:,1] <= hi2))
    elif cy + cavity_radius > side_of_box:
        lo1, hi1 = cy - cavity_radius, side_of_box
        lo2, hi2 = 0, cy + cavity_radius - side_of_box
        mask_y = ((molecules_xyz[:,1] >= lo1) & (molecules_xyz[:,1] <= hi1))
        mask_y |= ((molecules_xyz[:,1] >= lo2) & (molecules_xyz[:,1] <= hi2))
    else:
        lo, hi = cy - cavity_radius, cy + cavity_radius
        mask_y = (molecules_xyz[:,1] >= lo) & (molecules_xyz[:,1] <= hi)

    if two_dim_flag:
            mask_z  = np.ones(len(molecules_xyz), dtype=bool)  # In 2D, z is always 0, so all points are in the cavity in z direction.
    else:
        # Z mask
        cz = cavity_center[2]
        if cz - cavity_radius < 0:
            lo1, hi1 = 0, cz + cavity_radius
            lo2, hi2 = cz - cavity_radius + side_of_box, side_of_box
            mask_z = ((molecules_xyz[:,2] >= lo1) & (molecules_xyz[:,2] <= hi1))
            mask_z |= ((molecules_xyz[:,2] >= lo2) & (molecules_xyz[:,2] <= hi2))

        elif cz + cavity_radius > side_of_box:
            lo1, hi1 = cz - cavity_radius, side_of_box
            lo2, hi2 = 0, cz + cavity_radius - side_of_box
            mask_z = ((molecules_xyz[:,2] >= lo1) & (molecules_xyz[:,2] <= hi1))
            mask_z |= ((molecules_xyz[:,2] >= lo2) & (molecules_xyz[:,2] <= hi2))

        else:
            lo, hi = cz - cavity_radius, cz + cavity_radius
            mask_z = (molecules_xyz[:,2] >= lo) & (molecules_xyz[:,2] <= hi)

    # combine all three masks
    in_cavity = mask_x & mask_y & mask_z

    # return the indices of True entries
    return np.nonzero(in_cavity)[0]


def neighbor_indices_in_cavity_parallel(molecules_xyz, cavity_center, cavity_radius,
                                cell_dims, box_dims, two_dim_flag):
    """
    Find indices of molecules inside a square cavity, with PBC that respects
    cell boundaries in parallel decomposition.
 
    For a non-split cell [x_lower, x_upper] the cavity wraps at cell edges
    with period = cell_length.  For a split cell (straddles global box edge)
    the cavity wraps at [0, side_of_box].
 
    Args:
        molecules_xyz : (N, 2) or (N, 3) active molecule coordinates.
        cavity_center : (2,) or (3,) center of the square cavity.
        cavity_radius : half-side of the square cavity.
        cell_dims     : [[x_lo, x_hi, y_lo, y_hi],
                         [x_split_l, x_split_r, y_split_l, y_split_r]]
        box_dims      : [side_of_box, side_of_box] or [s, s, s].
        two_dim_flag  : bool — skip z-axis if True.
 
    Returns:
        1-D int array of indices into molecules_xyz that fall inside the cavity.
    """
    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]
    side_of_box = box_dims[0]
 
    # ---- X axis ----
    mask_x = _axis_mask(molecules_xyz[:, 0], cavity_center[0], cavity_radius,
                        x_lower, x_upper, x_split_left, x_split_right,
                        side_of_box)
 
    # ---- Y axis ----
    mask_y = _axis_mask(molecules_xyz[:, 1], cavity_center[1], cavity_radius,
                        y_lower, y_upper, y_split_left, y_split_right,
                        side_of_box)
 
    # ---- Z axis ----
    if two_dim_flag:
        mask_z = np.ones(len(molecules_xyz), dtype=bool)
    else:
        # Decomposition is 2-D; cells span full z range → global PBC
        cz = cavity_center[2]
        lo, hi = cz - cavity_radius, cz + cavity_radius
        if lo < 0:
            mask_z = ((molecules_xyz[:, 2] >= 0) & (molecules_xyz[:, 2] <= hi))
            mask_z |= ((molecules_xyz[:, 2] >= lo + side_of_box) & (molecules_xyz[:, 2] <= side_of_box))
        elif hi > side_of_box:
            mask_z = ((molecules_xyz[:, 2] >= lo) & (molecules_xyz[:, 2] <= side_of_box))
            mask_z |= ((molecules_xyz[:, 2] >= 0) & (molecules_xyz[:, 2] <= hi - side_of_box))
        else:
            mask_z = (molecules_xyz[:, 2] >= lo) & (molecules_xyz[:, 2] <= hi)
 
    return np.nonzero(mask_x & mask_y & mask_z)[0]
 
 
def _axis_mask(coords, center, radius,
               lo_bound, hi_bound, split_left, split_right,
               side_of_box):
    """
    Build a boolean mask for one axis.
 
    Non-split cell:  wraps at [lo_bound, hi_bound]  with period = hi_bound - lo_bound
    Split cell:      wraps at [0, side_of_box]       with period = side_of_box
    """
    lo = center - radius
    hi = center + radius
 
    if split_left == split_right:
        # ---- Non-split cell ----
        cell_length = hi_bound - lo_bound
        if lo < lo_bound:
            # Underflow wraps to upper end of cell
            mask = ((coords >= lo_bound) & (coords <= hi))
            mask |= ((coords >= lo + cell_length) & (coords <= hi_bound))
        elif hi > hi_bound:
            # Overflow wraps to lower end of cell
            mask = ((coords >= lo) & (coords <= hi_bound))
            mask |= ((coords >= lo_bound) & (coords <= hi - cell_length))
        else:
            mask = (coords >= lo) & (coords <= hi)
 
    elif split_left == 0 or split_right == side_of_box:
        # ---- Split cell: wraps at global box edges ----
        if lo < 0:
            mask = ((coords >= 0) & (coords <= hi))
            mask |= ((coords >= lo + side_of_box) & (coords <= side_of_box))
        elif hi > side_of_box:
            mask = ((coords >= lo) & (coords <= side_of_box))
            mask |= ((coords >= 0) & (coords <= hi - side_of_box))
        else:
            mask = (coords >= lo) & (coords <= hi)
 
    else:
        raise ValueError(f"Invalid split: split_left={split_left}, split_right={split_right}")
 
    return mask

def apply_pbc_array(trial_x, trial_y, cell_dims, box_dims):

    side_of_box = box_dims[0]

    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]

    if x_split_left == x_split_right:

        # cell_length = side_of_box for serial run                
        cell_length = x_upper-x_lower
        assert np.round(cell_length,4) == np.round(cell_length_global,4), str(cell_length) + " " + str(cell_length_global)

        assert cell_length >= 0

        trial_x[trial_x <= x_lower] += cell_length
        trial_x[trial_x >= x_upper] -= cell_length

        #assert np.all(trial_x >= x_lower) and np.all(trial_x <= x_upper), str(x_upper) + " " + str(x_lower)+ "-" +str(trial_x[trial_x <= x_lower]) + "+" + str(trial_x[trial_x >= x_upper])

    # This case is not used in serial run
    elif x_split_left == 0 or x_split_right == side_of_box:

        #raise ValueError(x_split_left, x_split_right)
    
        cell_length = x_lower-x_upper
        
        assert cell_length >= 0
        assert np.round(cell_length,4) == np.round(side_of_box - cell_length_global,4), str(cell_length) + " " + str(side_of_box - cell_length_global)

        trial_x[trial_x <= 0] += side_of_box
        trial_x[trial_x >= side_of_box] -= side_of_box

        mask_pbc_x_1 = (side_of_box/2 >= trial_x) & (trial_x >= x_upper)
        trial_x[mask_pbc_x_1] += cell_length

        mask_pbc_x_2 = (side_of_box/2 <= trial_x) & (trial_x <= x_lower)
        trial_x[mask_pbc_x_2] -= cell_length
        
        # The order is important. Should not be applied before cell_length pbc
        trial_x[trial_x <= 0] += side_of_box
        trial_x[trial_x >= side_of_box] -= side_of_box

    else: raise ValueError(x_split_left, x_split_right)

    if y_split_left == y_split_right:

        cell_length = y_upper-y_lower

        assert cell_length >= 0
        assert np.round(cell_length,4) == np.round(cell_length_global,4), str(cell_length) + " " + str(cell_length_global)

        trial_y[trial_y <= y_lower] += cell_length
        trial_y[trial_y >= y_upper] -= cell_length

        #assert np.all(trial_y >= y_lower) and np.all(trial_y <= y_upper), str(y_upper) + " " + str(y_lower) + "-" +str(trial_y[trial_y <= y_lower]) + "+" + str(trial_y[trial_y >= y_upper])

    # This case is not used in serial run
    elif  y_split_left == 0 or y_split_right == side_of_box:

        #raise ValueError(y_split_left, y_split_right)

        cell_length = y_lower-y_upper

        assert cell_length >= 0
        assert np.round(cell_length,4) == np.round(side_of_box - cell_length_global,4), str(cell_length) + " " + str(side_of_box - cell_length_global)

        trial_y[trial_y <= 0] += side_of_box
        trial_y[trial_y >= side_of_box] -= side_of_box            

        mask_pbc_y_1 = (side_of_box/2 >= trial_y) & (trial_y >= y_upper)
        trial_y[mask_pbc_y_1] += cell_length

        mask_pbc_y_2 = (side_of_box/2 <= trial_y) & (trial_y <= y_lower)
        trial_y[mask_pbc_y_2] -= cell_length

        # The order is important. Should not be applied before cell_length pbc
        trial_y[trial_y <= 0] += side_of_box
        trial_y[trial_y >= side_of_box] -= side_of_box            

    else: raise ValueError(y_split_left, y_split_right)            
    
    #Make sure all trial molecules are inside the box
    assert np.all(trial_x >= 0) and np.all(trial_x <= side_of_box), str(np.all(trial_x >= 0)) + " " + str(np.all(trial_x <= side_of_box))
    assert np.all(trial_y >= 0) and np.all(trial_y <= side_of_box), str(np.all(trial_y >= 0)) + " " + str(np.all(trial_y <= side_of_box))

    return trial_x, trial_y

def apply_PBC_single(moved_disk_x, moved_disk_y, cell_dims, box_dims):
    
    side_of_box = box_dims[0]

    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    #x and y coordinates of the middle of the box
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]

    if x_split_left == x_split_right:
        
        #assert x_lower <= x_upper, str(x_lower) + " <= " + str(x_upper)

        cell_length = x_upper-x_lower
        #assert cell_length >= 0

        if moved_disk_x <= x_lower: moved_disk_x += cell_length
        if moved_disk_x >= x_upper: moved_disk_x -= cell_length

        #assert 0 <= moved_disk_x <= side_of_box, "x " + str(0) + " <= " + str(moved_disk_x) + " <= " + str(side_of_box) + " random_x " + str(random_disk_xyz[0])    
        #assert x_lower <= moved_disk_x <= x_upper, str(x_upper) + " " + str(x_lower)+ " - " +str(moved_disk_x) + " random_x " + str(random_disk_xyz[0]) 

    elif x_split_left == 0 or x_split_right == side_of_box:

        #assert x_lower >= x_upper, str(x_lower) + " >= " + str(x_upper)

        cell_length = x_lower-x_upper #(side_of_box-x_lower) + (x_upper-0)
        #assert cell_length >= 0

        if moved_disk_x <= 0:           moved_disk_x += side_of_box
        if moved_disk_x >= side_of_box: moved_disk_x -= side_of_box

        if side_of_box/2 >= moved_disk_x >= x_upper : moved_disk_x += cell_length
        if side_of_box/2 <= moved_disk_x <= x_lower : moved_disk_x -= cell_length
        

        #assert 0 <= moved_disk_x <= side_of_box, "x " + str(0) + " <= " + str(moved_disk_x) + " <= " + str(side_of_box) + " random_x " + str(random_disk_xyz[0]) + " +- " + str(cell_length) + " x_lower " + str(x_lower) + " x_upper " + str(x_upper)  
        #assert (0 <= moved_disk_x <= x_upper) or (x_lower <= moved_disk_x <= side_of_box), str(x_upper) + " " + str(x_lower)+ "-" +str(moved_disk_x)+ " random_x " + str(random_disk_xyz[0]) 

    else: raise ValueError(x_split_left, x_split_right)

    if y_split_left == y_split_right:

        #assert y_lower <= y_upper, str(y_lower) + " <= " + str(y_upper)

        cell_length = y_upper-y_lower
        #assert cell_length >= 0

        if moved_disk_y <= y_lower: moved_disk_y += cell_length
        if moved_disk_y >= y_upper: moved_disk_y -= cell_length

        #assert 0 <= moved_disk_y <= side_of_box, "y " + str(0) + " <= " + str(moved_disk_y) + " <= " + str(side_of_box) + " random_y " + str(random_disk_xyz[1])
        #assert y_lower <= moved_disk_y <= y_upper, str(y_upper) + " " + str(y_lower)+ "-" +str(moved_disk_y) + " random_y " + str(random_disk_xyz[1])

    elif  y_split_left == 0 or y_split_right == side_of_box:

        #assert y_lower >= y_upper, str(y_lower) + " >= " + str(y_upper)

        cell_length = y_lower - y_upper #(side_of_box-y_lower) + (y_upper-0)
        #assert cell_length >= 0

        if moved_disk_y <= 0:           moved_disk_y += side_of_box
        if moved_disk_y >= side_of_box: moved_disk_y -= side_of_box

        if side_of_box/2 >= moved_disk_y >= y_upper : moved_disk_y += cell_length
        if side_of_box/2 <= moved_disk_y <= y_lower : moved_disk_y -= cell_length

        #assert 0 <= moved_disk_y <= side_of_box, "y " + str(0) + " <= " + str(moved_disk_y) + " <= " + str(side_of_box) + " random_y " + str(random_disk_xyz[1])  + " +- " + str(cell_length) + " y_lower " + str(y_lower) + " y_upper " + str(y_upper)
        #assert (0 <= moved_disk_y <= y_upper) or (y_lower <= moved_disk_y <= side_of_box), str(y_upper) + " " + str(y_lower)+ "-" +str(moved_disk_y) + " random_y " + str(random_disk_xyz[1])

    else: raise ValueError(y_split_left, y_split_right)

    ###################### Energy evaluation ############################

    #moved_disk_xyz = np.array([moved_disk_x, moved_disk_y, moved_disk_z])

    return moved_disk_x, moved_disk_y

def mc_mapper(input):

    molecules_xyz, border_molecules, mc_cycles, cell_dims, box_dims = input
    
    molecules_xyz, repack_step_counter, repack_accept_counter, add_step_counter, add_attempt_counter, add_accept_counter, remove_step_counter, remove_accept_counter, move_step_counter, move_accept_counter, total_time = run_mc_in_a_cell(molecules_xyz, border_molecules, mc_cycles, cell_dims, box_dims) 

    return [molecules_xyz, repack_step_counter, repack_accept_counter, add_step_counter, add_attempt_counter, add_accept_counter, remove_step_counter, remove_accept_counter, move_step_counter, move_accept_counter, total_time]

# -------------------- Monte Carlo Moves in a Cell --------------------
def run_mc_in_a_cell(molecules_xyz, border_molecules, mc_cycles, cell_dims, box_dims):
    """
    Main MC loop for a single cell.

# ============================================================================
# MONTE CARLO MOVE IMPLEMENTATION
# ============================================================================

    For serial runs, border_molecules is empty. The function performs "move", "add", and "remove"
    moves and applies periodic boundary conditions using the helper function. The KD-tree is rebuilt
    as needed for neighbor searches.
    
    Args:
        molecules_xyz (np.ndarray): Coordinates of all molecules.
        border_molecules (list): Molecules at the boundary (not used in serial runs).
        mc_cycles (int): Total number of Monte Carlo cycles.
        cell_dims (list): Cell dimensions (not critical for serial runs).
        box_dims (list): Box dimensions for PBC.
    
    Returns:
        tuple: (updated molecules_xyz, repack_step_counter, repack_accept_counter, add_step_counter, 
                add_accept_counter, remove_step_counter, remove_accept_counter, move_step_counter, 
                move_accept_counter, total_time)
    """
    molecules_xyz = np.array(molecules_xyz)

    # Initialize counters
    add_accept_counter    = 0
    add_step_counter      = 0
    add_attempt_counter = 0
    move_accept_counter   = 0
    move_step_counter     = 0
    remove_accept_counter = 0
    remove_step_counter   = 0
    repack_accept_counter = 0
    repack_step_counter   = 0
    total_time         = 0
    total_step_counter    = 0

    if PARALLEL:
        #################### Unpack the cell_dims and box_dims
        #Side of box
        side_of_box = box_dims[1]

        #All four corners of the box
        #[x_lower, x_upper, y_lower, y_upper] = cell_dims[0]

        #x and y coordinates of the middle of the box
        #[x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]

    else:
        side_of_box = box_dims[1]
    
        # Unpack cell dimensions (for serial run these are fixed)
        #[x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    
    start_time = time.time()

    while total_step_counter <= mc_cycles:

        # If the system is empty, only adding is possible.
        if len(molecules_xyz) == 0:
            choice = "add"
        else:
            # Choose randomly among "move", "add", "remove" with given probabilities.
            choice = np.random.choice(["move", "add", "remove"], p=[0.5, 0.25, 0.25])
            #choice = np.random.choice(["add", "remove"], p=[0.5, 0.5])
        
        #########
        # choice = "move"
        # total_step_counter += 1
        #########

        if choice == "move" and len(molecules_xyz) > 0:
            if total_step_counter >= burn_in:
                move_step_counter += 1

            # Randomly choose a molecule to move.
            random_disk_index = np.random.choice(len(molecules_xyz))
            random_disk_xyz = np.array(molecules_xyz[random_disk_index])
            # Exclude the selected molecule from neighbor search.
            disk_array_without_random = np.delete(molecules_xyz, random_disk_index, axis=0)

            # Generate random displacement (move_factor * disk radius).
            ranf_1, ranf_2 = np.random.uniform(0, 1, 2)
            moved_disk_x = random_disk_xyz[0] + move_factor * (ranf_1 - 0.5)
            moved_disk_y = random_disk_xyz[1] + move_factor * (ranf_2 - 0.5)

            if two_dim_flag:
                moved_disk_z = 0.0
                moved_disk_xyz = np.array([moved_disk_x, moved_disk_y]) 

            else:
                randf_3 = (ranf_1 + ranf_2) % 1
                moved_disk_z = random_disk_xyz[2] + move_factor * (randf_3 - 0.5)
                #moved_disk_z = np.mod(moved_disk_z, side_of_box) #Apply PBC
                        
            #############
            if PARALLEL:
                moved_disk_x, moved_disk_y = apply_PBC_single(moved_disk_x, moved_disk_y, cell_dims, box_dims)

                if not two_dim_flag: 
                    raise ValueError("3d paralle not  implemented yet")

            else:
                # Apply periodic boundary conditions using modulo arithmetic.
                moved_disk_x = np.mod(moved_disk_x, side_of_box)
                moved_disk_y = np.mod(moved_disk_y, side_of_box)

                if not two_dim_flag: 
                    moved_disk_z = np.mod(moved_disk_z, side_of_box) #Apply PBC


            ##############

            if two_dim_flag:
                moved_disk_xyz = np.array([moved_disk_x, moved_disk_y])   
            else:
                moved_disk_xyz = np.array([moved_disk_x, moved_disk_y, moved_disk_z])

            if PARALLEL:
                system_border_molecules = np.vstack((border_molecules, disk_array_without_random))
            else:
                system_border_molecules = disk_array_without_random

            # Build KD-tree for neighbor search and check for overlaps.
            mv_kdtree = cKDTree(system_border_molecules, boxsize=box_dims)

            if lj_pot_flag:
                # LJ translation needs a real Metropolis test on the energy change,
                # not a bare overlap check (that would sample hard spheres, not LJ).
                d_new, _ = mv_kdtree.query([moved_disk_xyz], k=no_neigh,
                                           distance_upper_bound=upper_cutoff)
                d_new = d_new[0]
                d_new = d_new[np.isfinite(d_new)]

                # hard core at sigma, consistent with the insertion neighbour search
                if d_new.size and np.any(d_new < diameter_of_disk):
                    pass  # overlap -> reject
                else:
                    d_old, _ = mv_kdtree.query([random_disk_xyz], k=no_neigh,
                                               distance_upper_bound=upper_cutoff)
                    d_old = d_old[0]
                    d_old = d_old[np.isfinite(d_old)]

                    def _u_lj(d):
                        if d.size == 0:
                            return 0.0
                        ir6 = (1.0 / d) ** 6
                        return np.sum(4.0 * ir6 * (ir6 - 1.0))

                    dU = _u_lj(d_new) - _u_lj(d_old)   # tail unchanged (N fixed) -> cancels
                    if np.log(np.random.uniform(0, 1)) <= -dU / Temp:
                        molecules_xyz = np.append(disk_array_without_random,
                                                  [moved_disk_xyz], axis=0)
                        if total_step_counter >= burn_in: move_accept_counter += 1
            else:
                neighbour_index = mv_kdtree.query_ball_point([moved_disk_xyz], diameter_of_disk)
                if len(neighbour_index[0]) == 0:
                    molecules_xyz = np.append(disk_array_without_random, [moved_disk_xyz], axis=0)
                    if total_step_counter >= burn_in: move_accept_counter += 1
        else:
            # For add or remove moves, increment corresponding step counter.

            if choice == "remove" and len(molecules_xyz) == 0: continue

            if USE_DEPTH:
                if choice == "add":
                    add_step_counter += 1
                    n_add       = depth +1        #1,2,3 ...
                    n_remove = depth       #0,1,2 ...
                elif choice == "remove":
                    remove_step_counter += 1
                    n_add       = depth              #0,1,2,3 ..
                    n_remove = depth + 1 #1,2,3 ...
                else:
                    raise ValueError(choice)
            else:
                if choice == "add":
                    add_step_counter += 1
                elif choice == "remove":
                    remove_step_counter += 1
                else:
                    raise ValueError(choice)
            
            total_step_counter += 1

            # Every sampling_window steps, record timing and exit the loop.
            if total_step_counter >= sampling_window:

                total_time = time.time() - start_time
                return (molecules_xyz, repack_step_counter, repack_accept_counter,
                        add_step_counter, add_attempt_counter, add_accept_counter, remove_step_counter,
                        remove_accept_counter, move_step_counter, move_accept_counter,
                        total_time)        
    
            if choice == "add":
                
                
                if PARALLEL:
                    [cavity_center_x,  cavity_center_y] = create_cavity(cell_dims, box_dims, two_dim_flag)
                else:
                    [cavity_center_x,  cavity_center_y, cavity_center_z] = np.random.default_rng().uniform(low=0, high=side_of_box, size=3)

                if two_dim_flag:  cavity_center = np.array([cavity_center_x,  cavity_center_y])
                else: cavity_center = np.array([cavity_center_x,  cavity_center_y, cavity_center_z])

            elif choice == "remove":

                if lj_pot_flag:
                    if PARALLEL:
                        [cavity_center_x,  cavity_center_y] = create_cavity(cell_dims, box_dims, two_dim_flag)
                    else:
                        [cavity_center_x,  cavity_center_y, cavity_center_z] = np.random.default_rng().uniform(low=0, high=side_of_box, size=3)
                    
                    cavity_center = np.array([cavity_center_x,  cavity_center_y, cavity_center_z])

                else:
                    cavity_center_index = np.random.choice(np.arange(len(molecules_xyz)), size=1, replace=False)[0]
                    cavity_center = molecules_xyz[cavity_center_index]
            else:
                raise ValueError("Invalid choice for move type.")

            ############## Splitting sysytem into disk inside cavity and everything else 

            if PARALLEL:
                neighbour_index = neighbor_indices_in_cavity_parallel(molecules_xyz, cavity_center, cavity_radius,
                                cell_dims, box_dims, two_dim_flag)
            else:
                neighbour_index = neighbor_indices_in_cavity(molecules_xyz, cavity_center, cavity_radius, side_of_box)
            
            n_cav = len(neighbour_index)

            #if choice == "remove" and n_cav == 0: continue

            #print(f"n_cav {n_cav} n_remove {n_remove} choice {choice}")

            if n_cav == 0: continue

            if USE_DEPTH:

                if n_cav < n_remove:

                    if choice == "add":
                        n_add, n_remove = n_cav, n_cav - 1   #depth=n_cav-1
                    elif choice == "remove":
                        n_add, n_remove = n_cav - 1, n_cav    #depth=n_cav-1
                    else:
                        raise ValueError(f"Unknown choice: {choice!r}")

            else:
                
                #if n_cav == 1: continue

                if choice == "add":
                    n_add, n_remove = n_cav+1, n_cav
                elif choice == "remove":
                    n_add, n_remove = n_cav-1, n_cav
                else:
                    raise ValueError(f"Unknown choice: {choice}")

            #print(f" n_add {n_add} n_rem {n_remove}")

            # if BRANCH_flag:
            #     if n_add == 0: continue

            if n_remove == 0:
                disk_inside_cavity_xyz = np.array([])
                disk_array_without_cavity = molecules_xyz
            else:
                if len(molecules_xyz) == 0:
                    raise ValueError("molecules_xyz is empty but n_remove > 0")
                
                #assert n_remove == len(neighbour_index), f" n_remove {n_remove} neigh idx  {len(neighbour_index)}"

                neighbour_index = np.random.choice(neighbour_index, n_remove, replace=False)

                disk_inside_cavity_xyz = molecules_xyz[neighbour_index]
                mask = np.ones(len(molecules_xyz), dtype=bool)
                mask[neighbour_index] = False
                disk_array_without_cavity = molecules_xyz[mask]

            assert len(disk_array_without_cavity) == len(molecules_xyz) - n_remove
            assert len(disk_inside_cavity_xyz) == n_remove 

            ##############  Generate trial molecules uniformly in the cavity.
            
            dim=2 if two_dim_flag else 3

            trial_nums = np.random.default_rng().uniform(low=-cavity_radius, high=cavity_radius, size=(dim, no_of_trial_molecules))

            trial_x = trial_nums[0] + cavity_center[0]
            trial_y = trial_nums[1] + cavity_center[1]

            if PARALLEL:
                trial_x, trial_y = apply_pbc_array(trial_x, trial_y, cell_dims, box_dims)
            else:
                # Apply PBC to trial molecules.
                trial_x = np.mod(trial_x, side_of_box)
                trial_y = np.mod(trial_y, side_of_box)

            if two_dim_flag:
                trial_molecules_array = np.column_stack((trial_x, trial_y))
            else:
                trial_z = trial_nums[2] + cavity_center[2]
                trial_z = np.mod(trial_z, side_of_box)
                trial_molecules_array = np.column_stack((trial_x, trial_y, trial_z))

            #### Generating neigbour list for trial molecules ####

            keep_mask = np.ones(no_of_trial_molecules, dtype=bool)

            if n_remove != 0:
                indices_to_remove = np.random.choice(
                    no_of_trial_molecules, 
                    size=n_remove, 
                    replace=False
                )
                keep_mask[indices_to_remove] = False

            if PARALLEL:
                system_border_molecules = np.vstack((border_molecules, disk_array_without_cavity))

                # Construct the KD-tree of system molecules with periodic boundary conditions
                kdtree = cKDTree(system_border_molecules, boxsize=box_dims)
            else:
                kdtree = cKDTree(disk_array_without_cavity, boxsize=box_dims)

            trial_mol_set, keep_mask = find_neighbors_fast(
                kdtree = kdtree,
                trial_mols=trial_molecules_array,
                lower_cutoff=diameter_of_disk,
                upper_cutoff=upper_cutoff,
                no_neigh=no_neigh,
                keep_mask = keep_mask,
                num_mols = len(molecules_xyz),
                choice= choice
                )

            if trial_mol_set is None: continue

            if n_remove != 0:
                mol_in_cav_set, _ = find_neighbors_fast(
                    kdtree=kdtree,
                    trial_mols=disk_inside_cavity_xyz,
                    lower_cutoff=diameter_of_disk,
                    upper_cutoff=upper_cutoff,
                    no_neigh=no_neigh,
                    num_mols = len(molecules_xyz),
                    choice= choice
                    )
            else:
                mol_in_cav_set = TrialMoleculeSet(coords=np.array([]), energy_weight=np.array([]))

            assert len(mol_in_cav_set) == n_remove

            if trial_mol_set is not None:
                
                if aux_pot_flag or lj_pot_flag:
                    assert np.all(trial_mol_set.cossum == 1.0)
                    assert np.all(trial_mol_set.sinsum == 1.0)
                    assert np.all(trial_mol_set.counts == 1.0)

                    assert np.all(mol_in_cav_set.cossum == 1.0)
                    assert np.all(mol_in_cav_set.sinsum == 1.0)
                    assert np.all(mol_in_cav_set.counts == 1.0)

                elif reweighting_flag and  two_dim_flag:

                    assert np.all(trial_mol_set.energy_weight == 1.0)
                    assert np.all(mol_in_cav_set.energy_weight == 1.0)


            sum_e_minus_beta_u_ij_current_arr  = []
            sum_e_minus_beta_u_ij_new_arr = []

            if len(disk_inside_cavity_xyz) != 0:

                trial_mol_mol_in_cav_set = trial_mol_set.copy()
                trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.mask(keep_mask) #k - n_remove + n_add
                trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.merge(mol_in_cav_set)  # k + n_remove
                # CRITICAL: Shuffle to avoid ordering bias
                indices = np.random.permutation(len(trial_mol_mol_in_cav_set))
                trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set[indices]

            else:
                trial_mol_mol_in_cav_set = trial_mol_set.copy()

            ############ Adding New Trial molecules ###############

            if n_add != 0:
                
                # All trial moleceules are overlapping with the cavity, W is 0 and step is rejected
                if trial_mol_set is None:  continue
                #if len(trial_mol_mol_in_cav_set) < depth+1: continue

                if len(trial_mol_set) < n_add: continue
                
                #assert  len(trial_mol_mol_in_cav_set) <= no_of_trial_molecules + 1, f"trial_mol_mol_in_cav_set: {len(trial_mol_mol_in_cav_set)}, no_of_trial_molecules: {no_of_trial_molecules + 1}"

                if BRANCH_flag:

                #     if len(disk_inside_cavity_xyz) != 0:

                #          trial_mol_mol_in_cav_set = trial_mol_set.copy()
                #          trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.mask(keep_mask) #k - n_remove + n_add
                #          trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.merge(mol_in_cav_set)  # k + n_remove
                #          # CRITICAL: Shuffle to avoid ordering bias
                #          indices = np.random.permutation(len(trial_mol_mol_in_cav_set))
                #          trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set[indices]

                #     else:
                #          trial_mol_mol_in_cav_set = trial_mol_set.copy()

                    #(num_configs, config_index) = count_and_sample(centers=trial_mol_set.coords, k=n_add, box_lengths=box_dims)
                    (num_configs, config_index) = count_and_sample(centers=trial_mol_mol_in_cav_set.coords, k=n_add, box_lengths=box_dims)

                    if num_configs == 0: continue

                    sum_e_minus_beta_u_ij_new_arr = num_configs

                    #chosen_idx = config_index
                    chosen_disk_coords = np.array([trial_mol_mol_in_cav_set.coords[index] for index in config_index])

                    # Create TrialMoleculeSet with chosen molecules
                    chosen_disk_set = TrialMoleculeSet(
                        coords=np.array(chosen_disk_coords),
                        energy_weight=np.ones(n_add),
                        cossum=None,
                        sinsum=None,
                        counts=None
                    )

                    sum_e_minus_beta_u_ij_new = np.log(sum_e_minus_beta_u_ij_new_arr)

                else:

                    (chosen_disk_set, sum_e_minus_beta_u_ij_new_arr) = rosenbluth_weight_new(trial_mol_mol_in_cav_set, n_add, choice)
                
                    if chosen_disk_set is None:  continue

                    sum_e_minus_beta_u_ij_new = np.sum(np.log(sum_e_minus_beta_u_ij_new_arr))

                    assert len(sum_e_minus_beta_u_ij_new_arr) == n_add, f"sum_e_minus_beta_u_ij_new_arr: {sum_e_minus_beta_u_ij_new_arr}, n_add: {n_add}"

                # Extract data from TrialMoleculeSet
                chosen_disk_array_xyz = chosen_disk_set.coords
                
                if reweighting_flag:
                    
                    cos_sum = chosen_disk_set.cossum
                    sin_sum = chosen_disk_set.sinsum
                    counts = chosen_disk_set.counts

                    # Average the complex phasors, then take magnitude
                    #psi_6 = np.ones(cos_sum.shape[0]) * 0.3 

                    psi_6= np.sqrt( (cos_sum / counts)**2 + (sin_sum / counts)**2)

                    #Start with psi6 of 0.3 which is the psi6 of random arrangement
                    psi_6[psi_6 == 0] = 0.3   
                    psi_6[np.isnan(psi_6)] = 0.3
                    
                    e_minus_beta_u_i_new_arr = psi_6
                    
                else:
                    e_minus_beta_u_i_new_arr = chosen_disk_set.energy_weight
                                
                #print(sum_e_minus_beta_u_ij_new_arr,  np.log(sum_e_minus_beta_u_ij_new_arr), e_minus_beta_u_i_new_arr, np.log(e_minus_beta_u_i_new_arr), "\n")

                e_minus_beta_u_r_new = np.sum(np.log(e_minus_beta_u_i_new_arr))

                if not aux_pot_flag and not reweighting_flag and not lj_pot_flag:
                    assert sum_e_minus_beta_u_ij_new >= 0, sum_e_minus_beta_u_ij_new
                    if not lj_pot_flag: assert e_minus_beta_u_r_new >= 0, e_minus_beta_u_r_new

            else:
                sum_e_minus_beta_u_ij_new = 0.0
                e_minus_beta_u_r_new = 0.0
                sum_e_minus_beta_u_ij_new_arr = np.array([])
                e_minus_beta_u_i_new_arr = np.array([])
                chosen_disk_array_xyz = np.array([])
                
            assert len(chosen_disk_array_xyz) == n_add
            assert len(e_minus_beta_u_i_new_arr) == n_add

            ############### Rosenbluth Weight for Current Configuration ###############
            if n_remove != 0:
                
                if BRANCH_flag:

                        # if len(disk_inside_cavity_xyz) != 0:

                        #     trial_mol_mol_in_cav_set = trial_mol_set.copy()
                        #     trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.mask(keep_mask) #k - n_remove + n_add
                        #     trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set.merge(mol_in_cav_set)  # k + n_remove
                        #     # CRITICAL: Shuffle to avoid ordering bias
                        #     indices = np.random.permutation(len(trial_mol_mol_in_cav_set))
                        #     trial_mol_mol_in_cav_set = trial_mol_mol_in_cav_set[indices]

                        # else:
                        #     trial_mol_mol_in_cav_set = trial_mol_set.copy()

                        (num_configs_curr, _) = count_and_sample(centers=trial_mol_mol_in_cav_set.coords, k=n_remove, box_lengths=box_dims)
                    
                        sum_e_minus_beta_u_ij_current_arr = num_configs_curr
                        e_minus_beta_u_i_current_arr = np.ones(n_remove)

                        sum_e_minus_beta_u_ij_current = np.log(sum_e_minus_beta_u_ij_current_arr)
                   
                else:

                    if trial_mol_set is not None:

                        assert len(mol_in_cav_set) == n_remove

                        assert len(trial_mol_set) == len(keep_mask)

                        trial_mol_n_removed_set = trial_mol_set.mask(keep_mask) #k - n_remove + n_add

                        if len(trial_mol_n_removed_set) == 0: trial_mol_n_removed_set = None

                    (sum_e_minus_beta_u_ij_current_arr,
                    e_minus_beta_u_i_current_arr) = rosenbluth_weight_current(
                        trial_mol_n_removed_set,
                        mol_in_cav_set, n_remove, choice
                    )

                    assert len(sum_e_minus_beta_u_ij_current_arr) == n_remove

                    sum_e_minus_beta_u_ij_current = np.sum(np.log(sum_e_minus_beta_u_ij_current_arr))
                
                e_minus_beta_u_r_current = np.sum(np.log(e_minus_beta_u_i_current_arr))

                if (not aux_pot_flag) and (not reweighting_flag) and (not lj_pot_flag):
                    assert sum_e_minus_beta_u_ij_current >= 0, sum_e_minus_beta_u_ij_current
                    if not lj_pot_flag: assert e_minus_beta_u_r_current >= 0, e_minus_beta_u_r_current
   
            else:
                sum_e_minus_beta_u_ij_current = 0.0
                e_minus_beta_u_r_current = 0.0

            # Acceptance criteria evaluation (using log scale to avoid overflow)
            if choice == "add":

                add_attempt_counter +=1

                assert len(chosen_disk_array_xyz) == n_add
                
                if PARALLEL:
                    mu_ideal = -np.log((cell_length_global*cell_length_global)/(len(molecules_xyz)+1))
                else:
                    mu_ideal = -np.log(Total_volume / (len(molecules_xyz) + 1))         

                if BRANCH_flag:
                    k_trials = -np.log((no_of_trial_molecules-n_remove)/n_add)
                    #k_trials = np.log(n_add/(no_of_trial_molecules-n_remove))

                else:
                    k_trials = -np.log(no_of_trial_molecules-n_remove)
 
                mu_excess_rosenbluth = -sum_e_minus_beta_u_ij_new  + sum_e_minus_beta_u_ij_current - k_trials 

                delta_E = e_minus_beta_u_r_new - e_minus_beta_u_r_current
                
                if lj_pot_flag:
                    delta_E = 0
                else:
                    if (not aux_pot_flag) and (not reweighting_flag) : assert delta_E == 0, delta_E

                log_alpha_add = mu_ideal - (BETA*chem_pot) + mu_excess_rosenbluth + delta_E
                alpha_add = min(1, np.exp(-log_alpha_add))
                log_alpha_add = -np.log(alpha_add)
                
                log_random_number = -np.log(np.random.uniform(0, 1))
                if log_alpha_add <= log_random_number:
                    
                    if total_step_counter >= burn_in: add_accept_counter += 1

                    if len(chosen_disk_array_xyz) == 0: raise ValueError("No molecules to add")

                    old_mol_counter = len(molecules_xyz)

                    molecules_xyz = np.append(disk_array_without_cavity, chosen_disk_array_xyz, axis=0)

                    assert len(molecules_xyz) == old_mol_counter + 1, f"{len(molecules_xyz)} != {old_mol_counter+1}"
                                
            elif choice == "remove":

                if PARALLEL:
                    mu_ideal = -np.log((len(molecules_xyz))/(cell_length_global*cell_length_global))
                else:
                    mu_ideal = -np.log(len(molecules_xyz) / Total_volume)
                
                if BRANCH_flag:
                    #k_trials = np.log((no_of_trial_molecules-n_remove)/n_add) 
                    k_trials = -np.log((no_of_trial_molecules-n_add)/n_remove)
                    #k_trials = np.log(n_remove/(no_of_trial_molecules-n_add))
                else:
                    k_trials = -np.log(no_of_trial_molecules-n_add)

                mu_excess_rosenbluth = -sum_e_minus_beta_u_ij_new + sum_e_minus_beta_u_ij_current + k_trials 

                delta_E = e_minus_beta_u_r_new-e_minus_beta_u_r_current

                if lj_pot_flag:
                    delta_E = 0
                else:
                    if (not aux_pot_flag) and (not reweighting_flag) : assert delta_E == 0, delta_E

                log_alpha_remove = mu_ideal + (BETA*chem_pot) + mu_excess_rosenbluth + delta_E

                alpha_remove = min(1, np.exp(-log_alpha_remove))
                log_alpha_remove = -np.log(alpha_remove)
                
                random_number = -np.log(np.random.uniform(0, 1))
                
                if log_alpha_remove <= random_number:

                    if total_step_counter >= burn_in: remove_accept_counter += 1

                    old_mol_counter = len(molecules_xyz)

                    if len(chosen_disk_array_xyz) == 0:
                        assert n_add == 0, n_add
                        assert len(molecules_xyz)-1 == len(disk_array_without_cavity), len(molecules_xyz)
                        molecules_xyz = disk_array_without_cavity
                    else:
                        molecules_xyz = np.append(disk_array_without_cavity, chosen_disk_array_xyz, axis=0)

                    assert len(molecules_xyz) == old_mol_counter - 1, f"{len(molecules_xyz)} != {old_mol_counter-1}"

            else: raise ValueError(choice)


# Faster than from scipy.spatial.distance_matrix
def dist_pbc(coords, box_size):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff -= box_size * np.round(diff / box_size)
    return (diff ** 2).sum(axis=2)

# -------------------- Rosenbluth Weight Calculations --------------------
def rosenbluth_weight_new(trial_mol_plus_mol_in_cav, n_atoms, choice):
    """
    Calculates forward Rosenbluth weights for a new configuration (adding a molecule).

    For each molecule to be added, the function:
      - Sums the interaction weights of all trial molecules.
      - Computes probabilities from these weights.
      - Chooses one trial molecule based on the computed probabilities.
      - Computes the Rosenbluth weight for the new configuration.

    Args:
        trial_mol_plus_mol_in_cav (TrialMoleculeSet): Set of trial molecules to choose from
        n_atoms (int): Number of atoms to select

    Returns:
        tuple: (chosen_disk_set, sum_e_minus_beta_u_ij_new_arr)
            - chosen_disk_set: TrialMoleculeSet containing the chosen molecules
            - sum_e_minus_beta_u_ij_new_arr: Array of Rosenbluth weights
    """
    sum_e_minus_beta_u_ij_new_arr = np.empty(n_atoms, dtype=float)

    chosen_disk_coords = []
    chosen_disk_energy_weight = []
    chosen_disk_cossum = []
    chosen_disk_sinsum = []
    chosen_disk_counts = []

    assert len(trial_mol_plus_mol_in_cav) > 0, "No non-overlapping trial molecules found."
    assert n_atoms != 0, "Number of atoms must be greater than zero."

    for i in range(n_atoms):

        gamma = gamma_dict[min(i+1,9)]

        #print(f"New {i} gamma {gamma}")

        if aux_pot_flag or lj_pot_flag:

            weights = trial_mol_plus_mol_in_cav.energy_weight
            sum_weight = np.sum(weights)
            probability = weights / sum_weight

        elif reweighting_flag:

            cos_sum = trial_mol_plus_mol_in_cav.cossum
            sin_sum = trial_mol_plus_mol_in_cav.sinsum
            counts =   trial_mol_plus_mol_in_cav.counts

            # Average the complex phasors, then take magnitude
            psi_6= np.sqrt( (cos_sum / counts)**2 + (sin_sum / counts)**2)

            #Start with psi6 of 0.3 which is the psi6 of random arrangement
            psi_6[psi_6 == 0] = 0.3
            psi_6[np.isnan(psi_6)] = 0.3   

            sum_weight = np.sum(psi_6)  
            probability = psi_6/sum_weight

            #print(cos_sum, sin_sum, counts)
            #print(psi_6, sum_weight, probability, np.sum(probability), '\n')

        else:
            #All trial molecules have equal probability it does not matter which index I choose

            sum_weight = len(trial_mol_plus_mol_in_cav)
            probability = np.ones(len(trial_mol_plus_mol_in_cav))/sum_weight            

        sum_e_minus_beta_u_ij_new_arr[i] = sum_weight

        np.testing.assert_almost_equal(np.sum(probability), 1.0, decimal=3)
        assert np.amin(probability) > 0
        assert np.amax(probability) <= 1

        cumulative = np.cumsum(probability)
        r = np.random.rand()
        chosen_index = np.searchsorted(cumulative, r)
        
        chosen_trial = trial_mol_plus_mol_in_cav[chosen_index]

        chosen_disk_coords.append(chosen_trial[0])
        chosen_disk_energy_weight.append(chosen_trial[1])
        chosen_disk_cossum.append(chosen_trial[2])
        chosen_disk_sinsum.append(chosen_trial[3])
        chosen_disk_counts.append(chosen_trial[4])

        if i == n_atoms - 1: break

        ######################### Interaction between chosen trial molecule and all the other trial molecules

        # Removing the chosen trial molecule from the pool of non-overlapping trial molecules
        trial_mol_plus_mol_in_cav = trial_mol_plus_mol_in_cav.delete(chosen_index)

        if len(trial_mol_plus_mol_in_cav) == 0:
            return None, None

        # coords of non-overlapping trial molecules
        coords_trial_mol = trial_mol_plus_mol_in_cav.coords

        # Calculate the difference vector of all trial molecules and chosen trial mol considering PBC
        delta_new = coords_trial_mol - chosen_trial[0]

        # Apply PBC
        delta_new -= side_of_box * np.round(delta_new / side_of_box)

        # Calculate the distances between the trial molecules and the chosen trial molecule
        sq_distances_chosen_mol_trial = np.sum(delta_new ** 2, axis=1)

        # Make sure the PBC are applied properly.
        if two_dim_flag: assert np.amax(sq_distances_chosen_mol_trial) <= (side_of_box/2)**2
        
        # Mask for overlapping molecules. 
        # The set of trial molecules overlap with chosen trial molecule
        mask = sq_distances_chosen_mol_trial >= r_squared

        # Applied mask to remove overlapping distances
        trial_mol_plus_mol_in_cav = trial_mol_plus_mol_in_cav.mask(mask)

        # Reject the step if there are no non-overlapping trial molecules leftF
        if len(trial_mol_plus_mol_in_cav) == 0:
            return None, None
        
        ##################

        if aux_pot_flag or lj_pot_flag:

            sq_distances_chosen_mol_trial = sq_distances_chosen_mol_trial[mask]
            
            if two_dim_flag or lj_pot_flag:
                g_r = calculate_interaction_energy(np.sqrt(sq_distances_chosen_mol_trial), gamma, len(molecules_xyz), choice)
            else:
                g_r = calculate_g3_pairwise(
                    trial_mol_plus_mol_in_cav,
                    chosen_trial[0],
                    sq_distances_chosen_mol_trial,
                    gamma
                )

            trial_mol_plus_mol_in_cav.energy_weight = trial_mol_plus_mol_in_cav.energy_weight * g_r

        elif reweighting_flag:

            if two_dim_flag:

                thetas = np.arctan2(delta_new[mask][..., 1], delta_new[mask][..., 0])  # (N, k)
    
                # 6-fold complex phasors: exp(i*6θ) = cos(6θ) + i*sin(6θ)
                cos_6theta = np.cos(6 * thetas)  # (N, k)
                sin_6theta = np.sin(6 * thetas)

                trial_mol_plus_mol_in_cav.cossum += cos_6theta
                trial_mol_plus_mol_in_cav.sinsum += sin_6theta
                trial_mol_plus_mol_in_cav.counts += 1

            else:

                g_r_new = calculate_q6_3d(delta_new, np.ones(len(delta_new)), gamma)
                trial_mol_plus_mol_in_cav.energy_weight = trial_mol_plus_mol_in_cav.energy_weight * g_r_new
   
        ################
    
    # Make sure one to one correspondence between interaction energy and chosen molecules in cavity
    assert len(sum_e_minus_beta_u_ij_new_arr) == n_atoms == len(chosen_disk_coords) == len(chosen_disk_energy_weight) == len(chosen_disk_cossum) != 0

    # Create TrialMoleculeSet with chosen molecules
    chosen_disk_set = TrialMoleculeSet(
        coords=np.array(chosen_disk_coords),
        energy_weight=np.array(chosen_disk_energy_weight),
        cossum=np.array(chosen_disk_cossum),
        sinsum=np.array(chosen_disk_sinsum),
        counts=np.array(chosen_disk_counts)

    )

    return chosen_disk_set, np.array(sum_e_minus_beta_u_ij_new_arr)


def rosenbluth_weight_current(trial_mol_setent, mol_in_cav_set, n_atoms, choice):

    assert  len(mol_in_cav_set) == n_atoms, f"Number of atoms in cavity {len(mol_in_cav_set)} does not match expected {n_atoms}"

    #sum_e_minus_beta_u_ij_current_arr = np.arange(n_atoms, 0, -1)
    sum_e_minus_beta_u_ij_current_arr = np.empty(n_atoms)

    #e_minus_beta_u_i_current_arr = mol_in_cav_set.energy_weight
    e_minus_beta_u_i_current_arr = np.empty(n_atoms)

    if trial_mol_setent is None:

        assert len(mol_in_cav_set) == n_atoms, "No trial molecules in the cavity to remove."

        if aux_pot_flag or lj_pot_flag:
            w = mol_in_cav_set.energy_weight
            for i in range(n_atoms):
                sum_e_minus_beta_u_ij_current_arr[i] = np.sum(w[i:])
                e_minus_beta_u_i_current_arr[i]      = w[i]

        elif reweighting_flag:
            for i in range(n_atoms):
                cs, ss, ct = mol_in_cav_set.cossum[i:], mol_in_cav_set.sinsum[i:], mol_in_cav_set.counts[i:]
                p = np.sqrt((cs / ct)**2 + (ss / ct)**2)
                p[p == 0] = 0.3
                p[np.isnan(p)] = 0.3
                sum_e_minus_beta_u_ij_current_arr[i] = np.sum(p)

                c1, s1, n1 = mol_in_cav_set.cossum[i], mol_in_cav_set.sinsum[i], mol_in_cav_set.counts[i]
                pi = np.sqrt((c1 / n1)**2 + (s1 / n1)**2)
                pi = 0.3 if (pi == 0 or np.isnan(pi)) else pi
                e_minus_beta_u_i_current_arr[i] = pi

        else:
            # hard sphere / hard disk — original behaviour
            sum_e_minus_beta_u_ij_current_arr = np.arange(n_atoms, 0, -1)
            e_minus_beta_u_i_current_arr      = np.ones(n_atoms)

        return sum_e_minus_beta_u_ij_current_arr, e_minus_beta_u_i_current_arr
    
    else:

        for i in range(len(mol_in_cav_set)):

            gamma = gamma_dict[min(i+1,9)]

            if aux_pot_flag or lj_pot_flag:

                sum_e_minus_beta_u_ij_current_arr[i] = sum(mol_in_cav_set.energy_weight[i:])
                e_minus_beta_u_i_current_arr[i] = mol_in_cav_set.energy_weight[i]

            elif reweighting_flag:

                cos_sum = mol_in_cav_set.cossum[i:]
                sin_sum = mol_in_cav_set.sinsum[i:]
                counts = mol_in_cav_set.counts[i:]

                psi_6= np.sqrt( (cos_sum / counts)**2 + (sin_sum / counts)**2)

                #Start with psi6 of 0.3 which is the psi6 of random arrangement
                psi_6[psi_6 == 0] = 0.3   
                psi_6[np.isnan(psi_6)] = 0.3

                sum_e_minus_beta_u_ij_current_arr[i] = np.sum(psi_6) 

                cos_sum = mol_in_cav_set.cossum[i]
                sin_sum = mol_in_cav_set.sinsum[i]
                counts = mol_in_cav_set.counts[i]

                psi_6= np.sqrt( (cos_sum / counts)**2 + (sin_sum / counts)**2)

                #Start with psi6 of 0.3 which is the psi6 of random arrangement
                if psi_6 == 0 or np.isnan(psi_6): psi_6 = 0.3

                e_minus_beta_u_i_current_arr[i] = psi_6

            else:
                
                sum_e_minus_beta_u_ij_current_arr[i] = len(mol_in_cav_set.energy_weight[i:])
                e_minus_beta_u_i_current_arr[i] = 1


            if len(trial_mol_setent) != 0:

                if aux_pot_flag or lj_pot_flag:

                    sum_e_minus_beta_u_ij_current_arr[i] += sum(trial_mol_setent.energy_weight)

                elif reweighting_flag:

                    cos_sum = trial_mol_setent.cossum
                    sin_sum = trial_mol_setent.sinsum
                    counts = trial_mol_setent.counts

                    psi_6= np.sqrt( (cos_sum / counts)**2 + (sin_sum / counts)**2)

                    #Start with psi6 of 0.3 which is the psi6 of random arrangement
                    psi_6[psi_6 == 0] = 0.3   
                    psi_6[np.isnan(psi_6)] = 0.3
                    
                    sum_e_minus_beta_u_ij_current_arr[i] += np.sum(psi_6) 
                
                else:

                    sum_e_minus_beta_u_ij_current_arr[i] += len(trial_mol_setent.energy_weight)

                trial_coords = trial_mol_setent.coords

                chosen_mol_in_cav = mol_in_cav_set[i][0]
                
                diff = trial_coords - chosen_mol_in_cav  # shape (n_current, 3)

                diff -= side_of_box * np.round(diff / side_of_box)

                dist_sq = np.sum(diff**2, axis=1)

                if two_dim_flag: assert np.amax(dist_sq) <= (side_of_box/2)**2, dist_sq

                mask_curr = dist_sq >= r_squared
                # Filter: keep points >= cutoff distance

                trial_mol_setent = trial_mol_setent.mask(mask_curr)

            if i == n_atoms - 1:
                return sum_e_minus_beta_u_ij_current_arr, e_minus_beta_u_i_current_arr
            
            if aux_pot_flag or lj_pot_flag:

                if len(trial_mol_setent) != 0:

                    distances_sq_filtered = dist_sq[mask_curr]
                    
                    if two_dim_flag or lj_pot_flag:
                        g_r = calculate_interaction_energy(np.sqrt(distances_sq_filtered), gamma, len(molecules_xyz), choice)
                    else:
                        g_r = calculate_g3_pairwise(
                            trial_mol_setent,
                            chosen_mol_in_cav,
                            distances_sq_filtered,
                            gamma
                        )

                    trial_mol_setent.energy_weight = trial_mol_setent.energy_weight * g_r

                #############################

                coords_mol_in_cav = mol_in_cav_set[i+1:].coords

                diff_cav = coords_mol_in_cav - chosen_mol_in_cav  # shape (n_current, 3)
                diff_cav -= side_of_box * np.round(diff_cav / side_of_box)
                diff_cav_sq = np.sum(diff_cav**2, axis=1)

                if two_dim_flag: assert np.amax(diff_cav_sq) <= (side_of_box/2)**2, diff_cav_sq
                
                if two_dim_flag or lj_pot_flag:
                    g_r_cav = calculate_interaction_energy(np.sqrt(diff_cav_sq), gamma, len(molecules_xyz), choice)
                else:
                    g_r_cav = calculate_g3_pairwise(
                        mol_in_cav_set[i+1:],
                        chosen_mol_in_cav,
                        diff_cav_sq,
                        gamma
                    )

                mol_in_cav_set.energy_weight[i+1:] *= g_r_cav
            
            elif reweighting_flag:

                    if len(trial_mol_setent) != 0:

                        thetas = np.arctan2(diff[mask_curr][..., 1], diff[mask_curr][..., 0])  # (N, k)
    
                        # 6-fold complex phasors: exp(i*6θ) = cos(6θ) + i*sin(6θ)
                        cos_6theta = np.cos(6 * thetas)  # (N, k)
                        sin_6theta = np.sin(6 * thetas)

                        trial_mol_setent.cossum += cos_6theta 
                        trial_mol_setent.sinsum += sin_6theta 
                        trial_mol_setent.counts += 1 

                    coords_mol_in_cav = mol_in_cav_set[i+1:].coords

                    diff_cav = coords_mol_in_cav - chosen_mol_in_cav  # shape (n_current, 3)

                    diff_cav -= side_of_box * np.round(diff_cav / side_of_box)

                    thetas = np.arctan2(diff_cav[..., 1], diff_cav[..., 0])  # (N, k)

                    # 6-fold complex phasors: exp(i*6θ) = cos(6θ) + i*sin(6θ)
                    cos_6theta = np.cos(6 * thetas)  # (N, k)
                    sin_6theta = np.sin(6 * thetas)

                    mol_in_cav_set[i+1:].cossum += cos_6theta 
                    mol_in_cav_set[i+1:].sinsum += sin_6theta 
                    mol_in_cav_set[i+1:].counts += 1 


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

# -------------------- Main Function --------------------
"""
Drop-in main() + restart for the existing on-disk layout:

    <cwd>/mu_<mu>_rhok_<rho>_r_<r>_k_<k>_2d_<2d>_.log          (logs in CWD)
    <save_file_folder>/mu_<mu>_r_<r>_rhok_<rho>_k_<k>_2d_<2d>_.extxyz   (trajs)

How identity is resolved on restart (filename-change tolerant):
  - First try the deterministic path the current parameters produce.
  - Verify it by a hybrid identity: read whatever the file's info carries,
    fall back to tokens parsed from the name for anything info lacks. Your
    legacy files carry only chem_pot in info, so the name supplies the rest.
  - If the deterministic path misses (renamed/moved), scan save_file_folder,
    build the same hybrid identity per file, match the full parameter set,
    and break ties by the furthest-along mc_cycle.
  - On a successful resume the checkpoint is rewritten with the FULL signature
    in info, so every run becomes self-describing and rename-proof after its
    first resume. No separate backfill needed.

This module expects, in its namespace, the run globals you already use
(restart_mode, PARALLEL, chem_pot, cavity_radius, trial_density,
no_of_trial_molecules, two_dim_flag, aux_pot_flag, gamma_max, delta_eta,
reweighting_flag, side_of_box, volume_of_molecule, Total_volume, box_dims,
sampling_window, burn_in, n_workers, chunk_size, save_file_folder,
LOG_FLUSH_EVERY, TRAJ_FLUSH_EVERY) and the three workers run_mc_in_a_cell,
mc_mapper, grid_up_the_system.
"""

# ---------------------------------------------------------------------------
# Names: one place that reproduces the current on-disk convention exactly.
# (The non-aux log keeps rhok before r, which is harmless: matching below is
#  token-order independent. Unify the two later if you like; nothing breaks.)
# ---------------------------------------------------------------------------
def build_paths():
    if aux_pot_flag:
        stem = (f"mu_{chem_pot}_r_{cavity_radius}_rhok_{trial_density}"
                f"_k_{no_of_trial_molecules}_2d_{two_dim_flag}"
                f"_aux_pot_{aux_pot_flag}_gammamax_{gamma_max}_deltaeta_{delta_eta}_")
        traj = os.path.join(save_file_folder, stem + ".extxyz")
        log = stem + ".log"
    else:
        traj = os.path.join(
            save_file_folder,
            f"mu_{chem_pot}_r_{cavity_radius}_rhok_{trial_density}"
            f"_k_{no_of_trial_molecules}_2d_{two_dim_flag}_.extxyz")
        log = (f"mu_{chem_pot}_rhok_{trial_density}_r_{cavity_radius}"
               f"_k_{no_of_trial_molecules}_2d_{two_dim_flag}_.log")
    return traj, log


# Each token is matched independently, so name order does not matter.
_NAME_TOKENS = [
    ("chem_pot",              r'mu_(-?\d+(?:\.\d+)?)_',     float),
    ("cavity_radius",         r'_r_(-?\d+(?:\.\d+)?)_',     float),
    ("trial_density",         r'rhok_(-?\d+(?:\.\d+)?)_',   float),
    ("no_of_trial_molecules", r'_k_(\d+)_',                 int),
    ("two_dim",               r'_2d_(True|False)_',         lambda s: s == "True"),
    ("aux_pot",               r'_aux_pot_(True|False)_',    lambda s: s == "True"),
    ("gamma_max",             r'gammamax_(-?\d+(?:\.\d+)?)_', float),
    ("delta_eta",             r'deltaeta_(-?\d+(?:\.\d+)?)_', float),
]


def _parse_run_name(path):
    b = os.path.basename(path)
    out = {}
    for key, pat, cast in _NAME_TOKENS:
        m = re.search(pat, b)
        if m:
            out[key] = cast(m.group(1))
    out.setdefault("aux_pot", False)  # non-aux names omit the token
    return out


def _run_signature():
    sig = dict(
        chem_pot=float(chem_pot),
        cavity_radius=float(cavity_radius),
        trial_density=float(trial_density),
        no_of_trial_molecules=int(no_of_trial_molecules),
        two_dim=bool(two_dim_flag),
        aux_pot=bool(aux_pot_flag),

    )
    if aux_pot_flag:
        sig["gamma_max"] = float(gamma_max)
        sig["delta_eta"] = float(delta_eta)
    return sig


def _match(path, info, sig, tol=1e-9):
    """Hybrid identity: info wins, name fills whatever info lacks."""
    ident = _parse_run_name(path)
    ident.update(info)
    if "no_of_trial_molecules" not in ident and "no_trials" in ident:
        ident["no_of_trial_molecules"] = ident["no_trials"]
    for k, want in sig.items():
        if k not in ident:
            return False
        got = ident[k]
        if isinstance(want, float):
            try:
                if abs(float(got) - want) > tol:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(got) != str(want):
            return False
    return True


def _locate_restart(traj_file_path, sig):
    if os.path.exists(traj_file_path):
        try:
            if _match(traj_file_path, read(traj_file_path, index=-1).info, sig):
                return traj_file_path
        except Exception:
            pass
    matches = []
    for f in sorted(glob.glob(os.path.join(save_file_folder, "*.extxyz"))):
        try:
            info = read(f, index=-1).info
        except Exception:
            continue
        if _match(f, info, sig):
            matches.append((int(info.get("mc_cycle", -1)), f))
    if not matches:
        return None
    matches.sort()
    if len(matches) > 1:
        print(f"[restart] {len(matches)} trajectories match these parameters; "
              f"resuming the furthest along: {matches[-1][1]}", flush=True)
    return matches[-1][1]


def _write_checkpoint(traj_file_path, molecules_xyz, total_step_counter,
                      packing_fraction, sig):
    positions_3d = (np.column_stack([molecules_xyz, np.zeros(len(molecules_xyz))])
                    if two_dim_flag else molecules_xyz)
    atoms = Atoms('H' * len(molecules_xyz), positions=positions_3d,
                  cell=[side_of_box] * 3, pbc=True)
    atoms.info['mc_cycle'] = int(total_step_counter)
    atoms.info['packing_fraction'] = float(packing_fraction)
    atoms.info['reweighting'] = str(reweighting_flag)
    atoms.info['no_trials'] = int(no_of_trial_molecules)
    for k, v in sig.items():        # full signature -> rename-proof from now on
        atoms.info[k] = v
    write(traj_file_path, atoms, append=False)


# ---------------------------------------------------------------------------
# main: one loop, runs until total target, both modes log and checkpoint.
# ---------------------------------------------------------------------------
def main(mc_cycles, molecules_xyz, log_file_path=None, traj_file_path=None):
    sig = _run_signature()
    traj_file_path, log_file_path = build_paths()   # single source of truth

    if not restart_mode:
        total_step_counter = 0
        molecules_xyz = np.asarray(molecules_xyz, float)
        if two_dim_flag:
            molecules_xyz = molecules_xyz[:, :2]
        need_header = not os.path.exists(log_file_path)
    else:
        located = _locate_restart(traj_file_path, sig)
        if located is None:
            present = []
            for f in glob.glob(os.path.join(save_file_folder, "*.extxyz")):
                try:
                    present.append(read(f, index=-1).info.get('chem_pot'))
                except Exception:
                    pass
            print(f"[restart] no trajectory matched params {sig}; chem_pot present: "
                  f"{sorted({p for p in present if p is not None})}", flush=True)
            raise SystemExit(1)
        traj_file_path = located
        atoms = read(located, index=-1)
        molecules_xyz = atoms.positions
        if two_dim_flag:
            molecules_xyz = molecules_xyz[:, :2]
        total_step_counter = int(atoms.info['mc_cycle'])
        need_header = not os.path.exists(log_file_path)
        print(f"[restart] resume '{located}' at cycle {total_step_counter}, "
              f"N={len(molecules_xyz)}", flush=True)

    add_accept_counter = add_attempt_counter = add_step_counter = 0
    move_step_counter = move_accept_counter = 0
    remove_accept_counter = remove_step_counter = 0
    repack_accept_counter = repack_step_counter = 0
    total_time = 0

    if need_header:
        with open(log_file_path, 'w') as lf:
            lf.write("#MC_cycle(M)  pckng_frctn   N_disk   add_acc ad_atmpt ad_stp "
                     "rm_acc rm_stp mv_acc mv_stp   time_total (ms)\n")

    log_buffer = []
    log_f = open(log_file_path, 'a')
    redraw_idx = 0

    try:
        while total_step_counter < mc_cycles:

            if PARALLEL:
                random_set, rest_of_the_molecules_xyz = grid_up_the_system(
                    system_array_xyz=molecules_xyz, side=side_of_box,
                    cell_length=30, buffer=2, x_spacing=4, y_spacing=3,
                    x_pos=0, y_pos=0)
                n_cells = len(random_set)
                input_array = [[c[0], c[1], sampling_window, c[2], box_dims]
                               for c in random_set]

                add_accept_counter = add_attempt_counter = add_step_counter = 0
                remove_accept_counter = remove_step_counter = 0
                move_accept_counter = move_step_counter = 0
                repack_accept_counter = repack_step_counter = 0
                total_time_taken = 0

                mc_output = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
                    for r in ex.map(mc_mapper, input_array, chunksize=chunk_size):
                        mc_output.append(r)

                for r in mc_output:
                    (_xyz, rpc_s, rpc_a, ad_s, ad_at, ad_a,
                     rm_s, rm_a, mv_s, mv_a, t) = r
                    add_accept_counter += ad_a
                    add_attempt_counter += ad_at
                    add_step_counter += ad_s
                    move_accept_counter += mv_a
                    move_step_counter += mv_s
                    remove_accept_counter += rm_a
                    remove_step_counter += rm_s
                    repack_accept_counter += rpc_a
                    repack_step_counter += rpc_s
                    total_time_taken += t

                total_time = (total_time_taken * 1000) / (sampling_window * n_cells)
                total_step_counter += sampling_window * n_cells
                molecules_xyz = np.vstack(
                    [r[0] for r in mc_output] + [rest_of_the_molecules_xyz])

            else:
                border_molecules = []
                cell_dims = [[0, side_of_box, 0, side_of_box], [side_of_box / 2] * 4]
                (molecules_xyz,
                 repack_step_counter, repack_accept_counter,
                 add_step_counter, add_attempt_counter, add_accept_counter,
                 remove_step_counter, remove_accept_counter,
                 move_step_counter, move_accept_counter,
                 total_time) = run_mc_in_a_cell(
                    molecules_xyz, border_molecules, sampling_window,
                    cell_dims, box_dims)
                total_step_counter += sampling_window

            # ---- common: packing fraction, log line, checkpoint ----
            packing_fraction = (len(molecules_xyz) * volume_of_molecule) / Total_volume
            logging.debug(f"mu: {chem_pot} cycle: {total_step_counter/1e6}M "
                          f"eta: {np.round(packing_fraction,4)} N: {len(molecules_xyz)}")

            if total_step_counter >= burn_in:
                log_buffer.append(
                    f"{total_step_counter/1e6:.6f}  {packing_fraction:.8f}  "
                    f"{len(molecules_xyz)}  {add_accept_counter}  {add_attempt_counter}  "
                    f"{add_step_counter}  {remove_accept_counter}  {remove_step_counter}  "
                    f"{move_accept_counter}  {move_step_counter}  {total_time*1000:.6f}\n")
                if len(log_buffer) >= LOG_FLUSH_EVERY:
                    log_f.writelines(log_buffer)
                    log_f.flush()
                    log_buffer.clear()

            if redraw_idx % TRAJ_FLUSH_EVERY == 0 or total_step_counter >= mc_cycles:
                _write_checkpoint(traj_file_path, molecules_xyz,
                                  total_step_counter, packing_fraction, sig)
            redraw_idx += 1

    finally:
        if log_buffer:
            log_f.writelines(log_buffer)
        log_f.close()

    return molecules_xyz, total_step_counter
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main(mc_cycles, molecules_xyz, log_file_path, traj_file_path)