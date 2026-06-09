"""
GCMC Simulation Analysis for High Density Regime
Analyzes Monte Carlo simulation data to compute density, acceptance rates, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import re, os
from scipy.optimize import brentq
import teqp


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Analysis configuration parameters"""
    folders: List[str] = None
    colors: List[str] = None
    box_size: float = 20.0
    burn_in_cycles: float = 0.75e6
    output_interval: float = 1000.0  # cycles per data point
    min_blocks: int = 20
    use_complex_scaling: bool = False
    dpi: int = 600
    
    # Chemical potential filtering
    min_chemical_potential: float = -7  # covers the negative μ range in LJ data
    max_chemical_potential: float = 0

    skip_chemical_potentials: set = None  # Additional specific values to skip
    
    # Reference data plotting
    plot_reference_data: bool = True
    # LJ reduced temperature T* for EOS reference curves
    # Set to None to skip EOS overlay; set to a list for multiple isotherms
    lj_temperature: float = 1.0   # T* = kT/ε

    # Which EOS branch to overlay: 'liquid', 'vapor', or 'both'
    eos_branch: str = 'both'
    
    # Inset control
    show_insets: bool = False  # Set to False to disable insets on density plot
    plot_alpha: float = 0.75
    
    # Line pattern control
    use_different_line_patterns: bool = True  # Set to False for all solid lines
    line_styles: List[str] = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        if self.folders is None:
                self.folders = [
                    #"d_0_T_0.85_k_1_r_10",  "d_0_T_0.85_rhok_400_r_1.5_empty",  "d_0_T_0.85_rhok_400_r_1.5", # "d_3_T_0.85_rhok_400_r_1.5", "d_0_T_0.85_rhok_200_r_2"
                    #"d_0_T_1.5_k_1_r_10",  "d_0_T_1.5_rhok_400_r_1.5_empty",  "d_0_T_1.5_rhok_400_r_1.5",#^ "d_0_T_1.5_rhok_200_r_2", "d_3_T_1.5_rhok_400_r_1.5",
                    "d_0_T_1.0_rhok_400_r_1.5_empty", "d_0_T_1.0_rhok_400_r_1.5", "d_3_T_1.0_rhok_400_r_1.5"
                    ]

        if self.colors is None:
            # Publication-quality color scheme (colorblind-friendly)
            # Option 1: Vibrant scheme
            self.colors = [
                '#0173B2',  # blue
                '#DE8F05',  # orange
                '#029E73',  # green
                '#CC78BC',  # pink/purple
                '#CA9161',  # tan/brown
                '#949494',  # gray
                '#D55E00',  # vermilion
                '#E69F00',  # amber
                '#56B4E9',  # sky blue
                '#009E73',  # teal
                '#F0E442',  # yellow
                '#0072B2',  # deep blue
                '#8B4513',  # sienna
                '#999933',  # olive
                '#882255',  # dark magenta
            ]
            
        if self.skip_chemical_potentials is None:
            self.skip_chemical_potentials = set()  # nothing skipped by default for LJ
        if self.line_styles is None:
            # Matplotlib line style options
            self.line_styles = [
                '-',      # solid
                '--',     # dashed
                '-.',     # dash-dot
                ':',      # dotted
                (0, (3, 1, 1, 1)),  # densely dashdotted
                (0, (5, 5)),        # dashed with equal spacing
                (0, (3, 5, 1, 5)),  # dashdotted
                (0, (1, 1)),        # densely dotted
                (0, (5, 1)),        # densely dashed
                (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
            ]
        self.burn_in_points = int(self.burn_in_cycles / self.output_interval)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

# ── LJ EOS models (teqp) — same setup as lj_mu_multi.py ─────────────────────
_z = np.array([1.0])
_kn   = teqp.make_model({'kind': 'LJ126_KolafaNezbeda1994', 'model': {}})
_jzg  = teqp.make_model({'kind': 'LJ126_Johnson1993',       'model': {}})
_thol = teqp.build_LJ126_TholJPCRD2016()

def _mu_lj(model, rho, T):
    """Residual chemical potential from teqp: μ* = T*(ln ρ + Ar00 + Ar01)"""
    return T * (np.log(rho) + model.get_Ar00(T, rho, _z) + model.get_Ar01(T, rho, _z))

def _vapor_bracket_upper(mu_fn, T: float) -> float:
    """
    Find the spinodal peak rho* for the vapor branch by scanning mu(rho) on a
    coarse grid and returning 90% of that rho.  For supercritical T (no local
    maximum) returns 0.55 which covers the full single branch.
    """
    rhos = np.linspace(1e-4, 0.60, 1000)
    mus  = np.array([mu_fn(r, T) for r in rhos])
    peak_idx = int(np.argmax(mus))
    if peak_idx >= len(rhos) - 5:   # monotone -> supercritical
        return 0.55
    return float(rhos[peak_idx]) * 0.90


def lj_eos_rho_vs_mu(T: float, mu_grid: np.ndarray,
                     branch: str = 'liquid',
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Invert three full-LJ EOS to get rho*(mu*) at temperature T*.

    branch: 'liquid' -- rho in [0.40, 1.20]
            'vapor'  -- rho in [1e-4, spinodal_peak * 0.90]  (computed per T)
            'both'   -- liquid first, vapor fallback

    Returns (rho_KN, rho_JZG, rho_Thol) aligned with mu_grid; NaN where no root.
    """
    fns = [
        lambda r, T: _mu_lj(_kn,   r, T),
        lambda r, T: _mu_lj(_jzg,  r, T),
        lambda r, T: _mu_lj(_thol, r, T),
    ]

    # Vapor ceiling computed from KN; all three EOS agree closely at low density
    vapor_hi = _vapor_bracket_upper(fns[0], T)

    _BRACKETS = {
        'liquid': (0.40, 1.20),
        'vapor':  (1e-4, vapor_hi),
    }

    def _invert_bracket(mu_fn, mu_target, rho_lo, rho_hi):
        try:
            fa = mu_fn(rho_lo, T) - mu_target
            fb = mu_fn(rho_hi, T) - mu_target
            if fa * fb > 0:
                return np.nan
            return brentq(lambda r: mu_fn(r, T) - mu_target,
                          rho_lo, rho_hi, xtol=1e-10, rtol=1e-10)
        except ValueError:
            return np.nan

    def _invert(mu_fn, mu_target):
        if branch == 'both':
            rho = _invert_bracket(mu_fn, mu_target, *_BRACKETS['liquid'])
            if np.isnan(rho):
                rho = _invert_bracket(mu_fn, mu_target, *_BRACKETS['vapor'])
            return rho
        lo, hi = _BRACKETS[branch]
        return _invert_bracket(mu_fn, mu_target, lo, hi)

    results = [np.array([_invert(fn, mu) for mu in mu_grid]) for fn in fns]
    return tuple(results)


# EOS display metadata: (label, color, linestyle, linewidth)
_EOS_STYLES = [
    ('KN 1994',    '#555555', '-',  1.6),
    ('JZG 1993',   '#555555', '--', 1.3),
    ('Thol 2016',  '#555555', ':',  1.8),
]


@dataclass
class SimulationData:
    """Container for simulation results at a single chemical potential"""
    chemical_potential: float
    density_mean: float
    density_stderr: float
    add_ratio: float
    remove_ratio: float
    add_rate: float
    remove_rate: float
    max_cycles: float
    time_per_step: float


@dataclass
class DatasetResults:
    """Complete results for one dataset (folder)"""
    label: str
    color: str
    linestyle: str
    trial_molecules: int
    data: List[SimulationData]
    
    def get_arrays(self):
        """Extract arrays for plotting"""
        if not self.data:
            return {k: np.array([]) for k in ['mu', 'density', 'stderr', 
                   'add_ratio', 'remove_ratio', 'add_rate', 'remove_rate',
                   'max_cycles', 'time']}
        
        return {
            'mu': np.array([d.chemical_potential for d in self.data]),
            'density': np.array([d.density_mean for d in self.data]),
            'stderr': np.array([d.density_stderr for d in self.data]),
            'add_ratio': np.array([d.add_ratio for d in self.data]),
            'remove_ratio': np.array([d.remove_ratio for d in self.data]),
            'add_rate': np.array([d.add_rate for d in self.data]),
            'remove_rate': np.array([d.remove_rate for d in self.data]),
            'max_cycles': np.array([d.max_cycles for d in self.data]),
            'time': np.array([d.time_per_step for d in self.data])
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def block_average_stderr(data: np.ndarray, min_blocks: int = 20) -> Tuple[float, float]:
    """
    Calculate mean and standard error using progressive block averaging.
    
    This accounts for autocorrelation in time series data by grouping
    observations into increasingly large blocks.
    
    Args:
        data: Time series data
        min_blocks: Minimum number of blocks to maintain
        
    Returns:
        mean: Sample mean
        stderr: Standard error of the mean
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    
    stderr_values = []
    block_size = 1
    
    while n // block_size >= min_blocks:
        n_blocks = n // block_size
        # Reshape into blocks and compute block means
        blocks = x[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = blocks.mean(axis=1)
        
        # Standard error from block means
        stderr = np.sqrt(block_means.var(ddof=1) / n_blocks)
        stderr_values.append(stderr)
        block_size *= 2
    
    mean = x.mean()
    # Use the last (most conservative) stderr estimate
    stderr = stderr_values[-1] if stderr_values else np.nan
    
    return mean, stderr


def extract_chemical_potential(filename: str) -> float:
    """Extract chemical potential from filename.

    New format: 'mu_-0.25_rhok_400.0_r_1.5_k_4583_2d_False_.log'
    Old format: 'dyn_-12.5.out'
    """
    name = os.path.basename(filename)

    # New format: mu_<signed_value>_...
    match = re.search(r"(?:^|_)mu_([-+]?\d+(?:\.\d+)?)", name)
    if match:
        return float(match.group(1))

    # Old format: last number in the stem (handles negatives too)
    stem = re.sub(r"\.[^.]+$", "", name)
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", stem)
    if not matches:
        raise ValueError(f"Could not extract chemical potential from {filename}")
    return float(matches[-1])

def extract_k_from_filename(filename: str) -> int:
    """Extract trial molecule count k from new-format filename like mu_7.0_r_1.2_k_58_..."""
    match = re.search(r"_k_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def load_simulation_file(filepath: Path, burn_in: int) -> Tuple[np.ndarray, ...]:
    """
    Load and parse simulation output file.

    New .log format (space-delimited, '#' header line):
      Col 0: MC_cycle(M)   Col 1: pckng_frctn   Col 2: N_disk
      Col 3: add_acc       Col 4: ad_stp
      Col 5: rm_acc        Col 6: rm_stp
      Col 7: mv_acc        Col 8: mv_stp         Col 9: time_step

    Old .out format (underscore-delimited):
      Col 0: cycle  Col 2: density  Col 5: add_acc  Col 8: add_stp
      Col 11: rm_acc  Col 14: rm_stp  Col 28/30: time

    Returns:
        density, add_accept, add_step, remove_accept, remove_step, max_cycle, time_per_step
    """
    suffix = filepath.suffix.lower()

    if suffix == '.log':
        # Space-delimited; skip comment lines starting with '#'
        # cols: 0=MC_cycle(M)  1=pckng_frctn  2=N_disk
        #       3=add_acc  4=ad_atmpt  5=ad_stp  6=rm_acc  7=rm_stp
        #       8=mv_acc   9=mv_stp   10=time_total(ms)
        data = np.loadtxt(filepath, comments='#')
        density       = data[:, 1]
        add_accept    = data[burn_in:, 3]
        add_step      = data[burn_in:, 5]
        remove_accept = data[burn_in:, 6]
        remove_step   = data[burn_in:, 7]
        max_cycle     = data[-1, 0] * 1e6

        # time_total is cumulative ms; compute total elapsed wall time over post-burn-in window
        time_ms       = data[burn_in:, 10]
        time_per_step = (time_ms[-1] - time_ms[0]) / 1000.0  # total seconds (used as wall time)

    else:
        # Legacy underscore-delimited .out format
        for time_col in [30, 28]:
            try:
                data = np.loadtxt(filepath, delimiter='_',
                                  usecols=(2, 5, 8, 11, 14, 0, time_col))
                break
            except (ValueError, IndexError):
                if time_col == 28:
                    raise ValueError(f"Could not load {filepath}")
        density       = data[:, 0]
        add_accept    = data[burn_in:, 1]
        add_step      = data[burn_in:, 2]
        remove_accept = data[burn_in:, 3]
        remove_step   = data[burn_in:, 4]
        max_cycle     = data[-1, 5] * 1e6
        time_per_step = data[burn_in:, 6].mean()

    return density, add_accept, add_step, remove_accept, remove_step, max_cycle, time_per_step


def analyze_folder(folder_path: Path, config: Config) -> DatasetResults:
    """
    Analyze all simulation files in a folder.
    
    Args:
        folder_path: Path to folder containing .out files
        config: Analysis configuration
        
    Returns:
        DatasetResults object with complete analysis
    """
    folder_name = folder_path.name
    
    # Find and sort output files — prefer .log, fall back to .out
    files = sorted(folder_path.glob('*.log'),
                   key=lambda f: extract_chemical_potential(f.name))
    if not files:
        files = sorted(folder_path.glob('*.out'),
                       key=lambda f: extract_chemical_potential(f.name))

    # Pull k from the first log filename so it can override folder-name heuristics
    k_from_file = None
    if files and files[0].suffix.lower() == '.log':
        k_from_file = extract_k_from_filename(files[0].name)

    # Determine label and trial molecule count

    if folder_name in [
            'n_0_k_100_r_1.5', 'n_1_k_100_r_1.5', 'n_2_k_100_r_1.5', 'n_3_k_100_r_1.5', 'n_4_k_100_r_1.5', 'n_5_k_100_r_1.5'
            ]:
        
        order = int(folder_name.split('_')[1])
        n_trial = int(folder_name.split('_')[3])


        label = f"CB order {order}"
    
    elif folder_name == 'n_0_k_1000_r_nocav':
        order = 0
        n_trial = 1000
        label = "CB, no trial box, 1000 trial disks"
    
    elif folder_name == 'n_0_k_1000_r_1.5':
        order = 0
        n_trial = 1000
        label = "CB, trial box side 3, 1000 trial disks"

    else:
        label = folder_name.replace('_', ' ').title()
        try:
            n_trial = int(folder_name.split('_')[3])
            order = int(folder_name.split('_')[1])
        except (IndexError, ValueError):
            n_trial = 2500
            order = 0

    # k from filename takes priority over folder-name heuristics
    if k_from_file is not None:
        n_trial = k_from_file

    if folder_name == 'n_0_k_100_r_1.5':
        order = 0
        n_trial = 100
        label = f"CB order 0, trial box side 3, 100 trial disks"
    
    results = []
    for filepath in files:
        try:
            mu = extract_chemical_potential(filepath.name)
            
            # Filter: only high chemical potentials
            if mu < config.min_chemical_potential:
                continue
            if mu > config.max_chemical_potential:
                continue
            if mu in config.skip_chemical_potentials:
                continue
            
            print(f"Processing: {filepath.name} (μ = {mu})")
            
            # Load data
            density, add_acc, add_step, rem_acc, rem_step, max_cyc, time = \
                load_simulation_file(filepath, config.burn_in_points)
            
            # Calculate density statistics
            density_mean, density_stderr = block_average_stderr(
                density[config.burn_in_points:], config.min_blocks
            )
            
            # Calculate acceptance metrics
            total_add    = add_step.sum()
            total_remove = rem_step.sum()
            
            if total_add == 0:
                raise ValueError(f"No add steps after burn-in for {filepath.name}")
            
            add_ratio    = add_acc.sum() / total_add
            remove_ratio = rem_acc.sum() / total_remove

            # Rate = accepted / total_wall_time; time holds total elapsed seconds
            add_rate     = add_acc.sum()  / time
            remove_rate  = rem_acc.sum()  / time
           
            add_rate *= (2*order + 1)
            remove_rate *= (2*order + 1)

            results.append(SimulationData(
                chemical_potential=mu,
                density_mean=density_mean,
                density_stderr=density_stderr,
                add_ratio=add_ratio,
                remove_ratio=remove_ratio,
                add_rate=add_rate,
                remove_rate=remove_rate,
                max_cycles=max_cyc,
                time_per_step=time
            ))
            
            print(f"  ρ = {density_mean:.6f} ± {density_stderr:.6f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    color_idx = config.folders.index(folder_name) % len(config.colors)
    linestyle_idx = config.folders.index(folder_name) % len(config.line_styles)
    
    # Choose linestyle based on toggle
    if config.use_different_line_patterns:
        linestyle = config.line_styles[linestyle_idx]
    else:
        linestyle = '-'  # solid line for all
    
    return DatasetResults(label, config.colors[color_idx], linestyle, n_trial, results)


def compute_theoretical_ratios(dataset: DatasetResults, config: Config):
    """
    Compute theoretically-scaled acceptance ratios.
    
    For complex bias: ratio ~ sqrt(raw_ratio / [k(k-1)])
    For simple bias: ratio ~ raw_ratio / k
    """
    arrays = dataset.get_arrays()
    k = dataset.trial_molecules
    
    if config.use_complex_scaling:
        k_factor = k * (k - 1)
        scale_factor = 1.5
        add_theory = scale_factor * np.sqrt(arrays['add_ratio'] / k_factor)
        remove_theory = scale_factor * np.sqrt(arrays['remove_ratio'] / k_factor)
        add_rate_scaled = scale_factor * np.sqrt(arrays['add_rate'])
        remove_rate_scaled = scale_factor * np.sqrt(arrays['remove_rate'])
    else:
        k_factor = k
        add_theory = arrays['add_ratio'] / k_factor
        remove_theory = arrays['remove_ratio'] / k_factor
        add_rate_scaled = arrays['add_rate']
        remove_rate_scaled = arrays['remove_rate']
    
    return add_theory, remove_theory, add_rate_scaled, remove_rate_scaled


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

class PlotManager:
    """Manages all plotting operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.datasets = []  # Track datasets for legend creation
        plt.rcParams['axes.labelsize'] = 13
        self._create_figures()
        
    def _create_figures(self):
        """Initialize all figure objects"""
        self.figs = {}
        self.axes = {}
        
        # Density plot with insets
        self.figs['density'], self.axes['density'] = plt.subplots(constrained_layout=True)
        self._setup_density_insets()
        
        # Acceptance plots
        for name in ['accept_rate', 'accept_theory', 'accept_prob']:
            self.figs[name], self.axes[name] = plt.subplots(constrained_layout=True)
            self.axes[name].set_yscale('log')
        
        # Performance plots
        for name in ['max_cycles', 'time']:
            self.figs[name], self.axes[name] = plt.subplots()
    
    def _setup_density_insets(self):
        """Add zoomed insets to density plot"""
        if not self.config.show_insets:
            return
        
        ax = self.axes['density']
        
        # First inset (high density transition region)
        x1, x2, y1, y2 = 11.9, 12.5, 0.66, 0.67
        rect1 = Rectangle((x1, y1), x2-x1, y2-y1, 
                         linewidth=1, edgecolor='k', facecolor='none', zorder=11)
        ax.add_patch(rect1)
        
        inset1 = self.figs['density'].add_axes([0.6, 0.35, 0.225, 0.225])
        inset1.set_xlim([x1, x2])
        inset1.set_ylim([y1, y2])
        inset1.grid(True, linestyle='--',  alpha=self.config.plot_alpha)
        self.axes['inset1'] = inset1
        
        # Second inset (near crystallization)
        x1, x2, y1, y2 = 12.0, 12.6, 0.662, 0.666
        rect2 = Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=1, edgecolor='k', facecolor='none', zorder=12)
        ax.add_patch(rect2)
        
        inset2 = self.figs['density'].add_axes([0.19, 0.62, 0.225, 0.225])
        inset2.set_xlim([x1, x2])
        inset2.set_ylim([y1, y2])
        inset2.grid(True, linestyle='--',  alpha=self.config.plot_alpha)
        self.axes['inset2'] = inset2
    
    def plot_dataset(self, dataset: DatasetResults):
        """Plot all metrics for one dataset"""
        arrays = dataset.get_arrays()
        mu = arrays['mu']
        color = dataset.color
        label = dataset.label
        linestyle = dataset.linestyle
        
        # Track dataset for legend creation
        self.datasets.append(dataset)
        
        # Plot density (pass label=None, we'll create custom legend)
        self._plot_with_error(self.axes['density'], mu, arrays['density'], 
                             arrays['stderr'], color, None, linestyle)
        
        # Plot to insets if they exist
        if self.config.show_insets:
            for inset_name in ['inset1', 'inset2']:
                if inset_name in self.axes:
                    self._plot_with_error(self.axes[inset_name], mu, arrays['density'],
                                         arrays['stderr'], color, None, linestyle)
        
        # Compute theoretical ratios
        add_theory, remove_theory, add_rate_scaled, remove_rate_scaled = \
            compute_theoretical_ratios(dataset, self.config)
        
        # Plot acceptance rates (pass label=None)
        self._plot_add_remove(self.axes['accept_rate'], mu, 
                             add_rate_scaled, remove_rate_scaled, color, None, linestyle)
        
        # Plot theoretical acceptance ratios
        self._plot_add_remove(self.axes['accept_theory'], mu,
                             add_theory, remove_theory, color, None, linestyle)
        
        # Plot raw acceptance probabilities
        self._plot_add_remove(self.axes['accept_prob'], mu,
                             arrays['add_ratio'], arrays['remove_ratio'], color, None, linestyle)
        
        # Plot performance metrics
        self._plot_simple(self.axes['max_cycles'], mu, arrays['max_cycles'], color, None, linestyle)
        self._plot_simple(self.axes['time'], mu, arrays['time'], color, None, linestyle)
    
    def plot_reference_data(self):
        """Overlay LJ EOS reference curves (KN 1994, JZG 1993, Thol 2016) on the density plot."""
        if not self.config.plot_reference_data:
            return
        if self.config.lj_temperature is None:
            return

        T = self.config.lj_temperature
        branch = self.config.eos_branch
        ax = self.axes['density']

        # Build a dense μ grid spanning the plotted range with margin
        mu_lo = self.config.min_chemical_potential - 0.5
        mu_hi = self.config.max_chemical_potential + 0.5
        mu_grid = np.linspace(mu_lo, mu_hi, 600)

        rhos = lj_eos_rho_vs_mu(T, mu_grid, branch=branch)

        # Simulation column 1 is packing fraction η = (π/6)ρ* — convert EOS ρ* → η
        eta_factor = np.pi / 6.0

        branch_tag = {'liquid': 'liq', 'vapor': 'vap', 'both': 'liq+vap'}[branch]
        for (rho_arr, (lbl, clr, ls, lw)) in zip(rhos, _EOS_STYLES):
            valid = ~np.isnan(rho_arr)
            eta_arr = rho_arr * eta_factor
            ax.plot(mu_grid[valid], eta_arr[valid],
                    color=clr, ls=ls, lw=lw, alpha=0.7, zorder=3,
                    label=f'{lbl}  T*={T:.2f} ({branch_tag})')

    def _plot_with_error(self, ax, x, y, yerr, color, label, linestyle='-'):
        """Plot line with error bars and scatter points"""
        ax.plot(x, y, linestyle=linestyle, linewidth=1, color=color, alpha=self.config.plot_alpha, zorder=1)
        ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=5, 
                   color=color, alpha=self.config.plot_alpha, zorder=7)
        ax.scatter(x, y, s=250, color='white', zorder=2)
        ax.scatter(x, y, s=25, facecolors='none', edgecolors=color,
                  linewidths=1, label=label, zorder=8)
    
    def _plot_add_remove(self, ax, x, add_data, remove_data, color, label, linestyle='-'):
        """Plot add and remove ratios together"""
        # Add ratio (circles)
        ax.plot(x, add_data, linestyle=linestyle, linewidth=1, color=color, alpha=self.config.plot_alpha, zorder=1)
        ax.scatter(x, add_data, s=100, marker='o', color='white', zorder=2)
        ax.scatter(x, add_data, s=25, marker='o', facecolors='none',
                  edgecolors=color, linewidths=1, label=label, zorder=12)
        
        # Remove ratio (squares)
        ax.plot(x, remove_data, linestyle=linestyle, linewidth=1, color=color, alpha=self.config.plot_alpha, zorder=3)
        ax.scatter(x, remove_data, s=100, marker='o', color='white', zorder=4)
        ax.scatter(x, remove_data, s=25, marker='s', facecolors='none',
                  edgecolors=color, linewidths=1, zorder=11)
    
    def _plot_simple(self, ax, x, y, color, label, linestyle='-'):
        """Simple scatter plot with line"""
        ax.plot(x, y, linestyle=linestyle, linewidth=1, color=color, alpha=self.config.plot_alpha, zorder=1)
        ax.scatter(x, y, s=25, facecolors='none', edgecolors=color,
                  linewidths=1, label=label)
    
    def format_axes(self):
        """Apply formatting to all axes"""
        # Density plot
        ax = self.axes['density']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        ax.set_ylabel(r'Packing fraction  $\eta = (\pi/6)\,\rho^*$')
        ax.grid(True, which='both', axis='both', ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', direction='in')
        
        # Acceptance rate
        ax = self.axes['accept_rate']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        if self.config.use_complex_scaling:
            ylabel = r'Acceptance Rate $\frac{3}{2} \sqrt{\frac{\text{Accepted}}{\text{Time} \cdot \text{Steps}}}$'
        else:
            ylabel = r'Acceptance Rate $\frac{\text{Accepted}}{\text{Time} \cdot \text{Steps}}$'
        ax.set_ylabel(ylabel)
        self._format_standard(ax)
        
        # Theoretical acceptance
        ax = self.axes['accept_theory']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        if self.config.use_complex_scaling:
            ylabel = r'Acceptance Ratio $\frac{3}{2} \sqrt{\frac{\text{Accepted}}{\text{Steps} \cdot k(k-1)}}$'
        else:
            ylabel = r'Acceptance Ratio $\frac{\text{Accepted}}{\text{Steps} \cdot \text{# trial disks}}$'
        ax.set_ylabel(ylabel)
        self._format_standard(ax)
        
        # Acceptance probability
        ax = self.axes['accept_prob']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        ax.set_ylabel(r'Acceptance Probability $\frac{\text{Accepted}}{\text{Steps}}$')
        self._format_standard(ax)
        
        # Max cycles
        ax = self.axes['max_cycles']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        ax.set_ylabel('MC Steps')
        self._format_standard(ax)
        burn_line = self.config.burn_in_cycles
        ax.axhline(y=burn_line, color='k', linestyle='--', linewidth=1,  alpha=self.config.plot_alpha)
        
        # Time
        ax = self.axes['time']
        ax.set_xlabel(r'Chemical potential  $\mu^*$')
        ax.set_ylabel('Wall Time (s)')
        self._format_standard(ax)
    
    def _format_standard(self, ax):
        """Apply standard formatting"""
        ax.grid(True, which='major', axis='y', ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', direction='in')
    
    def add_legends(self):
        """Add legends with combined add/remove markers and line styles"""
        # Create custom legend handles for datasets with line styles
        for ax_name in ['accept_rate', 'accept_theory', 'accept_prob']:
            self._add_combined_legend_with_lines(self.axes[ax_name])
        
        # Density plot - simple legend with line styles
        self._add_simple_legend_with_lines(self.axes['density'])
        
        # Performance plots
        self._add_simple_legend_with_lines(self.axes['max_cycles'], loc='upper right')
        self._add_simple_legend_with_lines(self.axes['time'])
    
    def _add_simple_legend_with_lines(self, ax, loc='lower right'):
        """Add simple legend with line styles for density/time plots"""
        handles = []
        labels = []
        
        # Add custom handles for each dataset
        for dataset in self.datasets:
            handle = Line2D([0], [0], 
                          linestyle=dataset.linestyle,
                          linewidth=1.5,
                          color=dataset.color,
                          marker='o',
                          markersize=5,
                          markerfacecolor='none',
                          markeredgecolor=dataset.color,
                          markeredgewidth=1)
            handles.append(handle)
            labels.append(dataset.label)
        
        # Add LJ EOS reference handles if present
        if self.config.plot_reference_data and self.config.lj_temperature is not None:
            T = self.config.lj_temperature
            branch_tag = {'liquid': 'liq', 'vapor': 'vap', 'both': 'liq+vap'}[self.config.eos_branch]
            for lbl, clr, ls, lw in _EOS_STYLES:
                eos_handle = Line2D([0], [0], ls=ls, lw=lw, color=clr, alpha=0.7)
                handles.append(eos_handle)
                labels.append(f'{lbl}  T*={T:.2f} ({branch_tag})')
        
        leg = ax.legend(handles, labels, ncol=2)
        leg.set_zorder(13)
    
    def _add_combined_legend_with_lines(self, ax):
        """Add legend with combined add/remove entry and line styles"""
        handles = []
        labels = []
        
        # Add custom handles for each dataset (with add/remove markers)
        for dataset in self.datasets:
            # Create combined add (circle) / remove (square) handle
            add_proxy = Line2D([0], [0], 
                              linestyle=dataset.linestyle,
                              linewidth=1.5,
                              color=dataset.color,
                              marker='o',
                              markersize=5,
                              markerfacecolor='none',
                              markeredgecolor=dataset.color,
                              markeredgewidth=1)
            rem_proxy = Line2D([0], [0],
                              linestyle=dataset.linestyle,
                              linewidth=1.5,
                              color=dataset.color,
                              marker='s',
                              markersize=5,
                              markerfacecolor='none',
                              markeredgecolor=dataset.color,
                              markeredgewidth=1)
            handles.append((add_proxy, rem_proxy))
            labels.append(dataset.label)
        
        # EOS reference lines are density-plot only; no entry needed here
        
        # Add "Add/Remove" label entry
        add_label_proxy = Line2D([0], [0], marker='o', linestyle='None',
                                markerfacecolor='none', markeredgecolor='red',
                                markersize=5.25, mew=1.0)
        rem_label_proxy = Line2D([0], [0], marker='s', linestyle='None',
                                markerfacecolor='none', markeredgecolor='red',
                                markersize=5.25, mew=1.0)
        handles.append((add_label_proxy, rem_label_proxy))
        labels.append('Add/Remove')
        
        leg = ax.legend(handles, labels,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       ncol=2, frameon=True, handlelength=3.5)
        leg.set_zorder(20)
    
    def _add_combined_legend(self, ax):
        """Add legend with combined add/remove entry"""
        handles, labels = ax.get_legend_handles_labels()
        
        # Create combined proxy artists
        add_proxy = Line2D([0], [0], marker='o', linestyle='None',
                          markerfacecolor='none', markeredgecolor='red',
                          markersize=5.25, mew=1.0)
        rem_proxy = Line2D([0], [0], marker='s', linestyle='None',
                          markerfacecolor='none', markeredgecolor='red',
                          markersize=5.25, mew=1.0)
        
        handles.append((add_proxy, rem_proxy))
        labels.append('Add/Remove')
        
        leg = ax.legend(handles, labels,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       ncol=2, frameon=True, handlelength=3.5)
        leg.set_zorder(20)
    
    def save_figures(self):
        """Save all figures to disk"""
        save_names = {
            'density': 'density.png',
            'accept_rate': 'accept_rate.png',
            'accept_theory': 'accept_ratio_theoretical.png',
            'accept_prob': 'accept_ratio.png',
            'time': 'time.png'
        }
        
        for key, filename in save_names.items():
            self.figs[key].savefig(filename, dpi=self.config.dpi)
            print(f"Saved {filename}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Main analysis pipeline"""
    config = Config()
    plotter = PlotManager(config)
    
    print("=" * 70)
    print("GCMC High Density Analysis")
    print("=" * 70)
    
    # Analyze each folder
    for folder_name in config.folders:
        folder_path = Path(folder_name)
        
        if not folder_path.exists():
            print(f"\nWarning: Folder '{folder_name}' not found, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Analyzing: {folder_name}")
        print(f"{'='*70}")
        
        dataset = analyze_folder(folder_path, config)
        plotter.plot_dataset(dataset)
    
    # Plot reference data
    if config.plot_reference_data:
        print("\n" + "="*70)
        print("Adding reference data...")
        print("="*70)
        plotter.plot_reference_data()
    
    # Finalize plots
    print("\n" + "="*70)
    print("Finalizing plots...")
    print("="*70)
    
    plotter.format_axes()
    plotter.add_legends()
    plotter.save_figures()
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
