#!/usr/bin/env python3
"""
Quick plotting script for GCMC data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
# Can be a single folder path (string) or list of folder paths
FOLDER_PATH = [
"n_-1_k_1_r_1.0",
"n_0_k_4000_r_1.0",
"n_1_k_4000_r_1.0",
"n_2_k_4000_r_1.0",
"n_3_k_4000_r_1.0",
"n_4_k_4000_r_1.0",
"n_5_k_4000_r_1.0",
"n_6_k_4000_r_1.0",
"n_7_k_4000_r_1.0",
"n_8_k_4000_r_1.0",
]

USE_3D = True                       # Set to True for 3D reference data
OUTPUT_PREFIX = ""                   # Optional prefix for output files
MIN_MU = 1.0                         # Minimum chemical potential to plot
MAX_MU = 15.0                        # Maximum chemical potential to plot
N_BLOCKS = 10                        # Number of blocks for error estimation
# ============================================================================


@dataclass
class ReferenceData:
    """Reference data from literature or standard GCMC"""
    
    # Döge et al. 2004 data
    doge_mu = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 13.0, 14.0])
    doge_density = np.array([
        0.4852337933295064, 0.5209654196417268, 0.5560623587623288,
        0.5841835660485017, 0.6083410134252748, 0.6480937011684030,
        0.6827737474397285, 0.7031100665749955, 0.7425042707323461
    ])
    
    # 2D GCMC reference
    gcmc_mu = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 11.0, 12.0, 13.0, 14.0])
    gcmc_density = np.array([
        2.270675556897113200e-01, 3.080602970690536346e-01, 3.782747352812219077e-01,
        4.363617914752629390e-01, 4.839704494940910706e-01, 5.232661971684527957e-01,
        5.560959978376565083e-01, 5.835854836379568145e-01, 6.071117977095142892e-01,
        6.288976877716574432e-01, 6.472474696710268649e-01, 6.651831369839993346e-01,
        6.877258145410410917e-01, 6.844738555635505595e-01, 6.896073995639250276e-01
    ])
    gcmc_add = np.array([
        2.891912503726839500e-01, 1.442707626521170516e-01, 6.519907105434773198e-02,
        2.766741063376092560e-02, 1.129431482552931999e-02, 4.494387518307122777e-03,
        1.754243731718720068e-03, 6.781718213641284949e-04, 2.597537347110750647e-04,
        9.907387827391635378e-05, 3.726349732698264097e-05, 1.425937787800049701e-05,
        5.499936999322463451e-06, 1.964139609048386726e-06, 7.366460183417525188e-07
    ])
    gcmc_remove = np.array([
        2.891461026616193300e-01, 1.442799404472037972e-01, 6.519362417941879095e-02,
        2.766616242180242394e-02, 1.129476174361521493e-02, 4.494510775339814225e-03,
        1.754204483283263111e-03, 6.781834934677754938e-04, 2.597481156343808893e-04,
        9.906431998189965899e-05, 3.726135695043652888e-05, 1.425865977545423057e-05,
        5.493214048463515394e-06, 1.960875692151087871e-06, 7.300970951624917864e-07
    ])
    gcmc_time = np.array([
        5.649854231347019029e-01, 5.614758587881824026e-01, 5.481599999999998696e-01,
        5.610068753129695551e-01, 5.633285928893341232e-01, 5.664876614922382814e-01,
        5.669880470706059050e-01, 5.655284777165749954e-01, 5.725731497245867274e-01,
        5.828921912767170976e-01, 5.694463770518982670e-01, 5.745054060610119695e-01,
        5.641966677180670908e-01, 5.602104753205231491e-01, 5.602104753205231491e-01
    ])
    
    # 3D GCMC reference
    gcmc_3d_mu = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                           10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    gcmc_3d_density = np.array([
        1.315971849469556587e-01, 1.777764887859887843e-01, 2.201270819816010527e-01,
        2.575422468146847388e-01, 2.901690246029303144e-01, 3.186848863452940717e-01,
        3.437908885216878763e-01, 3.659841640920545314e-01, 3.856813969676097242e-01,
        4.036637873454934633e-01, 4.191964483546940512e-01, 4.338163316073631037e-01,
        4.466816920190951334e-01, 4.596414805275544357e-01, 4.688149806038837686e-01,
        4.774655813179732311e-01
    ])
    gcmc_3d_add = np.array([
        2.512388573021117066e-01, 1.248651942100217210e-01, 5.688506482839391504e-02,
        2.448244816892913150e-02, 1.015328155972471807e-02, 4.101652909655893048e-03,
        1.627188991989418241e-03, 6.384524512115662305e-04, 2.469277552699201484e-04,
        9.507566048801454857e-05, 3.740398609627425069e-05, 1.377834027020214572e-05,
        5.958161772312180563e-06, 2.642654030790942026e-06, 1.787233786172765253e-06,
        1.359862687686877607e-06
    ])
    gcmc_3d_remove = np.array([
        2.512938542229078420e-01, 1.248668301626759125e-01, 5.688246173631175684e-02,
        2.448410975968275463e-02, 1.015281144813450290e-02, 4.101898898414743297e-03,
        1.626996599270115864e-03, 6.386110118627246444e-04, 2.469149311785865373e-04,
        9.488259869745393517e-05, 3.711957488602303767e-05, 1.353891862836033746e-05,
        5.417992186131369596e-06, 1.724558263621654928e-06, 7.556361311721715196e-07,
        2.405945918239101026e-07
    ])
    gcmc_3d_time = np.array([
        1.780886927173257916e+00, 1.959289307743853126e+00, 2.054996565224701754e+00,
        2.153571219905308975e+00, 2.370680867292104477e+00, 2.448840586667430763e+00,
        2.518604591855537311e+00, 2.585606036068079927e+00, 2.641157235621521160e+00,
        2.688234360491651209e+00, 2.726277851374213856e+00, 2.771996169079930183e+00,
        2.801911775320712561e+00, 2.840434226173151888e+00, 2.874486147135109970e+00,
        2.891618630290097691e+00
    ])


def block_average(data, n_blocks):
    """Calculate block average and standard error."""
    n_data = len(data)
    block_size = n_data // n_blocks
    
    if block_size == 0:
        return np.mean(data), 0.0
    
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n_data
        blocks.append(np.mean(data[start:end]))
    
    mean = np.mean(blocks)
    error = np.std(blocks, ddof=1) / np.sqrt(n_blocks)
    
    return mean, error


def parse_line(line):
    """Parse a single line from output file."""
    pattern = r'(\d+\.?\d*)_density_([\d.]+)_add_accept_(\d+)_add_step_(\d+)_remove_accept_(\d+)_remove_step_(\d+)_.*_time_per_step_([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'steps': float(match.group(1)),
            'density': float(match.group(2)),
            'add_accept': int(match.group(3)),
            'add_step': int(match.group(4)),
            'remove_accept': int(match.group(5)),
            'remove_step': int(match.group(6)),
            'time_per_step': float(match.group(7))
        }
    return None


def process_file(filepath, n_blocks):
    """Process a single output file and return averaged quantities with errors."""
    data = []
    with open(filepath) as f:
        for line in f:
            parsed = parse_line(line.strip())
            if parsed:
                data.append(parsed)
    
    if not data:
        return None
    
    densities = [d['density'] for d in data]
    add_accepts = [d['add_accept'] for d in data]
    add_steps = [d['add_step'] for d in data]
    remove_accepts = [d['remove_accept'] for d in data]
    remove_steps = [d['remove_step'] for d in data]
    times = [d['time_per_step'] for d in data]
    
    # Calculate block averages and errors
    density_mean, density_err = block_average(densities, n_blocks)
    time_mean, time_err = block_average(times, n_blocks)
    
    # Total acceptances
    total_add_accepts = sum(add_accepts)
    total_add_steps = sum(add_steps)
    total_remove_accepts = sum(remove_accepts)
    total_remove_steps = sum(remove_steps)
    
    add_ratio = total_add_accepts / total_add_steps if total_add_steps > 0 else 0
    remove_ratio = total_remove_accepts / total_remove_steps if total_remove_steps > 0 else 0
    
    # Error estimation for ratios (using binomial approximation)
    add_ratio_err = np.sqrt(add_ratio * (1 - add_ratio) / total_add_steps) if total_add_steps > 0 else 0
    remove_ratio_err = np.sqrt(remove_ratio * (1 - remove_ratio) / total_remove_steps) if total_remove_steps > 0 else 0
    
    add_rate = add_ratio / (time_mean / 5000)
    # Error propagation for rate
    add_rate_err = add_rate * np.sqrt((add_ratio_err/add_ratio)**2 + (time_err/time_mean)**2) if add_ratio > 0 else 0
    
    remove_rate = remove_ratio / (time_mean / 5000)
    remove_rate_err = remove_rate * np.sqrt((remove_ratio_err/remove_ratio)**2 + (time_err/time_mean)**2) if remove_ratio > 0 else 0
    
    return {
        'density': density_mean,
        'density_err': density_err,
        'add_ratio': add_ratio,
        'add_ratio_err': add_ratio_err,
        'remove_ratio': remove_ratio,
        'remove_ratio_err': remove_ratio_err,
        'add_rate': add_rate,
        'add_rate_err': add_rate_err,
        'remove_rate': remove_rate,
        'remove_rate_err': remove_rate_err,
        'time_per_step': time_mean,
        'time_per_step_err': time_err
    }


def collect_data(folder_path, min_mu, max_mu, n_blocks):
    """Collect data from all dyn_*.out files."""
    folder = Path(folder_path)
    results = {}
    
    for filepath in folder.glob('mu_*_*.out'):
        try:
            chem_pot = float(filepath.stem.split('_')[1])
            if min_mu <= chem_pot <= max_mu:
                data = process_file(filepath, n_blocks)
                if data:
                    results[chem_pot] = data
        except (ValueError, IndexError):
            continue
    
    chem_pots = sorted(results.keys())
    return {
        'chem_pot': np.array(chem_pots),
        'density': np.array([results[cp]['density'] for cp in chem_pots]),
        'density_err': np.array([results[cp]['density_err'] for cp in chem_pots]),
        'add_ratio': np.array([results[cp]['add_ratio'] for cp in chem_pots]),
        'add_ratio_err': np.array([results[cp]['add_ratio_err'] for cp in chem_pots]),
        'remove_ratio': np.array([results[cp]['remove_ratio'] for cp in chem_pots]),
        'remove_ratio_err': np.array([results[cp]['remove_ratio_err'] for cp in chem_pots]),
        'add_rate': np.array([results[cp]['add_rate'] for cp in chem_pots]),
        'add_rate_err': np.array([results[cp]['add_rate_err'] for cp in chem_pots]),
        'remove_rate': np.array([results[cp]['remove_rate'] for cp in chem_pots]),
        'remove_rate_err': np.array([results[cp]['remove_rate_err'] for cp in chem_pots]),
        'time_per_step': np.array([results[cp]['time_per_step'] for cp in chem_pots]),
        'time_per_step_err': np.array([results[cp]['time_per_step_err'] for cp in chem_pots])
    }


def filter_ref_data(mu, data, min_mu, max_mu):
    """Filter reference data by chemical potential range."""
    mask = (mu >= min_mu) & (mu <= max_mu)
    return mu[mask], data[mask]


def get_ref_time_dict(ref_mu, ref_time):
    """Create dictionary for reference times at integer mu values."""
    return {int(mu): time for mu, time in zip(ref_mu, ref_time)}


def add_combined_legend(ax, label, color, linestyle='-'):
    """Add legend with combined add (triangle) / remove (square) entry."""
    add_proxy = Line2D([0], [0],
                      linestyle=linestyle,
                      linewidth=1.5,
                      color=color,
                      marker='^',
                      markersize=6,
                      markerfacecolor='none',
                      markeredgecolor=color,
                      markeredgewidth=1.5)
    rem_proxy = Line2D([0], [0],
                      linestyle=linestyle,
                      linewidth=1.5,
                      color=color,
                      marker='s',
                      markersize=6,
                      markerfacecolor='none',
                      markeredgecolor=color,
                      markeredgewidth=1.5)
    return (add_proxy, rem_proxy), label


def plot_data(all_data, ref_data, folder_paths, use_3d=False, output_prefix='', min_mu=None, max_mu=None):
    """Create all plots."""
    
    # Select reference based on dimensionality
    if use_3d:
        ref_mu = ref_data.gcmc_3d_mu
        ref_density = ref_data.gcmc_3d_density
        ref_add = ref_data.gcmc_3d_add
        ref_remove = ref_data.gcmc_3d_remove
        ref_time = ref_data.gcmc_3d_time
    else:
        ref_mu = ref_data.gcmc_mu
        ref_density = ref_data.gcmc_density
        ref_add = ref_data.gcmc_add
        ref_remove = ref_data.gcmc_remove
        ref_time = ref_data.gcmc_time
    
    # Filter reference data
    ref_mu_filt, ref_density_filt = filter_ref_data(ref_mu, ref_density, min_mu, max_mu)
    ref_mu_add, ref_add_filt = filter_ref_data(ref_mu, ref_add, min_mu, max_mu)
    ref_mu_rem, ref_remove_filt = filter_ref_data(ref_mu, ref_remove, min_mu, max_mu)
    ref_mu_time, ref_time_filt = filter_ref_data(ref_mu, ref_time, min_mu, max_mu)
    doge_mu_filt, doge_density_filt = filter_ref_data(ref_data.doge_mu, ref_data.doge_density, min_mu, max_mu)
    
    ref_add_rate = ref_add_filt / (ref_time_filt / 5000)
    ref_remove_rate = ref_remove_filt / (ref_time_filt / 5000)
    
    # Create dictionary for reference times
    ref_time_dict = get_ref_time_dict(ref_mu_time, ref_time_filt)
    
    # Get colors for each dataset
    colors = [f'C{i}' for i in range(len(all_data))]
    
    # 1. Density plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, folder_path, color) in enumerate(zip(all_data, folder_paths, colors)):
        ax.errorbar(data['chem_pot'], data['density'], yerr=data['density_err'], 
                    fmt='o-', label=folder_path, markersize=8, capsize=3, color=color)
    
    if not USE_3D:
        ax.plot(doge_mu_filt, doge_density_filt, 's--', label='Döge', alpha=0.7, color='gray')
    
    ax.plot(ref_mu_filt, ref_density_filt, '^--', label='std GCMC', alpha=0.7, color='black')
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}density.png', dpi=150)
    
    # 2. Accept ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, folder_path, color) in enumerate(zip(all_data, folder_paths, colors)):
        ax.errorbar(data['chem_pot'], data['add_ratio'], yerr=data['add_ratio_err'],
                    fmt='^-', markersize=8, capsize=3, color=color)
        ax.errorbar(data['chem_pot'], data['remove_ratio'], yerr=data['remove_ratio_err'],
                    fmt='s-', markersize=8, capsize=3, color=color)
    ax.semilogy(ref_mu_add, ref_add_filt, '^--', color='gray', alpha=0.7)
    ax.semilogy(ref_mu_rem, ref_remove_filt, 's--', color='gray', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Accept Ratio')
    
    # Custom legend
    handles = []
    labels = []
    for folder_path, color in zip(folder_paths, colors):
        h, l = add_combined_legend(ax, folder_path, color, '-')
        handles.append(h)
        labels.append(l)
    h_ref, l_ref = add_combined_legend(ax, 'std GCMC', 'gray', '--')
    handles.append(h_ref)
    labels.append(l_ref)
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}accept_ratio.png', dpi=150)
    
    # 3. Accept rate
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, folder_path, color) in enumerate(zip(all_data, folder_paths, colors)):
        ax.errorbar(data['chem_pot'], data['add_rate'], yerr=data['add_rate_err'],
                    fmt='^-', markersize=8, capsize=3, color=color)
        ax.errorbar(data['chem_pot'], data['remove_rate'], yerr=data['remove_rate_err'],
                    fmt='s-', markersize=8, capsize=3, color=color)
    ax.semilogy(ref_mu_add, ref_add_rate, '^--', color='gray', alpha=0.7)
    ax.semilogy(ref_mu_rem, ref_remove_rate, 's--', color='gray', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Accept Rate')
    
    # Custom legend
    handles = []
    labels = []
    for folder_path, color in zip(folder_paths, colors):
        h, l = add_combined_legend(ax, folder_path, color, '-')
        handles.append(h)
        labels.append(l)
    h_ref, l_ref = add_combined_legend(ax, 'std GCMC', 'gray', '--')
    handles.append(h_ref)
    labels.append(l_ref)
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}accept_rate.png', dpi=150)
    
    # 4. Normalized time per step
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, folder_path, color) in enumerate(zip(all_data, folder_paths, colors)):
        normalized_time = np.array([data['time_per_step'][j] / ref_time_dict[int(mu)] 
                                    for j, mu in enumerate(data['chem_pot'])])
        normalized_time_err = np.array([data['time_per_step_err'][j] / ref_time_dict[int(mu)]
                                        for j, mu in enumerate(data['chem_pot'])])
        ax.errorbar(data['chem_pot'], normalized_time, yerr=normalized_time_err,
                    fmt='o-', label=folder_path, markersize=8, capsize=3, color=color)
    ax.axhline(y=1.0, color='gray', linestyle='--', label='std GCMC', alpha=0.7)
    ax.set_xlabel('Chemical Potential')
    ax.set_ylabel('Time per Step (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}time.png', dpi=150)
    
    plt.show()


def main():
    # Handle both single folder (string) and multiple folders (list)
    if isinstance(FOLDER_PATH, str):
        folder_paths = [FOLDER_PATH]
    else:
        folder_paths = FOLDER_PATH
    
    # Collect data from all folders
    all_data = []
    for folder_path in folder_paths:
        data = collect_data(folder_path, MIN_MU, MAX_MU, N_BLOCKS)
        all_data.append(data)
    
    ref_data = ReferenceData()
    
    output_prefix = OUTPUT_PREFIX if OUTPUT_PREFIX.endswith('_') or OUTPUT_PREFIX == '' else OUTPUT_PREFIX + '_'
    plot_data(all_data, ref_data, folder_paths, USE_3D, output_prefix, MIN_MU, MAX_MU)
    
    print(f"Plots saved with prefix: {output_prefix if output_prefix else '(none)'}")
    print(f"Reference data: {'3D' if USE_3D else '2D'}")
    print(f"Chemical potential range: {MIN_MU} to {MAX_MU}")
    print(f"Block averaging: {N_BLOCKS} blocks")
    print(f"Folders analyzed: {len(folder_paths)}")


if __name__ == '__main__':
    main()
