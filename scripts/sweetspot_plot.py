#!/usr/bin/env python3
"""GCMC sweet-spot plot, new-engine edition.

Reads a runs/ tree produced by `python -m gcmc run` (each run dir holds
params.json + run.log) and renders the combined N-row x 3-col figure.
Rows are (sampler, dimension) combinations; columns are the three metrics.
The accept_rate argmax pins the sweet spot; the other two columns reuse it.

Row count is free: pass ROWS = any 1..N of the known keys. The 2-row layout
is byte-identical to the original combined figure; 1- and 3-row layouts keep
the same absolute margins, colorbar gap and panel spacing.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── which rows to draw, in order (top -> bottom) ──────────────────────────────
ROWS = ['branch_2d']            # e.g. ['branch_2d','branch_3d','aux_3d']
RUNS_ROOT = '/home/claude/ss/runs'
OUT_DIR = '/home/claude/ss'
FS = 16                         # one knob for every label/tick/title
SKIP_WINDOWS = 3                # burn-in windows to drop (fresh starts); 0 when warm-started

# ── row identity: colour + label per (sampler,dim[,aux]) key ──────────────────
METHOD_COLORS = {
    'branch_2d': '#2171b5', 'branch_3d': '#08519c',
    'aux_2d':    '#238b45', 'aux_3d':    '#006d2c',
    'hd_2d':     '#cb181d', 'hd_3d':     '#a50f15',
    'ljbranch_3d': '#6a51a3', 'ljhd_3d': '#807dba',
    'channel_3d':  '#d94801', 'ljchannel_3d': '#8c2d04',
}
METHOD_LABELS = {
    'branch_2d': 'BRANCH 2D', 'branch_3d': 'BRANCH 3D',
    'aux_2d': 'HD-aux 2D',    'aux_3d': 'HD-aux 3D',
    'hd_2d': 'HD (SR) 2D',    'hd_3d': 'HD (SR) 3D',
    'ljbranch_3d': 'LJ BRANCH 3D', 'ljhd_3d': 'LJ (SR) 3D',
    'channel_3d': 'CHANNEL 3D',    'ljchannel_3d': 'LJ CHANNEL 3D',
}
METRICS = {'add_ratio': 'Acceptance Ratio',
           'time_per_step': 'Time per Step (ms)',
           'accept_rate': 'Accept Rate (acc/s)'}
RATIO_METRICS = {'add_ratio', 'accept_rate', 'time_per_step'}
GRID_METRICS = ['add_ratio', 'time_per_step', 'accept_rate']

def depth_to_int(label): return 99 if label == 'full' else int(label.split('_')[1])
def int_to_depth(n): return 'full' if n == 99 else f'depth_{n}'

# ── collector: runs/ tree -> record dicts ─────────────────────────────────────
def _read_log(p):
    with open(p) as fh:
        head = fh.readline().lstrip('# ').split()
    df = pd.read_csv(p, sep=r'\s+', comment='#', header=None, names=head, skiprows=1)
    return df if len(df) >= 5 else None

def _row_key(params):
    s = params['sampler']['type']; dim = int(params['ensemble']['dimension'])
    aux = 'auxiliary' in params
    lj = params.get('potential', {}).get('main') == 'lennard_jones'
    if s == 'channel':
        base = 'channel'
    elif s == 'branch':
        base = 'branch'
    elif s == 'rosenbluth':
        base = 'aux' if aux else 'hd'
    else:
        base = None            # metropolis -> baseline, handled separately
    if base is None:
        return None
    if lj and base != 'aux':
        base = 'lj' + base
    return f'{base}_{dim}d'

def _stats(df, skip):
    d = df.iloc[skip:] if len(df) > skip + 3 else df
    add_acc = d['add_acc'].sum(); add_stp = d['add_stp'].sum()
    rm_stp = d['rm_stp'].sum(); wall = d['wall_s'].sum()
    steps = add_stp + rm_stp
    return {
        'add_ratio':     float(add_acc / add_stp) if add_stp else np.nan,
        'accept_rate':   float(add_acc / wall) if wall else np.nan,      # acc / s
        'time_per_step': float(1000.0 * wall / steps) if steps else np.nan,  # ms
        'eta':           float(d['pckng_frctn'].mean()),
    }

def collect_runs(root, skip=SKIP_WINDOWS):
    root = Path(root); records = []; baseline = {}
    for rd in sorted(root.glob('*')):
        pj = rd / 'params.json'; lg = rd / 'run.log'
        if not (pj.exists() and lg.exists()):
            continue
        params = json.loads(pj.read_text()); df = _read_log(lg)
        if df is None:
            continue
        st = _stats(df, skip)
        s = params['sampler']
        if s['type'] == 'metropolis':                       # baseline row
            dim = int(params['ensemble']['dimension'])
            ljb = params.get('potential', {}).get('main') == 'lennard_jones'
            baseline[('lj' if ljb else 'hs', dim)] = st
            continue
        key = _row_key(params)
        if key is None:
            continue
        if s['type'] == 'channel':
            L = float(s.get('channel_length', np.nan))      # axis IS the channel length
            third = int(round(float(s.get('channel_width', 1.0)) * 1000))  # width_key
            third_lbl = f"w{float(s.get('channel_width',1.0)):g}"
        else:
            L = float(s.get('trial_box_side', 1.5))   # axis IS the side; no x2
            third = int(s.get('depth', 0))
            third_lbl = int_to_depth(third)
        st.update(method=key, rhok=float(s.get('trial_density', np.nan)),
                  L=L, k_actual=int(s.get('trials', 0)),
                  depth=third_lbl, depth_int=third)
        records.append(st)
    return records, baseline

# ── optimum ───────────────────────────────────────────────────────────────────
def find_optimum(sub, metric, higher=True):
    v = sub.dropna(subset=[metric])
    if v.empty:
        return None
    idx = v[metric].idxmax() if higher else v[metric].idxmin()
    r = v.loc[idx]
    return r['rhok'], r['L'], int(r['depth_int'])

_ACCEPT_RATE_OPTIMA: dict = {}
_PINNED_METRICS = {'add_ratio', 'time_per_step'}

# ── one method panel (heatmap + 3 cross-sections) — layout as original ────────
def draw_method_panel(outer_spec, fig, df_method, metric, method, vmin, vmax,
                      higher, label, pinned_opt=None):
    color = METHOD_COLORS.get(method, '#333333')
    cmap = LinearSegmentedColormap.from_list('', ['#f7f7f7', color])
    opt = pinned_opt if pinned_opt is not None else find_optimum(df_method, metric, higher)
    if opt is None:
        return None
    opt_rk, opt_r, opt_d = opt
    # the label the collector stored for this third-axis value (depth_N or w<width>)
    _match = df_method[df_method['depth_int'] == opt_d]
    opt_depth_label = _match['depth'].iloc[0] if not _match.empty else int_to_depth(opt_d)

    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer_spec, width_ratios=[4, 1.35],
        height_ratios=[1.15, 4, 1.15], hspace=0.30, wspace=0.18)
    ax_heat = fig.add_subplot(inner[1, 0]); ax_right = fig.add_subplot(inner[1, 1])
    ax_bot = fig.add_subplot(inner[2, 0]);  ax_top = fig.add_subplot(inner[0, 0])
    for sp in [inner[0, 1], inner[2, 1]]:
        fig.add_subplot(sp).set_visible(False)

    sub_heat = df_method[df_method['depth'] == opt_depth_label]
    pivot = (sub_heat.pivot_table(index='rhok', columns='L', values=metric)
             .sort_index().sort_index(axis=1))
    rk_vals = list(pivot.index); L_vals = list(pivot.columns)
    if pivot.values.size == 0:
        print(f"  WARNING: {method} has no heatmap cells at the optimum "
              f"({opt_depth_label}); skipping panel"); return None
    im = ax_heat.imshow(pivot.values, aspect='auto', cmap=cmap, origin='lower',
                        vmin=vmin, vmax=vmax, interpolation='bilinear')
    opt_ri = L_vals.index(opt_r) if opt_r in L_vals else 0
    opt_rki = rk_vals.index(opt_rk) if opt_rk in rk_vals else 0
    ax_heat.plot(opt_ri, opt_rki, '*', ms=16, color='gold',
                 markeredgecolor='#444', markeredgewidth=0.7, zorder=10)
    ax_heat.axhline(opt_rki, color='white', lw=0.9, ls='--', alpha=0.55)
    ax_heat.axvline(opt_ri, color='white', lw=0.9, ls='--', alpha=0.55)
    ax_heat.set_xticks(range(len(L_vals)))
    ax_heat.set_xticklabels([f'{v:.3g}' for v in L_vals], fontsize=FS)
    ax_heat.set_yticks(range(len(rk_vals)))
    ax_heat.set_yticklabels([str(int(v)) for v in rk_vals], fontsize=FS)
    ax_heat.set_xlabel('L', fontsize=FS, labelpad=6)
    ax_heat.set_ylabel(r'$\rho_k$', fontsize=FS, labelpad=6)
    peak_val = pivot.values[opt_rki, opt_ri]
    ax_heat.text(0.02, 0.97, METHOD_LABELS.get(method, method)
                 + (('\nwidth = ' + opt_depth_label.replace('w', ''))
                    if opt_depth_label.startswith('w')
                    else ('\ndepth = ' + opt_depth_label.replace('depth_', '')))
                 + '   peak = ' + f'{peak_val:.3g}',
                 transform=ax_heat.transAxes, fontsize=FS, va='top', ha='left',
                 color='white', fontweight='bold', linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.35', fc=color, alpha=0.65, lw=0))

    cs_rk = sub_heat[sub_heat['L'] == opt_r].sort_values('rhok')
    y_pos = [rk_vals.index(rk) for rk in cs_rk['rhok'] if rk in rk_vals]
    ax_right.plot(cs_rk[metric].values[:len(y_pos)], y_pos, color=color, lw=2,
                  marker='o', ms=5, markeredgewidth=0)
    ax_right.axhline(opt_rki, color='gold', lw=1.2, ls='--', alpha=0.85)
    ax_right.set_ylim(-0.5, len(rk_vals) - 0.5); ax_right.set_yticks(range(len(rk_vals)))
    ax_right.set_yticklabels([str(int(v)) for v in rk_vals], fontsize=FS)
    ax_right.yaxis.set_label_position('right'); ax_right.yaxis.tick_right()
    ax_right.tick_params(axis='x', labelsize=FS, rotation=30)
    ax_right.spines[['top', 'left']].set_visible(False)
    ax_right.set_xlabel(label, fontsize=FS, labelpad=6)
    ax_right.set_ylabel(r'$\rho_k$', fontsize=FS, labelpad=22, rotation=270)
    ax_right.set_title('vary ' + r'$\rho_k$' + f'  (L = {opt_r:.3g})', fontsize=FS, pad=6)

    cs_r = sub_heat[sub_heat['rhok'] == opt_rk].sort_values('L')
    x_pos = [L_vals.index(lv) for lv in cs_r['L'] if lv in L_vals]
    ax_bot.plot(x_pos, cs_r[metric].values[:len(x_pos)], color=color, lw=2,
                marker='o', ms=5, markeredgewidth=0)
    ax_bot.axvline(opt_ri, color='gold', lw=1.2, ls='--', alpha=0.85)
    ax_bot.set_xlim(-0.5, len(L_vals) - 0.5); ax_bot.set_xticks(range(len(L_vals)))
    ax_bot.set_xticklabels([f'{v:.3g}' for v in L_vals], fontsize=FS, rotation=45, ha='right')
    ax_bot.tick_params(axis='y', labelsize=FS); ax_bot.spines[['top', 'right']].set_visible(False)
    ax_bot.set_xlabel('L', fontsize=FS, labelpad=4)
    ax_bot.set_ylabel(label, fontsize=FS, labelpad=6)
    ax_bot.set_title('vary L  (' + r'$\rho_k$' + f' = {opt_rk:.0f})', fontsize=FS, pad=6)

    is_chan = method.startswith('channel') or method.startswith('ljchannel')
    third_name = 'channel width' if is_chan else 'depth'
    fmt3 = (lambda v: f'{v/1000.0:g}') if is_chan else (lambda v: 'full' if v == 99 else str(int(v)))
    cs_d = df_method[(np.isclose(df_method['L'], opt_r, atol=1e-6))
                     & (np.isclose(df_method['rhok'], opt_rk, atol=1e-6))].sort_values('depth_int')
    xs = cs_d['depth_int'].values
    ax_top.plot(xs, cs_d[metric].values, color=color, lw=2, marker='o', ms=5, markeredgewidth=0)
    ax_top.axvline(opt_d, color='gold', lw=1.2, ls='--', alpha=0.85)
    ax_top.set_xticks(xs)
    ax_top.set_xticklabels([fmt3(v) for v in xs], fontsize=FS, rotation=(45 if is_chan else 0),
                           ha=('right' if is_chan else 'center'))
    ax_top.tick_params(axis='y', labelsize=FS); ax_top.spines[['top', 'right']].set_visible(False)
    ax_top.set_xlabel(third_name, fontsize=FS, labelpad=4)
    ax_top.set_ylabel(label, fontsize=FS, labelpad=6)
    ax_top.set_title(f'vary {third_name}  (' + r'$\rho_k$' + f' = {opt_rk:.0f}, L = {opt_r:.3g})',
                     fontsize=FS, pad=6)
    if metric == 'time_per_step' and method.startswith('branch'):
        ax_right.set_xscale('log'); ax_bot.set_yscale('log'); ax_top.set_yscale('log')
    return im

# ── combined N-row x 3-col figure; margins preserve the 2-row look ────────────
def make_combined(df, out_dir, baseline=None, rows=None):
    rows = rows or ROWS
    if df.empty or 'method' not in df.columns:
        print('no usable runs collected'); return
    methods_present = [m for m in rows if m in set(df['method'])]
    metrics_present = [m for m in GRID_METRICS if m in df.columns]
    if not methods_present:
        print('no rows present'); return
    n_rows, n_cols = len(methods_present), len(metrics_present)

    # pin sweet spot from accept_rate per row
    _ACCEPT_RATE_OPTIMA.clear()
    for m in methods_present:
        opt = find_optimum(df[df['method'] == m], 'accept_rate', higher=True)
        if opt:
            _ACCEPT_RATE_OPTIMA[m] = opt
            print(f'  pin {m}: rhok={opt[0]:.0f} L={opt[1]:.3g} depth={opt[2]}')

    H = 8.4 * n_rows                       # inches; original 2-row -> 16.8
    fig = plt.figure(figsize=(8.6 * n_cols, H), facecolor='white')
    # keep the original 2-row absolute margins for any n_rows
    top = 1 - 0.588 / H; bottom = 2.10 / H
    cbar_y, cbar_h, tag_off = 0.605 / H, 0.185 / H, 0.84 / H
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.30, wspace=0.42,
                              left=0.05, right=0.985, top=top, bottom=bottom)
    col_ims = {c: None for c in range(n_cols)}
    for col, metric in enumerate(metrics_present):
        label = METRICS.get(metric, metric)
        is_ratio = (metric in RATIO_METRICS) and bool(baseline)
        # per-row baseline-normalised frames, then ONE colour scale down the column
        row_subs = {}
        for method in methods_present:
            sub = df[df['method'] == method].copy()
            if sub.dropna(subset=[metric]).empty:
                continue
            dim = 2 if method.endswith('2d') else 3
            pot = 'lj' if method.startswith('lj') else 'hs'
            std = (baseline.get((pot, dim), {}).get(metric) if is_ratio else None)
            if std:
                sub[metric] = sub[metric] / std
            row_subs[method] = sub
        if not row_subs:
            continue
        col_vals = pd.concat([s[metric] for s in row_subs.values()]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        vmin, vmax = float(col_vals.quantile(0.02)), float(col_vals.quantile(0.98))
        for row, method in enumerate(methods_present):
            sub = row_subs.get(method)
            if sub is None:
                continue
            pinned = _ACCEPT_RATE_OPTIMA.get(method) if metric in _PINNED_METRICS else None
            im = draw_method_panel(outer[row, col], fig, sub, metric, method,
                                   vmin, vmax, metric != 'time_per_step', label,
                                   pinned_opt=pinned)
            if im is not None:
                col_ims[col] = im
            pbox = outer[row, col].get_position(fig)
            fig.text((pbox.x0 + pbox.x1) / 2, pbox.y0 - tag_off,
                     f'({chr(97 + row * n_cols + col)})', ha='center', va='top', fontsize=FS)
    for col, metric in enumerate(metrics_present):
        im = col_ims[col]
        if im is None:
            continue
        box = outer[n_rows - 1, col].get_position(fig)
        w = (box.x1 - box.x0) * 0.72
        cax = fig.add_axes([box.x0, cbar_y, w, cbar_h])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label(METRICS.get(metric, metric), fontsize=FS, labelpad=4)
        cb.ax.tick_params(labelsize=FS)
    out = Path(out_dir) / 'gcmc_combined_sweetspot.png'
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('  saved ->', out)
    return out

def main():
    """Standalone replot: rebuild the figure from one or more runs/ trees without
    running anything. Rows may live in separate directories; pass each tree to
    --runs and the rows in top-to-bottom order to --rows."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=[RUNS_ROOT],
                    help='one or more runs/ dirs (space separated)')
    ap.add_argument('--out', default=OUT_DIR, help='where to write the png')
    ap.add_argument('--rows', nargs='+', default=list(ROWS),
                    help='row keys, top to bottom (space separated)')
    ap.add_argument('--baseline', default=None,
                    help='runs/ dir whose metropolis run normalises ALL rows; '
                         'ensures every row is compared against the same std GCMC')
    ap.add_argument('--font', type=float, default=None, help='font size for the whole figure')
    ap.add_argument('--depths', default=None, help='keep only these depths, e.g. 0,1,2,3,4,5')
    ap.add_argument('--rhok', default=None, help='keep only these rho_k values')
    ap.add_argument('--L', default=None, help='keep only these L values')
    ap.add_argument('--skip', type=int, default=0, help='burn-in windows to drop')
    a = ap.parse_args()

    global FS
    if a.font:
        FS = a.font

    # gather records from every tree; merge baselines by (potential, dim)
    frames = []; baseline = {}
    for r in a.runs:
        recs, base = collect_runs(r, skip=a.skip)
        frames.append(pd.DataFrame(recs))
        baseline.update(base)                     # last tree wins per (pot, dim)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print('no usable runs in', a.runs); return

    # one canonical baseline for ALL rows, so nothing is normalised against a
    # different std GCMC than its neighbours
    if a.baseline:
        _, base_only = collect_runs(a.baseline, skip=a.skip)
        if not base_only:
            print(f'WARNING: no metropolis run found in {a.baseline}; using per-tree baselines')
        baseline = base_only or baseline
        print(f'baseline: forced from {a.baseline} -> keys {sorted(baseline)}')
    else:
        print(f'baseline keys present: {sorted(baseline)} '
              f'(one per potential+dim; same key => same std GCMC for those rows)')

    # a row appearing in two trees would double up; keep the first occurrence
    if df.duplicated(subset=['method', 'depth_int', 'rhok', 'L']).any():
        df = df.drop_duplicates(subset=['method', 'depth_int', 'rhok', 'L'], keep='first')

    nums = lambda t: [float(x) for x in t.replace(',', ' ').split()] if t else None
    for col, vals in (('depth_int', nums(a.depths)), ('rhok', nums(a.rhok)), ('L', nums(a.L))):
        if vals:
            df = df[df[col].isin(vals)]

    present = [r for r in a.rows if r in set(df['method'])]
    missing = [r for r in a.rows if r not in present]
    if missing:
        print(f'WARNING: requested but not found in the given trees: {missing}')
    print(f'records: {len(df)}  rows: {present}')
    print(f'rho_k: {sorted(df.rhok.unique())}')
    print(f'L:     {sorted(df.L.unique())}')
    make_combined(df, a.out, baseline=baseline, rows=a.rows)


if __name__ == '__main__':
    main()
