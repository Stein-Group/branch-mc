#!/usr/bin/env python3
"""
GCMC plotter for the new runs/<group>/<hash>__<mu>/ layout.

Everything is driven by each run's params.json, never by folder or file names.
The only knobs you normally touch are in the CONFIG block below.

run.log columns are read by name through scripts/runlog.py, so logs from before
the raw-column format read too. A log without wall_s skips the wall-time plots.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import runlog  # noqa: E402

# ============================================================================
# CONFIG
# ============================================================================
RUN_GROUPS = [
    "qa_isotherm_branch"
    # a group dir (holds the per-mu leaf dirs), or a single leaf dir
]
MODE = "run"                  # "run"  -> raw curves + reference overlay
                              # "compare" -> efficiency as a multiple of std-Metropolis GCMC
BURN_IN_M = 0.0               # discard rows with MC_cycle < this (in millions)
MIN_MU = -np.inf
MAX_MU = np.inf
N_BLOCKS = 10                 # blocks for density stderr
OUTDIR = "figs"
DPI = 300
WALL_S_IS_CUMULATIVE = False  # wall_s is per output-window seconds; flip if it is cumulative
LABEL_OVERRIDES = {'qa_isotherm_branch':'BRANCH MC (QA)'}          # {group_dir_basename: "nice label"}
# ============================================================================

PALETTE = ['#5B54A8', "A4AB57", '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
           '#949494', '#D55E00', '#56B4E9', '#882255', '#999933']


# ----------------------------------------------------------------------------
# Reference data (std GCMC + literature). Reused verbatim as the compare baseline.
# ----------------------------------------------------------------------------
# 2D hard disk -- Döge et al. 2004 (literature density)
DOGE_MU = np.array([4., 5., 6., 7., 8., 10., 12., 13., 14.])
DOGE_RHO = np.array([0.4852337933295064, 0.5209654196417268, 0.5560623587623288,
                     0.5841835660485017, 0.6083410134252748, 0.6480937011684030,
                     0.6827737474397285, 0.7031100665749955, 0.7425042707323461])

# 2D hard disk -- standard Metropolis GCMC
HD2D_MU = np.arange(0., 15.)
HD2D_RHO = np.array([0.2270675556897113, 0.3080602970690536, 0.3782747352812219,
                     0.4363617914752629, 0.4839704494940911, 0.5232661971684528,
                     0.5560959978376565, 0.5835854836379568, 0.6071117977095143,
                     0.6288976877716574, 0.6472474696710269, 0.6651831369839993,
                     0.6877258145410411, 0.6844738555635506, 0.6896073995639250])
HD2D_ADD = np.array([2.891912503726840e-01, 1.442707626521171e-01, 6.519907105434773e-02,
                     2.766741063376093e-02, 1.129431482552932e-02, 4.494387518307123e-03,
                     1.754243731718720e-03, 6.781718213641285e-04, 2.597537347110751e-04,
                     9.907387827391635e-05, 3.726349732698264e-05, 1.425937787800050e-05,
                     5.499936999322463e-06, 1.964139609048387e-06, 7.366460183417525e-07])
HD2D_REM = np.array([2.891461026616193e-01, 1.442799404472038e-01, 6.519362417941879e-02,
                     2.766616242180242e-02, 1.129476174361521e-02, 4.494510775339814e-03,
                     1.754204483283263e-03, 6.781834934677755e-04, 2.597481156343809e-04,
                     9.906431998189966e-05, 3.726135695043653e-05, 1.425865977545423e-05,
                     5.493214048463515e-06, 1.960875692151088e-06, 7.300970951624918e-07])
HD2D_TIME = np.array([0.5649854231347019, 0.5614758587881824, 0.5481599999999999,
                      0.5610068753129696, 0.5633285928893341, 0.5664876614922383,
                      0.5669880470706059, 0.5655284777165750, 0.5725731497245867,
                      0.5828921912767171, 0.5694463770518983, 0.5745054060610120,
                      0.5641966677180671, 0.5602104753205231, 0.5602104753205231])

# 3D hard sphere -- standard Metropolis GCMC
HD3D_MU = np.arange(0., 16.)
HD3D_RHO = np.array([0.1315971849469557, 0.1777764887859888, 0.2201270819816011,
                     0.2575422468146847, 0.2901690246029303, 0.3186848863452941,
                     0.3437908885216879, 0.3659841640920545, 0.3856813969676097,
                     0.4036637873454935, 0.4191964483546941, 0.4338163316073631,
                     0.4466816920190951, 0.4596414805275544, 0.4688149806038838,
                     0.4774655813179732])
HD3D_ADD = np.array([2.512388573021117e-01, 1.248651942100217e-01, 5.688506482839392e-02,
                     2.448244816892913e-02, 1.015328155972472e-02, 4.101652909655893e-03,
                     1.627188991989418e-03, 6.384524512115662e-04, 2.469277552699201e-04,
                     9.507566048801455e-05, 3.740398609627425e-05, 1.377834027020215e-05,
                     5.958161772312181e-06, 2.642654030790942e-06, 1.787233786172765e-06,
                     1.359862687686878e-06])
HD3D_REM = np.array([2.512938542229078e-01, 1.248668301626759e-01, 5.688246173631176e-02,
                     2.448410975968275e-02, 1.015281144813450e-02, 4.101898898414743e-03,
                     1.626996599270116e-03, 6.386110118627246e-04, 2.469149311785865e-04,
                     9.488259869745394e-05, 3.711957488602304e-05, 1.353891862836034e-05,
                     5.417992186131370e-06, 1.724558263621655e-06, 7.556361311721715e-07,
                     2.405945918239101e-07])
HD3D_TIME = np.array([1.780886927173258, 1.959289307743853, 2.054996565224702,
                      2.153571219905309, 2.370680867292104, 2.448840586667431,
                      2.518604591855537, 2.585606036068080, 2.641157235621521,
                      2.688234360491651, 2.726277851374214, 2.771996169079930,
                      2.801911775320713, 2.840434226173152, 2.874486147135110,
                      2.891618630290098])

#########
#HD3D_MU = HD3D_MU - np.log(HD3D_RHO) - np.log(6/np.pi) #r=0.5
########

# Meroni-Levesque-Weis 1996 Table I, equal-diameter eta_matrix=0.1259.
# betamu_1 is the TOTAL chemical potential (== this code's mu, no conversion):
# verified because Meroni betamu_1=5.007 at rho1=0.383 minus ln(rho1) = 5.97,
# the EXCESS value Fernaud reports at the same state. So overlay directly vs mu.
MERONI_MU = np.array([-1.839, -1.162, -0.572, 1.769, 5.007])
MERONI_RHO = np.array([0.0178, 0.0293, 0.0424, 0.1121, 0.2005])


#MERONI_MU = MERONI_MU - np.log(HD3D_RHO) - np.log(6/np.pi) #r=0.5

# Lomba-Given-Stell-Weis-Levesque 1993 Table II, equal-diameter HS matrix
# (eta_matrix=0.157, box 9.48; eta_matrix=0.319, box 9.41). betamu_1 is NOT
# tabulated there -- GCMC was run at fixed activity and only rho1 reported -- so
# there is no mu-vs-eta isotherm to overlay; Lomba is a STRUCTURE-only reference
# (compare g11/g01). Hence it does not appear on the density figure.

def lj_eos_rho(T, mu_grid, branch='both'):
    """rho*(mu*) from three LJ EOS (KN94, JZG93, Thol16) via teqp; NaN where no root.
    Returns dict name->array, or None if teqp is unavailable."""
    try:
        import teqp
        from scipy.optimize import brentq
    except ImportError:
        warnings.warn("teqp/scipy not available -- skipping LJ EOS overlay.")
        return None

    z = np.array([1.0])
    models = {
        'KN 1994':   teqp.make_model({'kind': 'LJ126_KolafaNezbeda1994', 'model': {}}),
        'JZG 1993':  teqp.make_model({'kind': 'LJ126_Johnson1993', 'model': {}}),
        'Thol 2016': teqp.build_LJ126_TholJPCRD2016(),
    }

    def mu_of(model, rho):
        return T * (np.log(rho) + model.get_Ar00(T, rho, z) + model.get_Ar01(T, rho, z))

    # vapour ceiling = 0.9 * spinodal peak (from the KN model); 0.55 if supercritical
    rscan = np.linspace(1e-4, 0.60, 1000)
    mscan = np.array([mu_of(models['KN 1994'], r) for r in rscan])
    pk = int(np.argmax(mscan))
    vap_hi = 0.55 if pk >= len(rscan) - 5 else float(rscan[pk]) * 0.90
    brackets = {'liquid': (0.40, 1.20), 'vapor': (1e-4, vap_hi)}

    def root(model, mu, lo, hi):
        try:
            if (mu_of(model, lo) - mu) * (mu_of(model, hi) - mu) > 0:
                return np.nan
            return brentq(lambda r: mu_of(model, r) - mu, lo, hi, xtol=1e-10, rtol=1e-10)
        except ValueError:
            return np.nan

    def invert(model, mu):
        if branch == 'both':
            r = root(model, mu, *brackets['liquid'])
            return r if not np.isnan(r) else root(model, mu, *brackets['vapor'])
        return root(model, mu, *brackets[branch])

    return {name: np.array([invert(m, mu) for mu in mu_grid]) for name, m in models.items()}


# ----------------------------------------------------------------------------
# IO + stats
# ----------------------------------------------------------------------------
def block_average(x, n_blocks):
    """Mean and standard error of the mean via fixed block averaging."""
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, np.nan
    bs = x.size // n_blocks
    if bs == 0:
        return float(x.mean()), 0.0
    means = [x[i * bs:(i + 1) * bs if i < n_blocks - 1 else x.size].mean()
             for i in range(n_blocks)]
    return float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def read_params(leaf):
    p = json.loads((leaf / "params.json").read_text())
    ens, pot = p.get("ensemble", {}), p.get("potential", {})
    return {
        "mu": float(ens["mu"]),
        "dim": int(ens["dimension"]),
        "box": float(ens.get("box", np.nan)),
        "potential": pot.get("main", "hard_core"),
        "T": float(pot["T"]) if pot.get("T") is not None else None,
    }


def read_run(leaf, burn_in_M, n_blocks):
    """Parse one leaf dir into a record of scalars, or None if unusable."""
    par = read_params(leaf)
    data = runlog.load(leaf)
    data = data[data["MC_cycle"] >= burn_in_M]
    if len(data) < 2:
        return None

    eta = runlog.packing_fraction(data, json.loads((leaf / "params.json").read_text()))
    add_acc, add_stp = data["insert_accepted"], data["insert_tried"]
    add_att = data.get("insert_evaluated", add_stp)
    rm_acc, rm_stp = data["delete_accepted"], data["delete_tried"]
    mv_acc, mv_stp = data["translate_accepted"], data["translate_tried"]
    has_wall = "wall_s" in data
    cycle = data["MC_cycle"].to_numpy()

    eta_mean, eta_err = block_average(eta, n_blocks)
    safe = lambda a, b: a.sum() / b.sum() if b.sum() > 0 else np.nan

    rec = dict(par)
    rec.update({
        "leaf": leaf.name,
        "eta": eta_mean, "eta_err": eta_err,
        "add_ratio": safe(add_acc, add_stp),
        "rem_ratio": safe(rm_acc, rm_stp),
        "mv_ratio": safe(mv_acc, mv_stp),
        "add_per_attempt": safe(add_acc, add_att),
        "total_cycles": float((cycle[-1] - cycle[0]) * 1e6),
        "wall_per_window": np.nan,
        "total_wall": np.nan,
    })
    if has_wall:
        w = data["wall_s"].to_numpy()
        rec["total_wall"] = float(w[-1] - w[0]) if WALL_S_IS_CUMULATIVE else float(w.sum())
        rec["wall_per_window"] = (rec["total_wall"] / (len(w) - 1)) if WALL_S_IS_CUMULATIVE \
            else float(w.mean())
    return rec


def find_leaves(path):
    path = Path(path)
    if (path / "run.log").exists():
        return [path]
    return sorted(d for d in path.iterdir() if d.is_dir() and (d / "run.log").exists())


def collect_group(path, label):
    recs = []
    for leaf in find_leaves(path):
        try:
            r = read_run(leaf, BURN_IN_M, N_BLOCKS)
            if r and MIN_MU <= r["mu"] <= MAX_MU:
                recs.append(r)
        except Exception as e:
            print(f"  ! {leaf.name}: {e}")
    recs.sort(key=lambda r: r["mu"])
    keys = {(r["dim"], r["potential"], r["T"]) for r in recs}
    if len(keys) > 1:
        warnings.warn(f"{label}: mixed (dim, potential, T) in one group: {keys}")
    return recs


def arr(recs, k):
    return np.array([r[k] for r in recs], float)


def plot_density_var(recs):
    """(values, ylabel) for the density axis -- rho* for 3D LJ, packing fraction otherwise."""
    eta = arr(recs, "eta")
    dim, pot = recs[0]["dim"], recs[0]["potential"]
    if pot == "lennard_jones" and dim == 3:
        return 6.0 * eta / np.pi, arr(recs, "eta_err") * 6.0 / np.pi, r"Reduced density $\rho^*$"
    return eta, arr(recs, "eta_err"), r"Packing fraction $\eta$"


def ref_density(dim, pot, T, mu_lo, mu_hi):
    """Reference density curves for the density plot. Returns list of (mu, rho, label, style)."""
    out = []
    if pot == "hard_core" and dim == 2:
        m = (HD2D_MU >= mu_lo) & (HD2D_MU <= mu_hi)
        out.append((HD2D_MU[m], HD2D_RHO[m], "std GCMC", dict(ls="--", marker="^", color="k")))
        d = (DOGE_MU >= mu_lo) & (DOGE_MU <= mu_hi)
        out.append((DOGE_MU[d], DOGE_RHO[d], "Döge 2004", dict(ls="--", marker="s", color="gray")))
    elif pot == "hard_core" and dim == 3:
        m = (HD3D_MU >= mu_lo) & (HD3D_MU <= mu_hi)
        out.append((HD3D_MU[m], HD3D_RHO[m], "std GCMC 3D (bulk)", dict(ls="--", marker="^", color="k")))

        d = (MERONI_MU >= mu_lo) & (MERONI_MU <= mu_hi)
        out.append((MERONI_MU[d], MERONI_RHO[d], "Meroni 1996 (QA)",
                    dict(zorder=5, ls="--", marker="s", color= "#FF9F43"
                         )))

    elif pot == "lennard_jones" and dim == 3 and T is not None:
        grid = np.linspace(mu_lo, mu_hi, 200)
        eos = lj_eos_rho(T, grid)
        if eos:
            for name, rho in eos.items():
                out.append((grid, rho, f"{name} (T*={T:g})", dict(ls=":", color="#555555")))
    return out


def std_baseline(dim, pot):
    """std-Metropolis (mu, add_ratio, rem_ratio, wall) for compare mode, or None."""
    if pot == "hard_core" and dim == 2:
        return HD2D_MU, HD2D_ADD, HD2D_REM, HD2D_TIME
    if pot == "hard_core" and dim == 3:
        return HD3D_MU, HD3D_ADD, HD3D_REM, HD3D_TIME
    return None


def interp_ref(mu, ref_mu, ref_val):
    out = np.interp(mu, ref_mu, ref_val, left=np.nan, right=np.nan)
    out[(mu < ref_mu.min()) | (mu > ref_mu.max())] = np.nan
    return out


def style_axes(ax):
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in", which="both")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figure_density(groups, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    seen, ylabel = set(), r"Density"
    for (label, recs), c in zip(groups, PALETTE):
        mu = arr(recs, "mu")
        rho, rho_err, ylabel = plot_density_var(recs)
        ax.errorbar(mu, rho, yerr=rho_err, fmt="o-", ms=6, capsize=3, color=c, label=label)
        key = (recs[0]["dim"], recs[0]["potential"], recs[0]["T"])
        if key not in seen:
            seen.add(key)
            for rmu, rrho, rlabel, st in ref_density(*key, mu.min(), mu.max()):
                ax.plot(rmu, rrho, label=rlabel, alpha=0.8, **st)
    ax.set_xlabel(r"Chemical potential  $\mu$", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.legend(fontsize=19)
    
    #style_axes(ax)
    
    # -- Tufte style --
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(-6, 15)
    ax.spines["left"].set_bounds(0.0, 0.4)
    ax.xaxis.set_ticks(
            [-6,-3,0,3,6,9,12,15]
            #[-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]
            )
    ax.yaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(direction="in", labelsize=26)

    fig.tight_layout()
    fig.savefig(out / "Packing_fraction_meroni.png", dpi=DPI)
    print(f"  wrote {out/'density.png'}")


def figure_acceptance(groups, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, recs), c in zip(groups, PALETTE):
        mu = arr(recs, "mu")
        ax.semilogy(mu, arr(recs, "add_ratio"), "^-", color=c, label=f"{label} · add")
        ax.semilogy(mu, arr(recs, "rem_ratio"), "s--", color=c, label=f"{label} · remove")
        ax.semilogy(mu, arr(recs, "add_per_attempt"), "x:", color=c, label=f"{label} · add/attempt")
    ax.set_xlabel(r"Chemical potential  $\mu^*$")
    ax.set_ylabel("Acceptance ratio")
    ax.legend(fontsize=8, ncol=max(1, len(groups)))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "acceptance.png", dpi=DPI)
    print(f"  wrote {out/'acceptance.png'}")


def figure_walltime(groups, out):
    if not any(np.isfinite(arr(recs, "wall_per_window")).any() for _, recs in groups):
        print("  (no wall_s column found -- skipping walltime.png)")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, recs), c in zip(groups, PALETTE):
        ax.plot(arr(recs, "mu"), arr(recs, "wall_per_window"), "o-", color=c, label=label)
    ax.set_xlabel(r"Chemical potential  $\mu^*$")
    ax.set_ylabel("Wall time per output window (s)")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "walltime.png", dpi=DPI)
    print(f"  wrote {out/'walltime.png'}")


def figure_cycles(groups, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, recs), c in zip(groups, PALETTE):
        ax.plot(arr(recs, "mu"), arr(recs, "total_cycles"), "o-", color=c, label=label)
    ax.set_xlabel(r"Chemical potential  $\mu^*$")
    ax.set_ylabel("MC cycles (post burn-in)")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "cycles.png", dpi=DPI)
    print(f"  wrote {out/'cycles.png'}")


def figure_relative(groups, out):
    """compare mode: efficiency as a multiple of the std-Metropolis baseline."""
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for (label, recs), c in zip(groups, PALETTE):
        base = std_baseline(recs[0]["dim"], recs[0]["potential"])
        if base is None:
            print(f"  (no std-Metropolis baseline for {label} -- placeholder, add arrays to compare)")
            continue
        bmu, badd, brem, btime = base
        mu = arr(recs, "mu")
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(mu, arr(recs, "add_ratio") / interp_ref(mu, bmu, badd),
                    "^-", color=c, label=f"{label} · add ratio")
            ax.plot(mu, arr(recs, "rem_ratio") / interp_ref(mu, bmu, brem),
                    "s--", color=c, label=f"{label} · remove ratio")
            sim_wall = arr(recs, "wall_per_window")
            if np.isfinite(sim_wall).any():
                ax.plot(mu, interp_ref(mu, bmu, btime) / sim_wall,
                        "o:", color=c, label=f"{label} · wall speedup")
        plotted = True
    ax.axhline(1.0, color="gray", ls="-", lw=1, alpha=0.6)
    ax.set_xlabel(r"Chemical potential  $\mu^*$")
    ax.set_ylabel(r"Multiple of std GCMC ($\times$)")
    if plotted:
        ax.legend(fontsize=8, ncol=max(1, len(groups)))
    else:
        ax.text(0.5, 0.5, "no std-Metropolis baseline for these potentials",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "relative.png", dpi=DPI)
    print(f"  wrote {out/'relative.png'}")


# ----------------------------------------------------------------------------
def main():
    out = Path(OUTDIR)
    out.mkdir(exist_ok=True)

    groups = []
    for path in RUN_GROUPS:
        label = LABEL_OVERRIDES.get(Path(path).name, Path(path).name)
        print(f"reading {path}")
        recs = collect_group("runs/"+path, label)
        if recs:
            groups.append((label, recs))
            print(f"  {len(recs)} runs, mu in [{recs[0]['mu']:g}, {recs[-1]['mu']:g}]")
        else:
            print("  (no usable runs)")

    if not groups:
        print("nothing to plot.")
        return

    print(f"\nmode = {MODE}")
    if MODE == "run":
        figure_density(groups, out)
        figure_acceptance(groups, out)
        figure_walltime(groups, out)
        figure_cycles(groups, out)

    elif MODE == "compare":
        figure_density(groups, out)
        figure_cycles(groups, out)
        figure_relative(groups, out)
    else:
        raise ValueError(f"MODE must be 'run' or 'compare', got {MODE!r}")
    print("done.")

    plt.show()

if __name__ == "__main__":
    main()
