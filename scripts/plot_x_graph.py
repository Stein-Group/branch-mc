import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json, re
from pathlib import Path
#import matplotlib.ticker as ticker

# Modern layout:
#   runs/<group>/<hash>__seed<SEED>_startmu_<AApBB>_hard_<d>d_box_<L>/
#       params.json, run.log
# mu and dimension come from params.json, the columns from the run.log header,
# seed and start mu from the leaf name. One invocation writes the 2D and the 3D
# figure.
RUNS_ROOT = "runs"

# group dir -> method. The method sets colour and legend label.
GROUPS = {
    "SR_2d": "sr",  "SR_aux_2d": "sr_aux",  "branch_2d": "branch",  "std_2d": "std",
    "SR_3d": "sr",  "SR_aux_3d": "sr_aux",  "branch_3d": "branch",  "std_3d": "std",
}

# ── Seed choice, per curve ────────────────────────────────────────────────────
# A curve is one group at one start mu. Lookup order:
#   (group, start_mu)  ->  group  ->  "*"
# Value: one seed, a list of seeds (first = run 1, drawn solid; second = run 2,
# drawn faded), or None for every seed. The script prints the seeds it finds
# for every curve, so run it once to see what is there.
SEED_SELECT = {
    "*": None,
    # "SR_3d": 987533,                 # this seed for every SR 3D curve
    # ("SR_3d", 13): 576347,           # ...except start mu 13
    # ("branch_2d", 11): [576347, 987533],
}

MAX_STEPS = 3_000_000  # plot up to this many MC cycles
ENGINE_SELECT = None  # e.g. {"metropolis_dynamics"}; None -> every engine

# ── Per-dimension settings ────────────────────────────────────────────────────
cfg = {
    2: dict(
        mu         = 10.0,
        # packing fraction of each starting configuration, keyed by START MU
        initial_pf = {1:0.31583478144089383, 2:0.37992327157412564, 8:0.6077934587145053, 9:0.6299940467998731, 10:0.6450736915371043,
                      11:0.6605722152948139, 12:0.677746255134438},
        yticks     = [0.6, 0.62, 0.64, 0.66, 0.68],
        title      = "Packing fraction convergence — 2D GCMC",
    ),
    3: dict(
        mu         = 12.0,
        initial_pf = {1:0.18535396656179778, 2:0.22095868330248208, 10:0.42359140945902374, 11:0.4356341812977846, 12:0.4356341812977846,
                      13:0.45762532987291316, 14:0.46809730538487915, 15:0.4822344723260332},
        yticks     = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47],
        title      = "Packing fraction convergence — 3D GCMC",
    ),
}

# ── Methods: colour and legend label ─────────────────────────────────────────
METHODS = [  # (key, colour, label template) -- legend order
    ("sr_aux", "#2166ac", "SR {d}D with aux-pot"),
    ("sr",     "#d6604d", "SR {d}D"),
    ("branch", "#1a9641", "BRANCH {d}D"),
    ("std",    "k",       "std GCMC"),
]
COLOR = {k: c for k, c, _ in METHODS}

alpha_map = {1: 1.0, 2: 0.45}
lw_map    = {1: 1.6, 2: 1.1}

LEAF_RE = re.compile(r"seed(\d+)_startmu_(-?\d+)p(\d+)")

# ── Readers (same conventions as gcmc_plot_glauber.py) ───────────────────────
LEGACY_POSITIONS = ["MC_cycle(M)", "pckng_frctn", "N", "add_acc", "add_att",
                    "add_stp", "rm_acc", "rm_stp", "mv_acc", "mv_stp", "wall_s"]


def resolve_engine(p):
    eng = p.get("engine")
    if isinstance(eng, dict):
        eng = eng.get("type")
    elif not isinstance(eng, str):
        eng = None
    if not eng:
        eng = (p.get("sampler") or {}).get("engine")
    return eng or "metropolis_dynamics"


def read_params(leaf):
    p = json.loads((leaf / "params.json").read_text())
    ens = p.get("ensemble", {})
    mu = ens.get("mu")
    if isinstance(mu, dict):          # typed state point: species A
        mu = next(iter(mu.values()))
    return {"mu": float(mu), "dim": int(ens["dimension"]),
            "engine": resolve_engine(p)}


def read_trace(path):
    """(cycles, eta) read BY COLUMN NAME from the run.log header; the old
    positional layout is used only for headerless logs."""
    names, x, y = None, [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if names is None:
                    names = line.lstrip("#").split()
                continue
            if not line.strip():
                continue
            if names is None:
                names = LEGACY_POSITIONS
            if "MC_cycle(M)" not in names or "pckng_frctn" not in names:
                raise ValueError(f"{path}: header has no MC_cycle(M)/pckng_frctn column")
            i_c, i_e = names.index("MC_cycle(M)"), names.index("pckng_frctn")
            cols = line.split()
            try:
                step = float(cols[i_c]) * 1e6
                if step > MAX_STEPS:
                    break
                x.append(step)
                y.append(float(cols[i_e]))
            except (ValueError, IndexError):
                continue
    return x, y


def find_leaves(path):
    path = Path(path)
    if (path / "run.log").exists():
        return [path]
    if not path.is_dir():
        return []
    return sorted(d for d in path.iterdir() if d.is_dir() and (d / "run.log").exists())


def fmt_mu(v):
    return f"{v:g}"


def seed_choice(group, start_mu):
    for key in ((group, start_mu), (group, int(round(start_mu))), group, "*"):
        if key in SEED_SELECT:
            v = SEED_SELECT[key]
            if v is None:
                return None
            return [v] if isinstance(v, int) else list(v)
    return None


def initial_point(start_mu, first_eta, table):
    """Start-config eta: looked up by start mu, else the nearest table entry."""
    k = int(round(start_mu))
    if start_mu == k and k in table:
        return table[k]
    if table:
        best = min(table.values(), key=lambda v: abs(v - first_eta))
        if abs(best - first_eta) <= 0.01:
            return best
    return None


# ── Collect: curves[(dim)][(group, start_mu)] = {seed: (leaf, x, y)} ─────────
curves = {2: {}, 3: {}}
for group, method in GROUPS.items():
    leaves = find_leaves(Path(RUNS_ROOT) / group)
    if not leaves:
        print(f"  ! no runs found in {Path(RUNS_ROOT) / group}")
    for leaf in leaves:
        m = LEAF_RE.search(leaf.name)
        if not m:
            print(f"  ! {leaf.name}: no seed/startmu in the name -- skipped")
            continue
        seed = int(m.group(1))
        start_mu = float(f"{m.group(2)}.{m.group(3)}")
        try:
            par = read_params(leaf)
        except Exception as e:
            print(f"  ! {leaf}: {e}")
            continue
        d = par["dim"]
        if d not in cfg or abs(par["mu"] - cfg[d]["mu"]) > 1e-9:
            continue
        if ENGINE_SELECT is not None and par["engine"] not in ENGINE_SELECT:
            continue
        x, y = read_trace(leaf / "run.log")
        if not x:
            print(f"  ! {leaf.name}: empty run.log -- skipped")
            continue
        curves[d].setdefault((group, start_mu), {})[seed] = (leaf, x, y)


# ── One figure per dimension ─────────────────────────────────────────────────
def plot_dim(dim, dim_curves):
    c = cfg[dim]
    print(f"\n{dim}D, mu = {fmt_mu(c['mu'])}")

    prepared = []
    for (group, start_mu) in sorted(dim_curves, key=lambda k: (list(GROUPS).index(k[0]), k[1])):
        found = dim_curves[(group, start_mu)]
        want = seed_choice(group, start_mu)
        seeds = sorted(found) if want is None else [s for s in want if s in found]
        missing = [] if want is None else [s for s in want if s not in found]
        print(f"  {group:<10} start mu {fmt_mu(start_mu):>5}: seeds {sorted(found)}"
              f"  -> plotting {seeds}"
              + (f"   (! not found: {missing})" if missing else ""))
        for run, seed in enumerate(seeds, start=1):
            leaf, x, y = found[seed]
            init = initial_point(start_mu, y[0], c["initial_pf"])
            if init is not None:
                x, y = [0.0] + x, [init] + y
            else:
                print(f"    ! no initial_pf entry for start mu {fmt_mu(start_mu)} "
                      f"-- no cycle-0 point drawn")
            prepared.append((GROUPS[group], run, x, y))

    if not prepared:
        print("  (nothing selected)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for method, run, x, y in prepared:
        ax.plot(
            x, y,
            color=COLOR[method],
            ls="-",
            alpha=alpha_map.get(run, 1.0),
            lw=lw_map.get(run, 1.4),
        )

    present = {m for m, *_ in prepared}
    color_proxies = [
        mlines.Line2D([], [], color=col, lw=2.0, label=lab.format(d=dim))
        for key, col, lab in METHODS if key in present
    ]
    ax.legend(handles=color_proxies, fontsize=19, framealpha=0.85)
    ax.set_xlabel("Monte Carlo cycles", fontsize=26)
    ax.set_ylabel(r"Packing fraction $\eta$", fontsize=26)
    #ax.set_title(c["title"], fontsize=13)

    ax.spines[["top", "right"]].set_visible(False)

    # View limits pinned to what the reference figure had (autoscale of curves
    # spanning the full 0..1e6 cycles, 5 % margin), so shorter runs do not
    # push the curves onto the y spine.
    ax.set_xlim(-0.05 * MAX_STEPS, 1.05 * MAX_STEPS)
    ally = [v for *_, y in prepared for v in y]
    span = max(ally) - min(ally)
    yt = c["yticks"]
    ax.set_ylim(min(yt[0], min(ally) - 0.05 * span),
                max(yt[-1], max(ally) + 0.05 * span))

    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.yaxis.set_ticks(yt)

    #ax.xaxis.set_ticks([0, 0.2e6, 0.4e6, 0.6e6, 0.8e6, 1.0e6, 1.2e6, 1.4e6, 1.6e6, 1.8e6, 2.0e6, 2.2e6, 2.4e6, 2.6e6, 2.8e6, 3.0e6])
    ax.xaxis.set_ticks([0, 0.3e6, 0.6e6, 0.9e6, 1.2e6, 1.5e6, 1.8e6, 2.1e6, 2.4e6, 2.7e6, 3.0e6])

    ax.spines["bottom"].set_bounds(0.0e6, 3.0e6)
    ax.xaxis.get_offset_text().set_fontsize(26)
    ax.tick_params(direction="in", labelsize=26)
    
    #ax.grid(True, alpha=0.25, lw=0.6)
    plt.tight_layout()
    name = f"x_plot_mu_{fmt_mu(c['mu'])}_{dim}d.png"
    plt.savefig(name, dpi=300, bbox_inches="tight")
    print(f"  wrote {name}  ({len(prepared)} curves)")


for dim in (2, 3):
    if curves[dim]:
        plot_dim(dim, curves[dim])
    else:
        print(f"\n(no {dim}D runs at mu={fmt_mu(cfg[dim]['mu'])})")

plt.show()
