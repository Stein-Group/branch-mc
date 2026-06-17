"""Configuration: one place that turns a params.json into a frozen Config and
derives every mode-dependent constant the rest of the code needs.

params.json has five blocks, one per axis. Inactive keys cannot appear (an LJ
temperature on a hard-core run, a pmf block under branch), so nothing leaks.
Validation is a per-section check plus the compatibility rules below, each with
a specific message.

Schema
------
{
  "potential":  {"main": "hard_core" | "lennard_jones",
                 "T": float},                       # T required iff main = lennard_jones

  "auxiliary":  {"type": "pmf",                      # OPTIONAL block; omit = no aux
                 "g3_csv": str,                      # required iff present and 3D
                 "eta": float, "gamma_max": float, "delta_eta": float},

  "sampler":    {"type": "metropolis" | "rosenbluth" | "branch" | "reweight",
                 "depth": int, "use_depth": bool,    # forbidden for metropolis
                 "trials": int | "trial_density": float,   # forbidden for metropolis
                 "trial_box_side": float},                 # forbidden for metropolis

  "ensemble":   {"mu": float | "sweep": [float],     # exactly one; sweep is launcher-only
                 "dimension": 2 | 3, "box": float},

  "run":        {"kind": "single" | "parallel", "n_workers": int,
                 "sampling_window": int, "target_cycles": float, "burn_in": int,
                 "traj_every": int, "checkpoint_every": int, "seed": int,
                 "initial_config": str,
                 "radius": float, "move_factor": float,
                 "upper_cutoff": float, "lower_cutoff": float, "no_neigh": int,
                 "cell_length": float, "buffer": float, "x_spacing": float, "y_spacing": float}
}

trial_box_side is the full side of the trial box (twice the legacy cavity radius);
the trial-count relation uses trial_box_side^2 as the span.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


def default_side(two_dim, parallel):
    return {(True, True): 200.0, (True, False): 75.0,
            (False, True): 50.0, (False, False): 20.0}[(two_dim, parallel)]


_GAMMA_AUX_2D = {1: 0.193548, 2: 0.049948, 3: 0.022557, 4: 0.010187,
                 5: 0.007558, 6: 0.002438, 7: 0.001180, 8: 0.000495, 9: 0.000176}
_GAMMA_AUX_3D = {1: 0.2903, 2: 0.1030, 3: 0.0332, 4: 0.0193,
                 5: 0.0037, 6: 0.0042, 7: 0.0, 8: 0.0, 9: 0.0}


def _gamma_dict(pmf, reweight, two_dim):
    if pmf:
        return dict(_GAMMA_AUX_2D if two_dim else _GAMMA_AUX_3D)
    if reweight:
        return {i: 1.0 for i in range(1, 10)}
    return {i: 0.0 for i in range(1, 10)}


_MAIN = {"hard_core", "lennard_jones"}
_SAMPLERS = {"metropolis", "rosenbluth", "branch", "reweight"}


def _check(section, allowed, required=()):
    extra = set(section) - set(allowed)
    if extra:
        raise ValueError(f"unknown keys {sorted(extra)}; allowed {sorted(allowed)}")
    missing = set(required) - set(section)
    if missing:
        raise ValueError(f"missing required keys {sorted(missing)}")


@dataclass(frozen=True)
class Config:
    # identity-bearing
    potential_type: str
    sampler: str
    dimension: int
    chem_pot: float
    ensemble_type: str
    target_n: int | None
    box: float
    cavity_radius: float
    trial_box_side: float
    trial_density: float
    depth: int
    branch: bool
    metropolis: bool
    use_depth: bool
    move_factor: float
    temp: float
    rc: float | None
    eta: float | None
    gamma_max: float | None
    delta_eta: float | None
    g3_csv: str | None
    # derived
    two_dim: bool
    parallel: bool
    beta: float
    fugacity: float
    radius: float
    diameter: float
    r_squared: float
    no_of_trial_molecules: int
    no_neigh: int
    upper_cutoff: float
    lower_cutoff: float
    volume_of_molecule: float
    total_volume: float
    lj_tail_pref: float | None
    gamma_dict: dict
    # execution (not identity-bearing)
    mode: str
    n_workers: int
    cell_length: float
    buffer: float
    x_spacing: float | None
    y_spacing: float | None
    sampling_window: int
    target_cycles: float
    burn_in: int
    traj_every: int
    checkpoint_every: int
    initial_config: str | None
    # quenched-annealed matrix: frozen obstacle positions the fluid cannot
    # overlap but which are never inserted/deleted/moved/counted. Default empty
    # => ordinary bulk run, bit-identical to before. matrix_config is the file
    # path (loaded by the runner); matrix_positions holds the loaded coordinates.
    matrix_config: str | None = None
    matrix_positions: object = None   # np.ndarray (n_matrix, dim) or None
    # muVT target-mu control: drive mu so the running-mean N hits a target.
    # target_count is the target <N> (from target_N / target_eta / target_density);
    # None => ordinary fixed-mu muVT. The controller is Robbins-Monro on log-N.
    target_count: float | None = None
    control_gain: float = 0.5
    control_gain_tau: float = 20.0
    control_window: int = 20
    sweep_label: str | None = None       # tag from the launcher: the swept axis
    # confinement (ensemble_type == "confined"): hard slit or cylinder pore
    pore_geometry: str | None = None     # "slit" | "cylinder" | None
    pore_H: float | None = None          # slit gap (z), wall surfaces at 0 and H
    pore_R: float | None = None          # cylinder radius (axis along z, at centre)
    pore_axis_len: float | None = None   # cylinder axial length (periodic z)

    @property
    def box_dims(self):
        return [self.box] * self.dimension

    def build_potential(self):
        from .potentials import make_potential
        t, uc, d = self.potential_type, self.upper_cutoff, self.diameter
        if t == "hard_sphere":
            return make_potential("hard_sphere", diameter=d, upper_cutoff=uc)
        if t in ("lj", "ljbranch"):
            return make_potential("lj", beta=self.beta, upper_cutoff=uc,
                                   diameter=d, volume=self.total_volume)
        if t == "ljaux":
            return make_potential("ljaux3d", beta=self.beta, upper_cutoff=uc,
                                  diameter=d, side=self.box,
                                  n_trials=self.no_of_trial_molecules,
                                  g3_csv=self.g3_csv)
        if t == "aux":
            if self.two_dim:
                return make_potential("aux2d", diameter=d, upper_cutoff=uc,
                                      eta=self.eta, sigma=1.0)
            return make_potential("aux3d", diameter=d, upper_cutoff=uc,
                                  side=self.box, n_trials=self.no_of_trial_molecules,
                                  g3_csv=self.g3_csv)
        if t == "reweight":
            return None
        raise ValueError(f"unknown potential type {t!r}")


def from_params(p):
    _check(p, {"potential", "auxiliary", "sampler", "ensemble", "run"},
           {"potential", "sampler", "ensemble"})

    # ---- main potential ----
    pot = p["potential"]
    _check(pot, {"main", "T", "tail"}, {"main"})
    main = pot["main"]
    if main not in _MAIN:
        raise ValueError(f"unknown main potential {main!r}; allowed {sorted(_MAIN)}")
    lj = main == "lennard_jones"
    if lj and "T" not in pot:
        raise ValueError("Lennard-Jones requires potential.T")
    if not lj and "T" in pot:
        raise ValueError("only Lennard-Jones takes potential.T")

    # ---- sampler ----
    smp = p["sampler"]
    _check(smp, {"type", "depth", "use_depth", "trials", "trial_density", "trial_box_side"},
           {"type"})
    sampler = smp["type"]
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; allowed {sorted(_SAMPLERS)}")
    metropolis = sampler == "metropolis"
    branch = sampler == "branch"
    reweight = sampler == "reweight"

    # ---- ensemble ----
    ens = p["ensemble"]
    _check(ens, {"mu", "sweep", "dimension", "box", "type",
                 "N", "packing_fraction",
                 "target_eta", "target_density", "target_N", "pore"}, {"dimension"})
    dim = int(ens["dimension"])
    target_count = None   # set below iff a muVT target is requested
    _target_spec = None   # (kind, value) for a muVT target, finalized post-volume
    _pore_geometry = _pore_H = _pore_R = _pore_axis_len = None
    if dim not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    two_dim = dim == 2
    ens_type = ens.get("type", "muVT")
    if ens_type not in ("muVT", "NVT", "QA", "confined"):
        raise ValueError("ensemble.type must be 'muVT', 'NVT', 'QA', or 'confined'")
    nvt = ens_type == "NVT"
    qa = ens_type == "QA"
    confined = ens_type == "confined"
    if confined:
        # hard slit or cylinder pore. muVT physics (give mu, or a target) plus a
        # one-body hard-wall constraint. The pore is the only external field;
        # rejects a matrix (no fluid-in-matrix-in-pore here). 3D only.
        if "pore" not in ens:
            raise ValueError("ensemble.type='confined' requires an ensemble.pore block")
        if two_dim:
            raise ValueError("confined pores are 3D only")
        if "matrix_config" in p.get("run", {}):
            raise ValueError("a pore and a QA matrix cannot be combined")
        pore = ens["pore"]
        _pore_geometry = pore.get("geometry")
        if _pore_geometry == "slit":
            _pore_H = float(pore["H"])           # gap in z; walls at 0 and H
            if _pore_H <= 1.0:
                raise ValueError("slit gap H must exceed the sphere diameter (1)")
        elif _pore_geometry == "cylinder":
            _pore_R = float(pore["R"])           # radius; axis along z at centre
            _pore_axis_len = float(pore.get("L_z", ens["box"]))
            if 2 * _pore_R <= 1.0:
                raise ValueError("cylinder diameter 2R must exceed the sphere diameter (1)")
        else:
            raise ValueError("ensemble.pore.geometry must be 'slit' or 'cylinder'")
        # the wall lives inside the cubic periodic cell of side `box`; require a
        # forbidden margin >= 1 sigma so periodic images never interact across
        # the wall (then the existing cubic-periodic neighbour tree is exact and
        # no per-axis box change is needed).
        L = float(ens["box"])
        if _pore_geometry == "slit" and _pore_H > L - 1.0:
            raise ValueError(f"slit gap H={_pore_H} must be <= box-1={L-1} so the "
                             f"forbidden margin separates periodic z-images")
        if _pore_geometry == "cylinder" and 2 * _pore_R > L - 1.0:
            raise ValueError(f"cylinder diameter 2R={2*_pore_R} must be <= box-1={L-1} "
                             f"so the forbidden margin separates periodic x,y-images")
        _pore_axis_len = L     # cubic cell: cylinder axial period = box
    if "pore" in ens and not confined:
        raise ValueError("ensemble.pore is only valid with ensemble.type='confined'")

    if nvt:
        # NVT: fixed particle number, no chemical potential. Target given as an
        # exact integer N, or as a packing fraction converted to the nearest
        # integer N (held exactly thereafter). mu is meaningless here.
        if "mu" in ens or "sweep" in ens:
            raise ValueError("NVT takes no 'mu'/'sweep' (N is fixed, mu undefined)")
        has_n, has_pf = "N" in ens, "packing_fraction" in ens
        if has_n == has_pf:
            raise ValueError("NVT needs exactly one of 'N' or 'packing_fraction'")
        _swept = next((k for k in ("N", "packing_fraction")
                       if isinstance(ens.get(k), list)), None)
        if _swept is not None:
            raise ValueError(
                f"ensemble.{_swept} is a list (a sweep), but a single run got it "
                f"unexpanded. Launch sweeps with `python -m gcmc.launcher` (not "
                f"`gcmc run`); if you already did, the launcher on this machine "
                f"is an older version that does not expand '{_swept}' -- update "
                f"gcmc/launcher.py.")
        mu = 0.0   # unused under NVT; the prefactor cancels in repacking
    else:
        # muVT: either a fixed `mu` (or a sweep), OR a target the controller
        # drives mu toward: exactly one of target_eta / target_density / target_N.
        if "N" in ens or "packing_fraction" in ens:
            raise ValueError("'N'/'packing_fraction' are NVT-only; set ensemble.type='NVT'")
        target_keys = [k for k in ("target_eta", "target_density", "target_N") if k in ens]
        if len(target_keys) > 1:
            raise ValueError(f"give only one target, got {target_keys}")
        targeting = len(target_keys) == 1
        has_mu, has_sweep = "mu" in ens, "sweep" in ens

        if targeting:
            if has_sweep:
                raise ValueError("a target and a 'sweep' are mutually exclusive")
            for k in ("target_eta", "target_density", "target_N"):
                if isinstance(ens.get(k), list):
                    raise ValueError(f"ensemble.{k} must be a single value, not a list")
            # record the raw target; convert to a particle count <N> after the
            # box volume is known (computed further below). Starting mu is set
            # there too (explicit `mu` seed, or the ideal-gas guess).
            _target_spec = target_keys[0], float(ens[target_keys[0]])
            mu = float(ens["mu"]) if has_mu else None   # finalized after volume
        else:
            if has_mu and has_sweep:
                raise ValueError("ensemble takes either 'mu' or 'sweep', not both")
            if not has_mu and not has_sweep:
                raise ValueError("ensemble needs 'mu', a target "
                                 "(target_eta/target_density/target_N), or 'sweep'")
            if isinstance(ens.get("mu"), list):
                raise ValueError(
                    "ensemble.mu is a list (a sweep) but a single run got it "
                    "unexpanded. Launch sweeps with `python -m gcmc.launcher`; if "
                    "you did, gcmc/launcher.py on this machine is outdated.")
            if has_sweep:
                raise ValueError("this config has a 'sweep'; launch it with gcmc.launcher, "
                                 "which expands it into one run per mu")
            mu = float(ens["mu"])

    # ---- auxiliary ----
    aux_block = p.get("auxiliary")
    pmf = aux_block is not None
    if pmf:
        _check(aux_block, {"type", "g3_csv", "eta", "gamma_max", "delta_eta"}, {"type"})
        if aux_block["type"] != "pmf":
            raise ValueError("the only auxiliary potential is 'pmf'")

    # ---- run ----
    run = p.get("run", {})
    _check(run, {"kind", "n_workers", "n_cores", "sampling_window", "target_cycles", "burn_in",
                 "traj_every", "checkpoint_every", "seed", "initial_config",
                 "matrix_config",
                 "radius", "move_factor", "upper_cutoff", "lower_cutoff", "no_neigh",
                 "cell_length", "buffer", "x_spacing", "y_spacing",
                 "control_gain", "control_gain_tau", "control_window",
                 "sweep_label", "matrix_config"})
    if int(run.get("traj_every", 0)) < 0:
        raise ValueError("run.traj_every must be >= 0 (0 = save only the last "
                         "frame; N >= 1 = append every N windows)")
    kind = run.get("kind", "single")
    if kind not in ("single", "parallel"):
        raise ValueError("run.kind must be 'single' or 'parallel'")
    parallel = kind == "parallel"

    # ---- compatibility table ----
    if lj and two_dim:
        raise ValueError("Lennard-Jones is 3D only")
    if reweight and lj:
        raise ValueError("reweight requires a hard_core main potential")
    if parallel and reweight:
        raise ValueError("parallel does not wire the reweight (psi6) mode")
    if reweight and not two_dim:
        raise ValueError("reweight (psi6) is a 2D method; use dimension 2")
    if pmf and lj and two_dim:
        raise ValueError("the Lennard-Jones + pmf auxiliary mode is 3D only")
    if pmf and lj and "g3_csv" not in aux_block:
        raise ValueError("Lennard-Jones + pmf requires auxiliary.g3_csv")
    if pmf and branch:
        raise ValueError("branch takes no auxiliary potential")
    if pmf and reweight:
        raise ValueError("reweight uses psi6, not a pmf auxiliary; remove the auxiliary block")
    if not lj and mu is not None and mu < 0 and not qa and not confined and _target_spec is None:
        # bulk hard-core fluids want mu >= 0; but a quenched-annealed run (frozen
        # matrix) lives at low fluid density where mu is naturally negative -- the
        # entire dilute branch of the adsorption isotherm. Allow it there.
        raise ValueError("non-LJ runs require mu >= 0 (except quenched-annealed "
                         "matrix runs, where the dilute isotherm has mu < 0)")
    if pmf and not two_dim and "g3_csv" not in aux_block:
        raise ValueError("3D pmf auxiliary requires auxiliary.g3_csv")
    if parallel:
        buffer_v = float(run.get("buffer", 2.0))
        reach = (float(pot["rc"]) if (lj and "rc" in pot) else (2.5 if lj else None))
        if reach is None:
            reach = 2.0 * float(ens.get("radius", 0.5))
        if buffer_v < reach - 1e-9:
            raise ValueError(
                f"parallel buffer ({buffer_v}) must be >= the interaction "
                f"range ({reach}): border rings must contain every molecule "
                f"that can interact with a live one")
    if int(smp.get("depth", 0)) > 3:
        raise ValueError("depth is validated for 0-3 only")
    if int(smp.get("depth", 0)) > 0 and parallel:
        import warnings
        cl = float(run.get("cell_length", 0.0))
        cav_w = 2.0 * (float(smp.get("trial_box_side", 3.0)) / 2.0
                       if "trial_box_side" in smp
                       else float(smp.get("cavity_radius", 1.5)))
        warnings.warn(
            f"parallel depth >= 1 is approximate: the depth scheme satisfies "
            f"global balance, not detailed balance, so a cell decomposition "
            f"carries a residual density bias controlled by the cavity/cell "
            f"ratio. Measured on the hard-sphere EOS: +19% at cell_length = "
            f"2.7 cavity widths, +1.3% at 6.7, +0.9% at 10. Your cell_length "
            f"is {cl / cav_w:.1f} cavity widths; use >= 20 to push the bias "
            f"below the half-percent level, or serial mode, which is exact "
            f"at all depths.",
            UserWarning)
    if metropolis:
        forbidden = {"depth", "trials", "trial_density", "trial_box_side"} & set(smp)
        if forbidden:
            raise ValueError("metropolis is standard GCMC: it takes no "
                             "depth, trials, trial_density, or trial_box_side")
    if "trials" in smp and "trial_density" in smp:
        raise ValueError("give either sampler.trials or sampler.trial_density, not both")

    # ---- derived constants ----
    temp = float(pot["T"]) if lj else 1.0
    beta = 1.0 / temp
    box = float(ens["box"]) if "box" in ens else default_side(two_dim, parallel)

    radius = float(run.get("radius", 0.5))
    diameter = 2.0 * radius                       # physical particle size / LJ sigma
    disk = math.pi * radius * radius
    # the single exclusion floor: hard cores exclude at the diameter, LJ at 0.85
    # (below which the Boltzmann weight is negligible but the clique graph needs a
    # hard edge). r_squared is just its square, used by the Rosenbluth mask.
    _lower_default = 0.9 if (lj and branch) else (0.85 if lj else diameter)
    lower_cutoff = float(run.get("lower_cutoff", _lower_default))
    r_squared = lower_cutoff * lower_cutoff

    if metropolis:
        trial_box_side = 0.0
        cav = 0.0
        n_trials = 1
        trial_density = 0.0
    else:
        trial_box_side = float(smp.get("trial_box_side", 3.0))
        cav = trial_box_side / 2.0
        span = trial_box_side ** 2
        if "trial_density" in smp:
            trial_density = float(smp["trial_density"])
            n_trials = int((trial_density * span) / disk)
        else:
            n_trials = int(smp.get("trials", 30))
            trial_density = round((n_trials * disk) / span, 3)

    no_neigh = int(run.get("no_neigh", 40 if two_dim else 75))
    if "upper_cutoff" in run:
        uc = float(run["upper_cutoff"])
    elif reweight:
        uc = 3.0 * radius
    elif lj:
        uc = 2.5
    else:
        uc = 3.7545 if two_dim else 2.0

    if two_dim:
        vol_mol = math.pi * radius ** 2
        total_vol = box ** 2
    else:
        vol_mol = (4.0 / 3.0) * math.pi * radius ** 3
        total_vol = box ** 3

    # NVT target as an exact integer N
    target_n = None
    if nvt:
        if "N" in ens:
            target_n = int(ens["N"])
            if target_n < 0:
                raise ValueError("ensemble.N must be >= 0")
        else:
            pf = float(ens["packing_fraction"])
            if not 0.0 <= pf < 1.0:
                raise ValueError("packing_fraction must be in [0, 1)")
            target_n = int(round(pf * total_vol / vol_mol))

    # muVT target control: now that the box volume is known, convert the target
    # to a particle count <N>, and finalize the starting mu (ideal-gas guess if
    # the user gave no explicit seed).
    if _target_spec is not None:
        kind, val = _target_spec
        if kind == "target_N":
            target_count = val
        elif kind == "target_eta":
            target_count = val * total_vol / vol_mol
        else:  # target_density (rho* = N / V)
            target_count = val * total_vol
        if target_count <= 0:
            raise ValueError("the target must correspond to N > 0")
        if mu is None:
            mu = math.log(target_count / total_vol)   # ideal-gas cold start

    lj_tail_pref = None
    lj_tail_on = bool(pot.get("tail", True))
    if "tail" in pot and not lj:
        raise ValueError("potential.tail applies to lennard_jones only")
    if lj:
        x = uc ** -9 / 3.0 - uc ** -3
        lj_tail_pref = ((8.0 / 3.0) * math.pi / total_vol * x) if lj_tail_on else 0.0

    if reweight:
        ptype = "reweight"
    elif pmf and lj:
        ptype = "ljaux"
    elif lj and branch:
        # true LJ + branch: k-clique enumeration with full Rosenbluth weighting
        # under the real LJ energy (no hard-core auxiliary). Distinct algorithm
        # and acceptance from the rosenbluth/metropolis lj path, so it gets its
        # own potential_type and a tighter default exclusion floor.
        ptype = "ljbranch"
    elif lj:
        ptype = "lj"
    elif pmf:
        ptype = "aux"
    else:
        ptype = "hard_sphere"

    has_matrix_cfg = run.get("matrix_config") is not None
    if qa and not has_matrix_cfg:
        raise ValueError(
            "ensemble.type='QA' (quenched-annealed) requires run.matrix_config: "
            "an absolute path to the frozen matrix configuration (an NVT "
            "checkpoint.npz, .extxyz, or plain x y z file).")
    if has_matrix_cfg and not qa:
        raise ValueError(
            "run.matrix_config is only valid with ensemble.type='QA'. A frozen "
            "matrix defines a quenched-annealed run; set ensemble.type='QA' "
            "(it is grand-canonical, so give a chemical potential mu, not N).")
    if qa:
        # QA support is implemented only on the deep serial rosenbluth/branch
        # path. Reject every combination that would silently ignore the matrix
        # and produce fluid overlapping the frozen obstacles with no error.
        if ptype not in ("hard_sphere", "ljbranch"):
            raise ValueError(
                f"QA runs support hard-core and LJ+branch potentials only, not "
                f"{ptype!r}; the other modes pass kd-tree neighbour indices "
                f"downstream where frozen-matrix indices would be misread as "
                f"fluid particles")
        if metropolis:
            raise ValueError(
                "QA runs are not supported with the 'metropolis' sampler: its "
                "insertion/translation overlap test is built on the fluid only "
                "and would let fluid overlap the frozen matrix. Use 'rosenbluth' "
                "or 'branch'.")
        if kind == "parallel":
            raise ValueError(
                "QA runs are not supported with run.kind='parallel': the domain "
                "decomposition carries only the fluid in each cell and would "
                "drop the frozen matrix. Use run.kind='single'.")
    return Config(
        potential_type=ptype, sampler=sampler, dimension=dim, chem_pot=mu,
        ensemble_type=ens_type, target_n=target_n, box=box,
        target_count=target_count,
        control_gain=float(run.get("control_gain", 0.5)),
        control_gain_tau=float(run.get("control_gain_tau", 20.0)),
        control_window=int(run.get("control_window", 20)),
        cavity_radius=cav, trial_box_side=trial_box_side, trial_density=trial_density,
        depth=(0 if metropolis else int(smp.get("depth", 0))),
        branch=branch, metropolis=metropolis,
        use_depth=bool(smp.get("use_depth", True)),
        move_factor=float(run.get("move_factor", 0.25)),
        temp=temp, rc=(uc if lj else None),
        eta=(float(aux_block.get("eta", 0.69)) if pmf and two_dim else None),
        gamma_max=(float(aux_block["gamma_max"]) if pmf and "gamma_max" in aux_block else None),
        delta_eta=(float(aux_block["delta_eta"]) if pmf and "delta_eta" in aux_block else None),
        g3_csv=(aux_block.get("g3_csv") if pmf and not two_dim else None),
        two_dim=two_dim, parallel=parallel, beta=beta,
        fugacity=math.exp(beta * mu), radius=radius, diameter=diameter,
        r_squared=r_squared, no_of_trial_molecules=n_trials, no_neigh=no_neigh,
        upper_cutoff=uc, lower_cutoff=lower_cutoff,
        volume_of_molecule=vol_mol, total_volume=total_vol,
        lj_tail_pref=lj_tail_pref, gamma_dict=_gamma_dict(pmf, reweight, two_dim),
        mode=kind, n_workers=int(run.get("n_workers", 1)),
        cell_length=float(run.get("cell_length", 30.0)),
        buffer=float(run.get("buffer", 2.0)),
        x_spacing=(float(run["x_spacing"]) if "x_spacing" in run else None),
        y_spacing=(float(run["y_spacing"]) if "y_spacing" in run else None),
        sampling_window=int(run.get("sampling_window", 1000)),
        target_cycles=float(run.get("target_cycles", 1.0025e9)),
        burn_in=int(run.get("burn_in", 0)),
        traj_every=int(run.get("traj_every", 0)),
        checkpoint_every=int(run.get("checkpoint_every", 1)),
        initial_config=run.get("initial_config"),
        matrix_config=run.get("matrix_config"),
        sweep_label=run.get("sweep_label"),
        pore_geometry=_pore_geometry, pore_H=_pore_H, pore_R=_pore_R,
        pore_axis_len=_pore_axis_len,
    )
