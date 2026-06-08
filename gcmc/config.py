"""Configuration: one place that turns a params.json into a frozen Config and
derives every mode-dependent constant the rest of the code needs.

params.json is a tagged structure. The `potential.type` field selects the
physics and only that type's keys are read, so inactive parameters (an aux eta
on an LJ run, say) simply do not exist and cannot leak anywhere. Validation is a
small per-section check with clear errors; no third-party dependency.

The derived constants (box, cutoffs, neighbour count, volumes, tail prefactor,
gamma schedule, trial count) are computed here with exactly the original's
conditionals, so every mode lands on the numbers it lands on today. The Config
also knows how to build its potential, which is the single wiring point between
params and the physics modules.

Schema
------
{
  "potential": {"type": "hard_sphere" | "lj" | "aux" | "reweight",
                # lj:   "T" (required), "rc" (default 2.5)
                # aux:  "eta" (default 0.69 in 2D), "g3_csv" (required in 3D),
                #       "gamma_max", "delta_eta" (identity only, optional)
               },
  "ensemble":  {"mu": float, "dimension": 2 | 3, "box": float (optional)},
  "trials":    {"cavity_radius": float, "trial_density": float | "k": int},
  "algorithm": {"depth": int, "branch": bool, "use_depth": bool},
  "execution": {"mode": "serial" | "parallel", "n_workers": int,
                "sampling_window": int, "target_cycles": float, "burn_in": int,
                "traj_every": int, "checkpoint_every": int,
                "full_trajectory": bool},
  "move":      {"move_factor": float}
}
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math


# preset box sizes, keyed by (two_dim, parallel), matching the original
def default_side(two_dim, parallel):
    return {(True, True): 200.0, (True, False): 75.0,
            (False, True): 50.0, (False, False): 20.0}[(two_dim, parallel)]


_GAMMA_AUX_2D = {1: 0.193548, 2: 0.049948, 3: 0.022557, 4: 0.010187,
                 5: 0.007558, 6: 0.002438, 7: 0.001180, 8: 0.000495, 9: 0.000176}
_GAMMA_AUX_3D = {1: 0.2903, 2: 0.1030, 3: 0.0332, 4: 0.0193,
                 5: 0.0037, 6: 0.0042, 7: 0.0, 8: 0.0, 9: 0.0}


def _gamma_dict(aux, reweight, two_dim):
    if aux:
        return dict(_GAMMA_AUX_2D if two_dim else _GAMMA_AUX_3D)
    if reweight:
        return {i: 1.0 for i in range(1, 10)}
    return {i: 0.0 for i in range(1, 10)}


_POTENTIAL_KEYS = {
    "hard_sphere": (set(), set()),
    "lj": ({"T"}, {"rc"}),
    "aux": (set(), {"eta", "g3_csv", "gamma_max", "delta_eta"}),
    "reweight": (set(), set()),
}


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
    dimension: int
    chem_pot: float
    box: float
    cavity_radius: float
    trial_density: float
    depth: int
    branch: bool
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
    volume_of_molecule: float
    total_volume: float
    lj_tail_pref: float | None
    gamma_dict: dict
    # execution (not identity-bearing)
    mode: str
    n_workers: int
    cell_length: float
    sampling_window: int
    target_cycles: float
    burn_in: int
    traj_every: int
    checkpoint_every: int
    full_trajectory: bool
    initial_config: str | None

    @property
    def box_dims(self):
        return [self.box] * self.dimension

    def build_potential(self):
        from .potentials import make_potential
        t, uc, d = self.potential_type, self.upper_cutoff, self.diameter
        if t == "hard_sphere":
            return make_potential("hard_sphere", diameter=d, upper_cutoff=uc)
        if t == "lj":
            return make_potential("lj", beta=self.beta, upper_cutoff=uc,
                                   diameter=d, volume=self.total_volume)
        if t == "aux":
            if self.two_dim:
                return make_potential("aux2d", diameter=d, upper_cutoff=uc,
                                      eta=self.eta, sigma=1.0)
            return make_potential("aux3d", diameter=d, upper_cutoff=uc,
                                  side=self.box, n_trials=self.no_of_trial_molecules,
                                  g3_csv=self.g3_csv)
        if t == "reweight":
            return None  # weights come from the reweighting sampler, not a potential
        raise ValueError(f"unknown potential type {t!r}")


def from_params(p):
    _check(p, {"potential", "ensemble", "trials", "algorithm", "execution", "move"},
           {"potential", "ensemble", "trials"})

    pot = p["potential"]
    ptype = pot["type"]
    if ptype not in _POTENTIAL_KEYS:
        raise ValueError(f"unknown potential type {ptype!r}")
    req, opt = _POTENTIAL_KEYS[ptype]
    _check({k: v for k, v in pot.items() if k != "type"}, req | opt, req)

    ens = p["ensemble"]
    _check(ens, {"mu", "dimension", "box"}, {"mu", "dimension"})
    dim = int(ens["dimension"])
    if dim not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    two_dim = dim == 2
    mu = float(ens["mu"])

    exe = p.get("execution", {})
    _check(exe, {"mode", "n_workers", "cell_length", "sampling_window", "target_cycles",
                 "burn_in", "traj_every", "checkpoint_every", "full_trajectory", "seed",
                 "initial_config"})
    mode = exe.get("mode", "serial")
    parallel = mode == "parallel"

    lj = ptype == "lj"
    aux = ptype == "aux"
    reweight = ptype == "reweight"

    if lj and two_dim:
        raise ValueError("LJ is 3D only")
    if not lj and mu < 0:
        raise ValueError("non-LJ runs require mu >= 0")
    if aux and not two_dim and "g3_csv" not in pot:
        raise ValueError("3D aux requires potential.g3_csv")

    temp = float(pot["T"]) if lj else 1.0
    beta = 1.0 / temp

    box = float(ens["box"]) if "box" in ens else default_side(two_dim, parallel)

    radius, diameter, r_squared = 0.5, 1.0, 1.0

    tr = p["trials"]
    _check(tr, {"cavity_radius", "trial_density", "k"}, {"cavity_radius"})
    cav = float(tr["cavity_radius"])
    span = (2.0 * cav) ** 2
    disk = math.pi * radius * radius
    if "k" in tr:
        k = int(tr["k"])
        trial_density = round((k * disk) / span, 3)
        n_trials = k
    else:
        trial_density = float(tr["trial_density"])
        n_trials = int((trial_density * span) / disk)

    no_neigh = 40 if two_dim else 75
    if reweight:
        uc = 3 * radius
    elif lj:
        uc = float(pot.get("rc", 2.5))
    else:
        uc = 3.7545 if two_dim else 2.0

    if two_dim:
        vol_mol = math.pi * radius ** 2
        total_vol = box ** 2
    else:
        vol_mol = (4.0 / 3.0) * math.pi * radius ** 3
        total_vol = box ** 3

    lj_tail_pref = None
    if lj:
        x = uc ** -9 / 3.0 - uc ** -3
        lj_tail_pref = (8.0 / 3.0) * math.pi / total_vol * x

    alg = p.get("algorithm", {})
    _check(alg, {"depth", "branch", "use_depth"})
    move = p.get("move", {})
    _check(move, {"move_factor"})

    return Config(
        potential_type=ptype, dimension=dim, chem_pot=mu, box=box,
        cavity_radius=cav, trial_density=trial_density,
        depth=int(alg.get("depth", 0)), branch=bool(alg.get("branch", False)),
        use_depth=bool(alg.get("use_depth", True)),
        move_factor=float(move.get("move_factor", 0.25)),
        temp=temp, rc=(uc if lj else None),
        eta=(float(pot.get("eta", 0.69)) if aux and two_dim else None),
        gamma_max=(float(pot["gamma_max"]) if "gamma_max" in pot else None),
        delta_eta=(float(pot["delta_eta"]) if "delta_eta" in pot else None),
        g3_csv=(pot.get("g3_csv") if aux and not two_dim else None),
        two_dim=two_dim, parallel=parallel, beta=beta,
        fugacity=math.exp(beta * mu), radius=radius, diameter=diameter,
        r_squared=r_squared, no_of_trial_molecules=n_trials, no_neigh=no_neigh,
        upper_cutoff=uc, volume_of_molecule=vol_mol, total_volume=total_vol,
        lj_tail_pref=lj_tail_pref, gamma_dict=_gamma_dict(aux, reweight, two_dim),
        mode=mode, n_workers=int(exe.get("n_workers", 1)),
        cell_length=float(exe.get("cell_length", 30.0)),
        sampling_window=int(exe.get("sampling_window", 1000)),
        target_cycles=float(exe.get("target_cycles", 1.0025e9)),
        burn_in=int(exe.get("burn_in", 0)),
        traj_every=int(exe.get("traj_every", 1)),
        checkpoint_every=int(exe.get("checkpoint_every", 1)),
        full_trajectory=bool(exe.get("full_trajectory", False)),
        initial_config=exe.get("initial_config"),
    )
