"""Config-level checks for items 5, 7, 8 and the item-2 no-op guard.

These exercise gcmc.run.sim_setup.from_params directly: they are fast and pin the
decisions that are easy to regress in the config layer.

Item 5  every LJ sampler shares one hard-core floor (0.9 sigma by default), and
        hard-core runs use the diameter.
Item 7  the bulk LJ tail correction is switched off in a pore or a QA matrix, and
        asking for it explicitly there is an error, not a silent drop.
Item 8  pore-size floors scale with the actual sphere diameter 2*radius, not a
        hardcoded 1.
Item 2  an NVT window with no particles returns immediately instead of looping.
"""

import numpy as np

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics
from gcmc.run import sim_setup
from gcmc.system.state import ChainState


def _lj(sampler_block, **run):
    p = {
        "potential": {"main": "lennard_jones", "T": 1.2},
        "ensemble": {"mu": -3.0, "dimension": 3, "box": 10.0},
        "sampler": sampler_block,
        "run": run,
    }
    return sim_setup.from_params(p)


# ---------------------------------------------------------------- item 5
def test_all_lj_samplers_share_one_floor():
    metro = _lj({"type": "metropolis"})
    rosen = _lj({"type": "rosenbluth", "trials": 20, "trial_box_side": 3.0, "depth": 0})
    branch = _lj({"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1})
    floors = {round(c.lower_cutoff, 6) for c in (metro, rosen, branch)}
    print(f"LJ floors: {floors}")
    assert floors == {0.9}, f"LJ samplers disagree on the hard floor: {floors}"
    # 0.85 must be gone as an implicit default anywhere.
    assert metro.lower_cutoff == 0.9


def test_hard_core_floor_is_diameter():
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": -3.0, "dimension": 3, "box": 10.0},
        "sampler": {"type": "metropolis"},
        "run": {"radius": 0.6},
    }
    cfg = sim_setup.from_params(p)
    assert abs(cfg.lower_cutoff - 1.2) < 1e-12, cfg.lower_cutoff


def test_lower_cutoff_override_respected():
    cfg = _lj({"type": "metropolis"}, lower_cutoff=0.8)
    assert cfg.lower_cutoff == 0.8


# ---------------------------------------------------------------- item 7
def test_bulk_lj_has_tail():
    cfg = _lj({"type": "metropolis"})
    assert cfg.lj_tail_pref is not None and cfg.lj_tail_pref != 0.0


def _confined_lj(pore, tail=None):
    pot = {"main": "lennard_jones", "T": 1.2}
    if tail is not None:
        pot["tail"] = tail
    p = {
        "potential": pot,
        "ensemble": {
            "type": "confined",
            "mu": -3.0,
            "dimension": 3,
            "box": 10.0,
            "pore": pore,
        },
        "sampler": {"type": "metropolis"},
        "run": {"radius": 0.5},
    }
    return sim_setup.from_params(p)


def test_pore_disables_tail_by_default():
    cfg = _confined_lj({"geometry": "slit", "H": 4.0})
    assert cfg.lj_tail_pref == 0.0, "tail should be off inside a pore"


def test_pore_explicit_tail_raises():
    try:
        _confined_lj({"geometry": "slit", "H": 4.0}, tail=True)
    except ValueError as e:
        assert "tail" in str(e).lower()
        return
    raise AssertionError("explicit tail=true in a pore must raise")


def test_qa_matrix_disables_tail():
    # from_params records the matrix path without reading the file, so a dummy
    # path is enough to exercise the QA branch. QA needs a branch sampler for LJ.
    p = {
        "potential": {"main": "lennard_jones", "T": 1.2},
        "ensemble": {"type": "QA", "mu": -3.0, "dimension": 3, "box": 10.0},
        "sampler": {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
        "run": {"matrix_config": "/tmp/does_not_need_to_exist.xyz"},
    }
    cfg = sim_setup.from_params(p)
    assert cfg.lj_tail_pref == 0.0


# ---------------------------------------------------------------- item 8
def _slit(H, radius, box=10.0):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {
            "type": "confined",
            "mu": -3.0,
            "dimension": 3,
            "box": box,
            "pore": {"geometry": "slit", "H": H},
        },
        "sampler": {"type": "metropolis"},
        "run": {"radius": radius},
    }
    return sim_setup.from_params(p)


def test_pore_gap_uses_actual_diameter():
    # radius 0.5 -> diameter 1.0: H=1.5 fits, H=0.8 does not.
    _slit(1.5, 0.5)
    try:
        _slit(0.8, 0.5)
        raise AssertionError("H below the diameter must raise")
    except ValueError:
        pass
    # radius 0.75 -> diameter 1.5: H=1.2 must now FAIL (it passed under a
    # hardcoded 1), H=2.0 fits.
    _slit(2.0, 0.75)
    try:
        _slit(1.2, 0.75)
        raise AssertionError("H=1.2 must raise for diameter 1.5 (hardcoded-1 bug)")
    except ValueError:
        pass


# ---------------------------------------------------------------- item 2
def test_nvt_empty_window_returns():
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"type": "NVT", "N": 8, "dimension": 3, "box": 10.0},
        "sampler": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
            "depth": 1,
        },
        "run": {"sampling_window": 5},
    }
    cfg = sim_setup.from_params(p)
    pot = cfg.build_potential()
    w = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(3, seed=0)
    st.positions = np.empty((0, 3))  # empty start: repack has nothing to do
    # must return promptly rather than spinning forever on an empty box.
    metropolis_dynamics.advance(st, cfg, pot, w)
    assert len(st.positions) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("PASS test_items_config_guards")
