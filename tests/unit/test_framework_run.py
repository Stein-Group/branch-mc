"""Framework runs driven by params files (gcmc.framework).

What is pinned:
  - the schema: every refusal names the key, defaults are the documented ones,
    and the run id tracks physics, never length, cadence, cores or label;
  - the driver: one row per window, the documented move split per window, the
    running energy agrees with a recompute at every checkpoint, resume and
    killed-leg resume are bit-identical, preloading starts where it says;
  - the entry points: `python -m gcmc run` and the launcher run a framework file;
  - slow tier: a run whose only number-changing move is the ladder samples the
    conventional equilibrium, and the framework driver with an empty host
    reproduces the bulk engine at the same state point.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gcmc.forcefield.species_file import (
    ANGSTROM_M,
    KB_J_PER_K,
    load_forcefield,
    reduce_tables,
)
from gcmc.framework.driver import preload_guests, run_framework
from gcmc.framework.params import identity, parse, run_dir, run_id
from gcmc.framework.system import setup_system
from gcmc.run.io import read_log

ROOT = Path(__file__).resolve().parents[2]
FF = "ff/forcefield_A_shift.yaml"


def pa(z: float, T: float = 298.0, ff: str = FF) -> float:
    """Fugacity in Pa of a reduced activity z* (inverse of ReducedTables.z_star)."""
    sigma = reduce_tables(load_forcefield(ROOT / ff), T).sigma_ref
    return z * KB_J_PER_K * T / (sigma * ANGSTROM_M) ** 3


def params(**over) -> dict:
    """A small empty-box Xe/Kr run: fast enough for the unit tier."""
    p = {
        "potential": {"main": "lennard_jones", "T": 298.0, "species_file": FF},
        "framework": {"empty_box": 30.0},
        "engine": {"type": "metropolis_dynamics"},
        "sampler": {"type": "metropolis"},
        "ensemble": {
            "type": "muVT",
            "dimension": 3,
            "fugacity": {"Xe": pa(0.02), "Kr": pa(0.02)},
        },
        "run": {"seed": 5, "sampling_window": 60, "target_cycles": 240},
    }
    for path, val in over.items():
        sec, _, key = path.partition(".")
        if key:
            p.setdefault(sec, {})[key] = val
        else:
            p[sec] = val
    return p


BRANCH = {
    "engine": {"type": "suwa_todo_dynamics"},
    "sampler": {"type": "branch", "trial_box_side": 1.5, "trial_density": 1.5},
}
CHANNEL = {
    "engine": {"type": "suwa_todo_dynamics"},
    "sampler": {
        "type": "channel",
        "channel_length": 2.5,
        "channel_width": 1.0,
        "trial_density": 4.0,
    },
}
ARMS = {"branch": BRANCH, "channel": CHANNEL}


def rows(res) -> list[dict[str, float]]:
    log = read_log(res.directory / "run.log")
    return [dict(zip(log, vals)) for vals in zip(*log.values())]


# ------------------------------------------------------------------ schema
@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"engine.type": "suwa_todo_dynamics"}, "sampler.type 'branch' or 'channel'"),
        ({"sampler": {"type": "branch"}}, "conventional GCMC"),
        (
            {"engine.move_probabilities": {"translation": 0.6, "exchange": 0.6}},
            "sum to 1",
        ),
        ({"engine.move_probabilities": {"translation": 1.0}}, "nothing would change N"),
        ({"engine.bogus": 1}, "engine.bogus is not a framework-run key"),
        ({"run.esjd": True}, "run.esjd is not a framework-run key"),
        ({"run.burn_in": 10}, "applies to RASPA2 runs"),
        ({"ensemble.pressure": 1e5}, "exactly one of ensemble.fugacity"),
        ({"ensemble.fugacity": [{"Xe": 1.0}, {"Xe": 2.0}]}, "is a list: a sweep"),
        ({"ensemble.fugacity": {"Ar": 1.0}}, "Ar is not a guest"),
        (
            {"framework": {"cif": "structures/MFM-300-In.cif", "empty_box": 30.0}},
            "exactly one of cif or empty_box",
        ),
        ({"run.initial_config": "grid"}, "'empty' or a map"),
        ({"potential.species_file": "ff/species_xekr_reduced.yaml"}, "reduced-units"),
    ],
)
def test_refusals_name_the_key(over, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        parse(params(**over))


def test_identity_swap_needs_a_mixture():
    one = params(**{"ensemble.fugacity": {"Xe": 1e4}})
    one["engine"]["move_probabilities"] = {
        "translation": 0.4,
        "exchange": 0.4,
        "identity_swap": 0.2,
    }
    with pytest.raises(ValueError, match="needs a mixture"):
        parse(one)


def test_documented_defaults():
    mix = parse(params())
    assert mix.move_probs == (0.4, 0.4, 0.2)
    one = parse(params(**{"ensemble.fugacity": {"Xe": 1e4}}))
    assert one.move_probs == (0.5, 0.5, 0.0)
    assert mix.preload is None and mix.checkpoint_every == 10


def test_pressure_route_is_peng_robinson_and_fugacity_route_is_exact():
    p = parse(
        params(
            **{
                "ensemble": {
                    "type": "muVT",
                    "dimension": 3,
                    "pressure": 1e6,
                    "y": {"Xe": 0.2, "Kr": 0.8},
                }
            }
        )
    )
    assert 0.8 < p.phi["Xe"] < 1.0 and abs(sum(p.y.values()) - 1) < 1e-12
    f = parse(params())
    assert f.phi == {"Xe": 1.0, "Kr": 1.0}
    assert abs(f.pressure_pa - 2 * pa(0.02)) < 1e-9 * f.pressure_pa


def test_run_id_is_physics_only():
    base = parse(params())
    same = params(
        **{
            "run.target_cycles": 9e9,
            "run.checkpoint_every": 7,
            "run.n_cores": 16,
            "run.sweep_label": "x",
        }
    )
    assert run_id(parse(same)) == run_id(base)
    for over in (
        {"run.seed": 6},
        {"run.sampling_window": 61},
        {
            "engine.move_probabilities": {
                "translation": 0.5,
                "exchange": 0.3,
                "identity_swap": 0.2,
            }
        },
        {"run.initial_config": {"Xe": 3}},
        BRANCH,
    ):
        assert run_id(parse(params(**over))) != run_id(base), over
    assert "target_cycles" not in json.dumps(identity(base))


def test_paths_resolve_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fr = parse(
        params(
            framework={"cif": "structures/MgMOF-74.cif"},
            potential={
                "main": "lennard_jones",
                "T": 313.0,
                "species_file": "ff/forcefield_CH4_MgMOF74.yaml",
            },
            ensemble={"type": "muVT", "dimension": 3, "fugacity": {"CH4": 1e5}},
        )
    )
    assert fr.cif == (ROOT / "structures" / "MgMOF-74.cif").resolve()
    assert run_dir(fr, "runs").name.endswith("__single")


def test_every_shipped_example_parses():
    files = sorted((ROOT / "examples" / "mof").glob("*.json"))
    assert len(files) == 27
    from gcmc.run.launcher import expand_sweep

    for f in files:
        combos, _labels = expand_sweep(json.loads(f.read_text()))
        assert len(combos) in (1, 16), f
        for c in combos:
            parse(c)


# ------------------------------------------------------------------ driver
def test_rows_window_split_and_closure(tmp_path):
    res = run_framework(params(**BRANCH), tmp_path)
    r = rows(res)
    assert len(r) == 4  # ceil(240 / 60)
    assert list(r[0])[:5] == [
        "MC_cycle",
        "N_Xe",
        "N_Kr",
        "U_host_guest",
        "U_guest_guest",
    ]
    assert "ladder_unchanged" in r[0] and "insert_tried" not in r[0]
    for x in r:
        # 0.4 of 60 moves are window moves, the other 36 translations + swaps
        assert x["translate_tried"] + x["swap_tried"] == 36
    assert sum(x["ladder_sum_abs_dN"] for x in r) > 0, "the ladder never changed N"
    sp = json.loads((res.directory / "state_point.json").read_text())
    assert sp["ladder"]["window_moves_per_row"] == 24
    assert (
        json.loads((res.directory / "params.json").read_text())["engine"]
        == BRANCH["engine"]
    )


def test_conventional_engine_counts_every_move(tmp_path):
    r = rows(run_framework(params(), tmp_path))
    for x in r:
        n = sum(
            x[c]
            for c in ("translate_tried", "insert_tried", "delete_tried", "swap_tried")
        )
        assert n == 60


@pytest.mark.parametrize("arm", list(ARMS))
def test_resume_is_bit_identical(tmp_path, arm):
    short = params(**ARMS[arm], **{"run.target_cycles": 120})
    run_framework(short, tmp_path / "a")
    a = run_framework(params(**ARMS[arm]), tmp_path / "a")
    b = run_framework(params(**ARMS[arm]), tmp_path / "b")
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.species, b.species)
    assert [x["MC_cycle"] for x in rows(a)] == [6e-05, 0.00012, 0.00018, 0.00024]


def test_a_killed_leg_resumes_without_duplicate_windows(tmp_path):
    """A leg killed between checkpoints has streamed rows the next leg replays."""
    import shutil

    p = params(**{"run.checkpoint_every": 2})
    first = run_framework(
        params(**{"run.checkpoint_every": 2, "run.target_cycles": 120}), tmp_path
    )
    ck = first.directory / "checkpoint.npz"
    shutil.copy(ck, tmp_path / "ck2.npz")
    full = rows(run_framework(p, tmp_path))
    shutil.copy(tmp_path / "ck2.npz", ck)
    again = rows(run_framework(p, tmp_path))
    drop = ("wall_s",)
    assert [{k: v for k, v in r.items() if k not in drop} for r in again] == [
        {k: v for k, v in r.items() if k not in drop} for r in full
    ]


def test_preload_starts_where_it_says(tmp_path):
    sysm = setup_system(Path(FF), 298.0, bulk_box=30.0)
    t = sysm.table
    placed, kinds = preload_guests(
        {t.index("Xe"): 20, t.index("Kr"): 10},
        np.random.default_rng(1),
        field=sysm.field,
        table=t,
        box=sysm.box,
    )
    assert int(np.sum(kinds == t.index("Xe"))) == 20
    for i in range(len(placed)):
        dr = placed - placed[i]
        dr -= sysm.box * np.round(dr / sysm.box)
        d = np.sqrt((dr * dr).sum(1))
        d[i] = np.inf
        assert np.all(d >= t.core_floor[kinds[i], kinds] - 1e-12)
    res = run_framework(
        params(
            **{
                "run.initial_config": {"Xe": 20},
                "run.target_cycles": 1,
                "run.sampling_window": 1,
            }
        ),
        tmp_path,
    )
    assert int(rows(res)[0]["N_Xe"]) >= 19


def test_channel_window_is_refused_past_the_guard(tmp_path):
    p = params(**CHANNEL)
    p["sampler"] = {**CHANNEL["sampler"], "channel_length": 6.5}
    with pytest.raises(ValueError, match="periodic guard"):
        run_framework(p, tmp_path)


def test_gcmc_run_and_the_launcher_run_a_framework_file(tmp_path):
    p = params(**BRANCH)
    f = tmp_path / "one.json"
    f.write_text(json.dumps(p))
    env = {"PYTHONPATH": str(ROOT)}
    import os

    env = {**os.environ, **env}
    out = subprocess.run(
        [sys.executable, "-m", "gcmc", "run", str(f), "--root", str(tmp_path / "r")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=True,
    ).stdout
    assert out.startswith("done: step=4 N=")
    p["ensemble"]["fugacity"] = [
        {"Xe": pa(0.01), "Kr": pa(0.01)},
        {"Xe": pa(0.02), "Kr": pa(0.02)},
    ]
    p["run"]["n_cores"] = 2
    f.write_text(json.dumps(p))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gcmc.run.launcher",
            str(f),
            "--root",
            str(tmp_path / "s"),
            "--no-pin",
        ],
        capture_output=True,
        env=env,
        cwd=tmp_path,
        check=True,
    )
    names = sorted(d.name.split("__")[1] for d in (tmp_path / "s").iterdir())
    assert len(names) == 2 and all(n.startswith("fXe_") for n in names), names


# ------------------------------------------------------------------ slow gates
def _means(r, burn):
    from scripts.mof.analysis import blocked_stderr

    out = {}
    for g in ("Xe", "Kr"):
        x = np.array([float(q["N_" + g]) for q in r])[burn:]
        out[g] = (float(x.mean()), blocked_stderr(x, 10))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_v4_ladder_arm_samples_the_conventional_equilibrium(tmp_path, arm):
    """The arm whose ONLY number-changing move is the ladder against the
    conventional arm: same Hamiltonian, same state point."""
    n = 150_000
    conv = params(
        **{"run.target_cycles": n * 2, "run.sampling_window": 200, "run.seed": 11}
    )
    lad = copy.deepcopy(conv)
    lad.update(copy.deepcopy(ARMS[arm]))
    lad["run"]["target_cycles"] = n
    c = _means(rows(run_framework(conv, tmp_path / "c")), 300)
    m = _means(rows(run_framework(lad, tmp_path / "m")), 150)
    for g in ("Xe", "Kr"):
        z = (c[g][0] - m[g][0]) / np.hypot(c[g][1], m[g][1])
        assert abs(z) < 4.0, (g, c[g], m[g], z)


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_the_framework_driver_with_no_host_is_the_bulk_engine(tmp_path, arm):
    """Same kernel, two drivers: the bulk engine on ff/species_xekr_reduced.yaml
    (FF-A reduced, exactly) and the framework driver on FF-A with an empty box of
    the same size. Same activities and body: <N_Xe>, <N_Kr> must agree."""
    from gcmc import config_specs
    from gcmc.engine import metropolis_dynamics, suwa_todo_dynamics
    from gcmc.run import sim_setup
    from gcmc.system.state import ChainState

    z, box_star, W = 0.02, 7.3170731707, 600
    cfg = sim_setup.from_params(
        {
            "potential": {
                "main": "lennard_jones",
                "T": 298.0 / 221.0,
                "species_file": "ff/species_xekr_reduced.yaml",
            },
            "sampler": ARMS[arm]["sampler"],
            "engine": {"type": "suwa_todo_dynamics"},
            "ensemble": {"dimension": 3, "box": box_star, "z": {"Xe": z, "Kr": z}},
            "run": {"seed": 3, "sampling_window": 60},
        }
    )
    pot = cfg.build_potential()
    wgh = metropolis_dynamics.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(3, seed=3)
    st.positions = np.empty((0, 3))
    st.species = np.empty(0, dtype=np.int64)
    bulk_rows = []
    for _ in range(W):
        c = suwa_todo_dynamics.advance(st, cfg, pot, wgh)["n_by_species"]
        bulk_rows.append({"N_Xe": c.get("Xe", 0), "N_Kr": c.get("Kr", 0)})
    bulk = _means(bulk_rows, W // 5)

    unshifted = "ff/forcefield_A.yaml"
    p = params(
        **ARMS[arm],
        **{
            "potential.species_file": unshifted,
            "framework": {"empty_box": box_star * 4.1},
            "ensemble.fugacity": {"Xe": pa(z, ff=unshifted), "Kr": pa(z, ff=unshifted)},
            "run.sampling_window": 60,
            "run.target_cycles": 60 * W,
            "run.seed": 3,
        },
    )
    mof = _means(rows(run_framework(p, tmp_path)), W // 5)
    for g in ("Xe", "Kr"):
        zz = (bulk[g][0] - mof[g][0]) / np.hypot(bulk[g][1], mof[g][1])
        assert abs(zz) < 4.0, (g, bulk[g], mof[g], zz)
    assert math.isfinite(zz)


def test_species_follow_the_force_field_order_not_the_json_order():
    """The launcher snapshots params with sorted keys, so {Kr, Xe} arrives for a
    file that said {Xe, Kr}; columns and labels must not depend on that."""
    a = parse(params(**{"ensemble.fugacity": {"Kr": 2e4, "Xe": 1e4}}))
    b = parse(params(**{"ensemble.fugacity": {"Xe": 1e4, "Kr": 2e4}}))
    assert a.guests == b.guests == ["Xe", "Kr"]
    assert run_id(a) == run_id(b)
