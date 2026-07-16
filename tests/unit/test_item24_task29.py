"""Item 24 (fail fast on invalid configs) and task 29 (channel uses trial_density
over an oriented cuboid, never trial_box_side).

Item 24 gives each invalid input its own specific error instead of a late crash
or a silent nan, and from_params should never raise anything other than
ValueError on bad input.

Task 29: the channel trial region is a cuboid (channel_length long, square
channel_width cross-section), so trial_box_side is meaningless there and is
rejected; the trial count is trial_density (spheres per unit volume) times that
cuboid volume, channel_length * channel_width**2.
"""

import numpy as np

from gcmc.sim_setup import from_params


def _base(**over):
    p = {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": -2.0, "dimension": 3, "box": 10.0},
        "sampler": {"type": "metropolis"},
        "run": {},
    }
    for k, v in over.items():
        p[k] = v
    return p


def _raises(p, needle=None):
    try:
        from_params(p)
    except ValueError as e:
        if needle:
            assert needle in str(e).lower(), f"wrong message: {e}"
        return
    raise AssertionError(f"expected ValueError for {p}")


# ---------------------------------------------------------------- item 24
def test_box_below_min_image_raises():
    # hard core: interaction range is the diameter; box must be >= 2*diameter.
    _raises(
        _base(ensemble={"mu": -2.0, "dimension": 3, "box": 1.5}, run={"radius": 0.5}),
        needle="minimum-image",
    )
    # LJ: interaction range is the cutoff.
    _raises(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "ensemble": {"mu": -2.0, "dimension": 3, "box": 4.0},
            "sampler": {"type": "metropolis"},
            "run": {"upper_cutoff": 2.5},
        },
        needle="minimum-image",
    )


def test_box_at_min_image_ok():
    # box exactly 2*diameter is allowed (boundary of the convention).
    from_params(
        _base(ensemble={"mu": -2.0, "dimension": 3, "box": 2.0}, run={"radius": 0.5})
    )


def test_move_factor_and_window():
    _raises(_base(run={"move_factor": 0.0}), needle="move_factor")
    _raises(_base(run={"move_factor": -1.0}), needle="move_factor")
    _raises(_base(run={"sampling_window": 0}), needle="sampling_window")


def test_branch_needs_depth_plus_one_trials():
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": -2.0, "dimension": 3, "box": 10.0},
            "sampler": {
                "type": "branch",
                "trials": 2,
                "trial_box_side": 3.0,
                "depth": 3,
            },
            "run": {},
        },
        needle="depth+1",
    )


# ---------------------------------------------------------------- task 29
def test_channel_rejects_trial_box_side():
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "channel",
                "trials": 30,
                "trial_box_side": 3.0,
                "depth": 1,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            "run": {},
        },
        needle="trial_box_side",
    )


def test_channel_trial_count_from_density_times_volume():
    length, width, density = 6.0, 1.5, 2.0
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "channel",
                "trial_density": density,
                "depth": 1,
                "channel_length": length,
                "channel_width": width,
            },
            "run": {},
        }
    )
    expected = round(density * length * width * width)  # 2.0 * 6 * 2.25 = 27
    assert cfg.no_of_trial_molecules == expected, cfg.no_of_trial_molecules
    assert cfg.trial_box_side == 0.0
    # trials given directly is also accepted and back-fills a density
    cfg2 = from_params(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "channel",
                "trials": 30,
                "depth": 1,
                "channel_length": length,
                "channel_width": width,
            },
            "run": {},
        }
    )
    assert cfg2.no_of_trial_molecules == 30


def test_channel_density_too_low_raises():
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "channel",
                "trial_density": 0.0001,
                "depth": 1,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            "run": {},
        },
        needle="trial pool is empty",
    )


# ---------------------------------------------------------------- fuzz
def test_fuzz_never_crashes_uncontrolled():
    rng = np.random.default_rng(0)
    mains = ["hard_core", "lennard_jones"]
    samplers = ["metropolis", "rosenbluth", "branch", "channel"]
    ok = bad = 0
    for _ in range(400):
        main = rng.choice(mains)
        samp = rng.choice(samplers)
        pot = {"main": main}
        if main == "lennard_jones":
            pot["T"] = float(rng.uniform(0.5, 2.0))
        smp = {"type": samp}
        if samp != "metropolis":
            smp["depth"] = int(rng.integers(0, 4))
            if rng.random() < 0.5:
                smp["trials"] = int(rng.integers(0, 40))
            else:
                smp["trial_density"] = float(rng.uniform(0, 3))
            if samp == "channel":
                smp["channel_length"] = float(rng.uniform(0, 8))
                smp["channel_width"] = float(rng.uniform(0, 3))
            elif rng.random() < 0.5:
                smp["trial_box_side"] = float(rng.uniform(0.5, 4))
        p = {
            "potential": pot,
            "ensemble": {
                "mu": float(rng.uniform(-5, 3)),
                "dimension": 3,
                "box": float(rng.uniform(1.0, 15.0)),
            },
            "sampler": smp,
            "run": {
                "move_factor": float(rng.uniform(-0.1, 0.5)),
                "sampling_window": int(rng.integers(0, 5)),
            },
        }
        try:
            from_params(p)
            ok += 1
        except ValueError:
            bad += 1  # acceptable: a specific, intended rejection
    # every input either built a Config or raised ValueError; no other exception
    # escaped (an uncaught TypeError/ZeroDivision would have failed the test).
    print(f"fuzz: {ok} built, {bad} rejected with ValueError")
    assert ok > 0 and bad > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("PASS test_item24_task29")
