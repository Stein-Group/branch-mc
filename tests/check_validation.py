"""The schema validation: valid axis combinations parse, and every disallowed
combination from the compatibility table raises with a specific message."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gcmc.sim_setup import from_params  # noqa: E402

G3 = os.path.join(HERE, "..", "orig", "g3_analysis_data.csv")


def good(label, p):
    from_params(p)
    print(f"  ok  parses: {label}")


def bad(label, p, needle):
    try:
        from_params(p)
    except ValueError as e:
        assert needle in str(e), f"{label}: message {str(e)!r} lacks {needle!r}"
        print(f"  ok  rejects: {label} -> {e}")
        return
    raise SystemExit(f"FAIL: {label} did not raise")


HC = {"main": "hard_core"}
ROSEN = {"type": "rosenbluth"}

# valid corners of the table
good("hard_core rosenbluth 3D", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": 3.0, "dimension": 3}})
good("hard_core+pmf rosenbluth 2D", {"potential": HC, "auxiliary": {"type": "pmf"},
     "sampler": ROSEN, "ensemble": {"mu": 2.0, "dimension": 2}})
good("hard_core+pmf rosenbluth 3D", {"potential": HC, "auxiliary": {"type": "pmf", "g3_csv": G3},
     "sampler": ROSEN, "ensemble": {"mu": 2.0, "dimension": 3}})
good("lj rosenbluth 3D", {"potential": {"main": "lennard_jones", "T": 1.2}, "sampler": ROSEN,
     "ensemble": {"mu": -3.5, "dimension": 3}})
good("branch 3D", {"potential": HC, "sampler": {"type": "branch"}, "ensemble": {"mu": 3.0, "dimension": 3}})
good("branch + lj 3D", {"potential": {"main": "lennard_jones", "T": 1.2}, "sampler": {"type": "branch"},
     "ensemble": {"mu": -3.0, "dimension": 3}})
good("reweight 2D", {"potential": HC, "sampler": {"type": "reweight"}, "ensemble": {"mu": 2.0, "dimension": 2}})
good("metropolis hard_core 3D", {"potential": HC, "sampler": {"type": "metropolis"},
     "ensemble": {"mu": 3.0, "dimension": 3}})
good("metropolis lj 3D", {"potential": {"main": "lennard_jones", "T": 1.2}, "sampler": {"type": "metropolis"},
     "ensemble": {"mu": -3.5, "dimension": 3}})
good("hard_core 2D parallel", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": 1.0, "dimension": 2},
     "run": {"kind": "parallel", "n_workers": 4}})

# every disallowed cell, with the message it must carry
good("lj in 2D", {"potential": {"main": "lennard_jones", "T": 1.2}, "sampler": ROSEN,
     "ensemble": {"mu": -3.0, "dimension": 2}})
good("parallel 3D", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": 3.0, "dimension": 3},
     "run": {"kind": "parallel"}})
bad("reweight 3D", {"potential": HC, "sampler": {"type": "reweight"}, "ensemble": {"mu": 2.0, "dimension": 3}}, "2D method")
bad("reweight + lj", {"potential": {"main": "lennard_jones", "T": 1.2}, "sampler": {"type": "reweight"},
    "ensemble": {"mu": -3.0, "dimension": 3}}, "hard_core")
bad("lj + pmf", {"potential": {"main": "lennard_jones", "T": 1.2}, "auxiliary": {"type": "pmf"},
    "sampler": ROSEN, "ensemble": {"mu": -3.0, "dimension": 3}}, "g3_csv")
bad("branch + pmf", {"potential": HC, "auxiliary": {"type": "pmf"}, "sampler": {"type": "branch"},
    "ensemble": {"mu": 3.0, "dimension": 3}}, "no auxiliary")
bad("reweight + pmf", {"potential": HC, "auxiliary": {"type": "pmf"}, "sampler": {"type": "reweight"},
    "ensemble": {"mu": 2.0, "dimension": 2}}, "psi6")
bad("metropolis + trials", {"potential": HC, "sampler": {"type": "metropolis", "trials": 10},
    "ensemble": {"mu": 3.0, "dimension": 3}}, "standard GCMC")
bad("metropolis + trial_box_side", {"potential": HC, "sampler": {"type": "metropolis", "trial_box_side": 3.0},
    "ensemble": {"mu": 3.0, "dimension": 3}}, "standard GCMC")
bad("lj missing T", {"potential": {"main": "lennard_jones"}, "sampler": ROSEN,
    "ensemble": {"mu": -3.0, "dimension": 3}}, "requires potential.T")
bad("hard_core with T", {"potential": {"main": "hard_core", "T": 1.0}, "sampler": ROSEN,
    "ensemble": {"mu": 3.0, "dimension": 3}}, "only Lennard-Jones")
bad("aux 3D no g3_csv", {"potential": HC, "auxiliary": {"type": "pmf"}, "sampler": ROSEN,
    "ensemble": {"mu": 2.0, "dimension": 3}}, "g3_csv")
good("non-lj mu<0", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": -1.0, "dimension": 3}})
bad("trials and trial_density", {"potential": HC, "sampler": {"type": "rosenbluth", "trials": 30, "trial_density": 0.5},
    "ensemble": {"mu": 3.0, "dimension": 3}}, "not both")
bad("mu and sweep", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": 3.0, "sweep": [1, 2], "dimension": 3}}, "not both")
bad("unknown sampler", {"potential": HC, "sampler": {"type": "wang_landau"}, "ensemble": {"mu": 3.0, "dimension": 3}}, "unknown sampler")
bad("unknown main", {"potential": {"main": "yukawa"}, "sampler": ROSEN, "ensemble": {"mu": 3.0, "dimension": 3}}, "unknown main")
bad("typo key in sampler", {"potential": HC, "sampler": {"type": "rosenbluth", "trialz": 30},
    "ensemble": {"mu": 3.0, "dimension": 3}}, "unknown keys")
bad("bad dimension", {"potential": HC, "sampler": ROSEN, "ensemble": {"mu": 3.0, "dimension": 4}}, "2 or 3")

print("\nALL VALIDATION TESTS PASSED")
