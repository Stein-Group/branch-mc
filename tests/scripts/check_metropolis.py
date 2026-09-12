"""Metropolis (standard single-particle GCMC) correctness.

There is no original to diff against (metropolis is new), and its chemical-
potential convention differs from the cavity/Rosenbluth scheme, so the two do
NOT share an equilibrium density at the same mu. The decisive, convention-free
check is the ideal-gas limit: with a vanishing core the grand-canonical mean
occupancy must be <N> = V * exp(beta*mu). That isolates the insertion/deletion
acceptance, which is the only thing metropolis adds.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc import config_specs
from gcmc.engine import metropolis_dynamics as engine
from gcmc.run.sim_setup import from_params
from gcmc.system.state import ChainState


def mean_N(mu, box, dim, radius, nwin=60, tail=25, seed=1, n0=None):
    c = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {"type": "metropolis"},
            "ensemble": {"mu": mu, "dimension": dim, "box": box},
            "run": {"sampling_window": 2000, "radius": radius},
        }
    )
    pot = c.build_potential()
    w = engine.make_weigher(config_specs.kernel_spec(c), pot)
    st = ChainState.fresh(dim, seed=seed)
    n0 = n0 if n0 is not None else max(1, int(0.5 * c.total_volume * math.exp(mu)))
    st.positions = np.random.default_rng(seed + 1).uniform(0, box, size=(n0, dim))
    Ns = [engine.advance(st, c, pot, w)["N"] for _ in range(nwin)]
    return float(np.mean(Ns[-tail:])), c.total_volume


# ideal-gas limit in 3D: tiny core, <N> = V e^{beta mu} (mu >= 0 per the HS domain)
for mu in (0.0, 1.0, 2.0):
    got, V = mean_N(mu, box=4.0, dim=3, radius=0.01)
    want = V * math.exp(mu)
    rel = abs(got - want) / want
    print(f"  3D mu={mu:+.1f}: <N>={got:7.1f}  V e^mu={want:7.1f}  rel={rel:.3f}")
    assert rel < 0.06, f"ideal-gas occupancy off by {rel:.3f} at mu={mu}"
print("  ok  3D ideal-gas occupancy matches V e^(beta mu)")

# ideal-gas limit in 2D
got, V = mean_N(0.5, box=10.0, dim=2, radius=0.01)
want = V * math.exp(0.5)
rel = abs(got - want) / want
print(f"  2D mu=+0.5: <N>={got:7.1f}  V e^mu={want:7.1f}  rel={rel:.3f}")
assert rel < 0.06, f"2D ideal-gas occupancy off by {rel:.3f}"
print("  ok  2D ideal-gas occupancy matches V e^(beta mu)")

# finite hard core: metropolis must grow and settle below the ideal value
got, V = mean_N(3.0, box=8.0, dim=3, radius=0.5, nwin=35, tail=12, n0=250)
phi = got * (math.pi / 6.0) / V
print(f"  3D hard-core mu=3: <N>={got:.1f}  phi={phi:.3f}")
assert 0.15 < phi < 0.45, f"hard-core packing fraction {phi:.3f} out of physical range"
print("  ok  hard-core metropolis settles at a physical packing fraction")

print("\nALL METROPOLIS TESTS PASSED")
