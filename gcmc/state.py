"""The live chain state.

TLDR: Holds the live chain (positions, N, counters, step, and the single RNG that io checkpoints)

Holds the configuration the chain evolves (positions), the cumulative step
counter, the move counters, and the single random generator that everything
draws from and that io checkpoints. Keeping one generator here is what makes a
restart reproducible: its state is saved and restored, so the Markov chain is
continuous across a walltime kill.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def zero_counters():
    return {"add_step": 0, "add_attempt": 0, "add_accept": 0,
            "remove_step": 0, "remove_accept": 0,
            "move_step": 0, "move_accept": 0,
            "repack_step": 0, "repack_attempt": 0, "repack_accept": 0}
@dataclass
class ChainState:
    positions: np.ndarray
    step: int
    rng: np.random.Generator
    counters: dict = field(default_factory=zero_counters)
    # target-mu control (muVT only): `mu` is the live, adjusted chemical
    # potential (None => use the fixed cfg.chem_pot, the default fixed-mu run);
    # `n_history` is the rolling window of per-window N used for the running mean
    # that the controller steers. Both are checkpointed so a resume continues
    # the search rather than restarting it.
    mu: float | None = None
    n_history: list = field(default_factory=list)

    @classmethod
    def fresh(cls, dimension, seed=None):
        pos = np.empty((0, dimension), dtype=float)
        return cls(positions=pos, step=0, rng=np.random.default_rng(seed))

    @property
    def n(self):
        return len(self.positions)
