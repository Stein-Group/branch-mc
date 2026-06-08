"""The live chain state.

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
            "move_step": 0, "move_accept": 0}


@dataclass
class ChainState:
    positions: np.ndarray
    step: int
    rng: np.random.Generator
    counters: dict = field(default_factory=zero_counters)

    @classmethod
    def fresh(cls, dimension, seed=None):
        pos = np.empty((0, dimension), dtype=float)
        return cls(positions=pos, step=0, rng=np.random.default_rng(seed))

    @property
    def n(self):
        return len(self.positions)
