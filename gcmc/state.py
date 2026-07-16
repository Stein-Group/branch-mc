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


def zero_counters() -> dict[str, int]:
    """Fresh per-move-type attempt/accept counters. `*_step` counts proposals
    of that move type, `*_attempt` the ones that reached the acceptance test,
    `*_accept` the accepted ones; acceptance ratios in the logs come from
    these. A dict (not a dataclass) so io can serialise it as-is."""
    return {
        "add_step": 0,
        "add_attempt": 0,
        "add_accept": 0,
        "remove_step": 0,
        "remove_accept": 0,
        "move_step": 0,
        "move_accept": 0,
        "repack_step": 0,
        "repack_attempt": 0,
        "repack_accept": 0,
    }


@dataclass
class ChainState:
    """Everything that defines the Markov chain at a point in time.

    Physics: `positions` is the configuration (fluid only; frozen obstacles
    live in the config, not here), `step` the cumulative move count, and `rng`
    the generator whose state stream IS the chain -- with the same rng state
    and configuration, the continuation is bit-identical, which is what makes
    a checkpoint/restart indistinguishable from an uninterrupted run.

    Engineering: this is the unit io serialises. Anything that must survive a
    walltime kill belongs here and nowhere else; anything derived (cell list,
    slot maps) is rebuilt per window from this.
    """

    positions: np.ndarray
    step: int
    rng: np.random.Generator
    counters: dict[str, int] = field(default_factory=zero_counters)
    # target-mu control (muVT only): `mu` is the live, adjusted chemical
    # potential (None => use the fixed cfg.chem_pot, the default fixed-mu run);
    # `n_history` is the rolling window of per-window N used for the running mean
    # that the controller steers. Both are checkpointed so a resume continues
    # the search rather than restarting it.
    mu: float | None = None
    n_history: list[float] = field(default_factory=list)
    # rolling window of per-window mean cavity occupancy, used only for the
    # avg_n_cav diagnostic column (local density check n_cav/v_cav vs N/V).
    ncav_history: list[float] = field(default_factory=list)

    @classmethod
    def fresh(cls, dimension: int, seed: int | None = None) -> ChainState:
        """Empty chain (N = 0) with a newly seeded generator; the standard
        starting point for a from-scratch run before equilibration fills it."""
        pos = np.empty((0, dimension), dtype=float)
        return cls(positions=pos, step=0, rng=np.random.default_rng(seed))

    @property
    def n(self) -> int:
        """Current particle number N (fluid particles only)."""
        return len(self.positions)
