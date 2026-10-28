"""The live chain state.

Holds the configuration the chain evolves (positions), the cumulative step
counter, the running potential energy, and the single random generator that
everything draws from and that io checkpoints. Keeping one generator here is what makes a
restart reproducible: its state is saved and restored, so the Markov chain is
continuous across a walltime kill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    # D2 (Phase 4): a per-particle integer TYPE id parallel to `positions`, drawn
    # from the same type table framework atoms use, so molecules later reuse it as
    # a per-site type id without a second mechanism. `None` means single-species:
    # every particle is guest type 0 and no species stream is consumed, which is
    # how a Phase 0-3 run stays bit-identical (the main muVT engine never sets it).
    # `species_of` materialises the zeros when a caller needs a concrete array.
    species: np.ndarray | None = None
    # target-mu control (muVT only): `mu` is the live, adjusted chemical
    # potential (None => use the fixed cfg.chem_pot, the default fixed-mu run);
    # `n_history` is the rolling window of per-window N used for the running mean
    # that the controller steers. Both are checkpointed so a resume continues
    # the search rather than restarting it.
    mu: float | None = None
    n_history: list[float] = field(default_factory=list)
    # running potential energy U of `positions`, updated by each window's
    # energy change; None until the runner computes it (LJ runs only)
    energy: float | None = None

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

    @property
    def species_of(self) -> np.ndarray:
        """The species column as a concrete int array of length N: the stored
        column when a multi-species run set one, otherwise all-zeros (the single
        guest type). Callers read this rather than `species` directly so the
        single-species case needs no None handling downstream."""
        if self.species is None:
            return np.zeros(self.n, dtype=np.int64)
        return self.species
