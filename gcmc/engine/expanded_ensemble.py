"""Expanded-ensemble (staged) grand-canonical insertion: the second incumbent.

G0.5 of the design plan. Siderius & Shen (J. Phys. Chem. C 2013,
117, 5861) hit the same pathology the window move exists to solve -- "since it
can be difficult to insert particles at high density, we adopted the method of
expanded ensembles for the GC particle insertion/deletion moves" -- and solved
it by growing the LJ sigma of one molecule stepwise through stages. That remedy
is cheap, general, needs no transfer matrix and works in 3D, so a new engine
that has not been measured against it has not been measured.

It is deliberately a SEPARATE, SELF-CONTAINED LOOP rather than a branch inside
`metropolis_dynamics`. Two reasons, in order:

  1. The plan's group 2C -- "changes a live chain" -- must stay empty through
     G1, and a new move type inside `advance` or `metropolis_window` is exactly
     that. Here nothing existing can reach this file.
  2. It is a BASELINE. A baseline computed by the code under test shares that
     code's mistakes. The pair energy, the minimum image and the acceptance
     below are written out from scratch and share nothing with the samplers,
     so agreement with them is evidence.

THE ENSEMBLE, written down because the boundary terms are where these schemes
go wrong. The extended state is

    I = N * M + m,     N = I // M,   m = I % M,   lambda_m = m / M

with N full molecules and ONE always-present "ghost" molecule coupled at
`lambda_m`. At m = 0 the ghost is ideal: it occupies a position and interacts
with nothing. Its weight is

    pi(N, m)  ~  exp(w_m) * z^(N+1) / N! * exp(-beta U(r^N ; ghost at lambda_m))

Transitions are I -> I +/- 1 only, so the ladder is one-dimensional and every
move is a single stage step:

  * m -> m+1 with m+1 <= M-1  (same N): the ghost's coupling grows in place.
        acc = exp(w_{m+1} - w_m - beta dU_ghost)
  * m = M-1 -> (N+1, m=0): the ghost reaches lambda = 1, becomes the (N+1)-th
    full molecule, and a FRESH ghost is drawn uniformly.
        acc = z V / (N+1) * exp(w_0 - w_{M-1} - beta dU)
    which is the ordinary grand-canonical insertion factor -- as it must be,
    because that is the transition that changes N.
  * the reverse of each, with the reverse of the boundary move choosing one of
    the N+1 full molecules uniformly to demote and discarding the old ghost.

WHY <N> IS MEASURED ONLY AT m = 0. Conditioned on m = 0 the ghost is fully
decoupled, so the joint density of (N, r^N) is the muVT one times the constant
`z V exp(w_0)`. The conditional distribution is therefore EXACTLY muVT, and
`P(N | m = 0)` is exactly Pi(N). Averaging N over all stages instead would
report the extended ensemble's mean, which is not a physical quantity and is
the standard way to get a plausible wrong isotherm out of this method.

SIGMA-SCALING, not epsilon-scaling, following Siderius & Shen: the ghost's pair
potential is the 12-6 form at `sigma_m = lambda_m * sigma`, so at small lambda
it is a small soft sphere that fits between the neighbours rather than a
full-size one with a shallow well. At lambda = 0 the potential vanishes
identically, which is what makes the m = 0 state ideal. The hard core scales
with it, so no configuration is ever forbidden at low lambda -- which is the
entire point of the method.

THE STAGE WEIGHTS `w_m` are a bias exactly like eta[N]: they flatten stage
occupancy and cancel out of every physical average. `tune_stage_weights` sets
them from a short pilot run by the standard -log(visits) rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._types import RNG, FloatArray


@dataclass
class ExpandedState:
    """The extended chain's state: full molecules, the ghost, and its stage."""

    pos: FloatArray  # (N, dim) the full molecules
    ghost: FloatArray  # (dim,) the ghost's position, always present
    stage: int  # m in 0..M-1; lambda = m / M


@dataclass
class ExpandedGCMC:
    """Single-species expanded-ensemble muVT for LJ or hard spheres.

    box, dim, beta, mu       the state point, in reduced units
    sigma, epsilon           the full-strength pair parameters
    rc                       interaction cutoff (LJ only)
    hard_core                True: exclusion only, no soft interaction
    n_stages                 M; stage m couples at sigma_m = (m/M) * sigma
    move_factor              translation amplitude
    stage_weights            w_m, length M; zeros means unbiased
    tail_pref                mean-field tail prefactor, U_tail = tail * N^2,
                             or None. Applied at the ONE transition that
                             changes N, so it enters exactly once per particle
                             exactly as it does in `metropolis_dynamics`.
    """

    box: float
    dim: int
    beta: float
    mu: float
    sigma: float = 1.0
    epsilon: float = 1.0
    rc: float = 2.5
    hard_core: bool = False
    n_stages: int = 8
    move_factor: float = 0.25
    stage_weights: FloatArray = field(default_factory=lambda: np.zeros(0))
    tail_pref: float | None = None
    #: visits per stage, for `tune_stage_weights`
    stage_visits: FloatArray = field(default_factory=lambda: np.zeros(0))
    #: counters, same naming convention as `system.state.zero_counters`
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Size the stage arrays and zero the counters."""
        M = self.n_stages
        if M < 2:
            raise ValueError(f"expanded ensemble needs at least 2 stages, got {M}")
        if self.stage_weights.size == 0:
            self.stage_weights = np.zeros(M)
        if self.stage_weights.size != M:
            raise ValueError(
                f"stage_weights has {self.stage_weights.size} entries for {M} stages"
            )
        self.stage_visits = np.zeros(M)
        for k in (
            "up_step",
            "up_accept",
            "down_step",
            "down_accept",
            "move_step",
            "move_accept",
            "ghost_step",
            "ghost_accept",
            "insert",
            "delete",
        ):
            self.counters.setdefault(k, 0)

    # -- geometry and energy, written out; shares nothing with the samplers --
    @property
    def volume(self) -> float:
        """Box volume in reduced units."""
        return float(self.box) ** self.dim

    def _minimage_d2(self, p: FloatArray, others: FloatArray) -> FloatArray:
        """Squared minimum-image distances from `p` to every row of `others`."""
        if len(others) == 0:
            return np.empty(0)
        d = np.asarray(others, float) - np.asarray(p, float)
        d -= self.box * np.round(d / self.box)
        return np.asarray(np.einsum("ij,ij->i", d, d), dtype=np.float64)

    def _u_pair(self, d2: FloatArray, sig: float) -> float:
        """Truncated-shifted-free 12-6 energy of one molecule of diameter `sig`
        against squared distances `d2`; +inf when anything is inside the core.

        `sig` is the SCALED diameter, so the core and the well move together and
        a stage-0 ghost has neither.
        """
        if sig <= 0.0:
            return 0.0
        if self.hard_core:
            return math.inf if bool((d2 <= (sig - 1e-12) ** 2).any()) else 0.0
        # the core of a 12-6 potential is its own exclusion: nothing extra is
        # imposed, so a stage can never be geometrically forbidden
        within = d2 < self.rc * self.rc
        if not within.any():
            return 0.0
        x = (sig * sig) / d2[within]
        return float(4.0 * self.epsilon * np.sum(x**6 - x**3))

    def _u_full(self, p: FloatArray, others: FloatArray) -> float:
        """Energy of a FULL molecule at `p` against `others`."""
        return self._u_pair(self._minimage_d2(p, others), self.sigma)

    def _u_ghost(self, st: ExpandedState, stage: int) -> float:
        """Energy of the ghost at coupling `stage` against the full molecules."""
        if stage == 0:
            return 0.0
        sig = self.sigma * stage / self.n_stages
        return self._u_pair(self._minimage_d2(st.ghost, st.pos), sig)

    def _tail(self, n: int) -> float:
        """beta * U_tail(n), zero when no tail is configured."""
        if self.tail_pref is None:
            return 0.0
        return self.beta * self.tail_pref * float(n) ** 2

    # -- the moves --------------------------------------------------------
    def _translate(self, st: ExpandedState, rng: RNG) -> None:
        """Displace one full molecule, ordinary Metropolis.

        The ghost is part of the environment at ITS OWN coupling, and the pair
        is scored with exactly the sigma `_u_ghost` uses -- one scaled diameter
        for every pair the ghost is in. A mixing rule here instead would make
        the ghost-full energy depend on which molecule moved, so the same
        configuration would have two energies and detailed balance would fail
        in a way that shows up as a density offset and nothing louder.
        """
        self.counters["move_step"] += 1
        n = len(st.pos)
        if n == 0:
            return
        j = int(rng.integers(n))
        others = np.delete(st.pos, j, axis=0)
        old = st.pos[j].copy()
        new = (
            old + self.move_factor * (rng.uniform(0.0, 1.0, self.dim) - 0.5)
        ) % self.box
        du = self._u_full(new, others) - self._u_full(old, others)
        if st.stage > 0:
            sig_g = self.sigma * st.stage / self.n_stages
            g = st.ghost[None, :]
            du += self._u_pair(self._minimage_d2(new, g), sig_g)
            du -= self._u_pair(self._minimage_d2(old, g), sig_g)
        if du == math.inf:
            return
        if math.log(rng.uniform(0.0, 1.0)) <= -self.beta * du:
            st.pos[j] = new
            self.counters["move_accept"] += 1

    def _move_ghost(self, st: ExpandedState, rng: RNG) -> None:
        """Reposition the ghost.

        At stage 0 it interacts with nothing, so a UNIFORM redraw is exact and
        accepted with probability 1; that is what lets the chain try a different
        insertion site without paying for a stage climb, and it is the move that
        makes stage 0 mix at all (a decoupled ghost is otherwise frozen where
        the last demotion left it). At stage > 0 it is an ordinary displacement
        against the full molecules.
        """
        self.counters["ghost_step"] += 1
        if st.stage == 0:
            st.ghost = rng.uniform(0.0, self.box, self.dim)
            self.counters["ghost_accept"] += 1
            return
        old = st.ghost.copy()
        u_old = self._u_ghost(st, st.stage)
        st.ghost = (
            old + self.move_factor * (rng.uniform(0.0, 1.0, self.dim) - 0.5)
        ) % self.box
        u_new = self._u_ghost(st, st.stage)
        if u_new == math.inf or math.log(rng.uniform(0.0, 1.0)) > -self.beta * (
            u_new - u_old
        ):
            st.ghost = old
            return
        self.counters["ghost_accept"] += 1

    def _stage_step(self, st: ExpandedState, rng: RNG, up: bool) -> None:
        """One I -> I +/- 1 stage move, including the two boundary transitions.

        The four branches are the four cases in the module docstring, and the
        only ones that touch N are the two boundaries -- which is why the tail
        term appears there and nowhere else, exactly as it appears at one place
        in `metropolis_dynamics.advance`.
        """
        M = self.n_stages
        w = self.stage_weights
        m = st.stage
        n = len(st.pos)
        log_z = self.beta * self.mu
        self.counters["up_step" if up else "down_step"] += 1

        if up and m + 1 <= M - 1:
            du = self._u_ghost(st, m + 1) - self._u_ghost(st, m)
            if du == math.inf:
                return
            if math.log(rng.uniform(0.0, 1.0)) <= w[m + 1] - w[m] - self.beta * du:
                st.stage = m + 1
                self.counters["up_accept"] += 1
            return

        if up:
            # m = M-1 -> (N+1, 0): the ghost reaches lambda = 1, becomes the
            # (N+1)-th full molecule, and a FRESH ghost is drawn uniformly.
            # z V / (N+1) is the ordinary grand-canonical insertion factor, as
            # it must be: this is the one transition that changes N.
            u_ghost = self._u_ghost(st, m)
            u_full = self._u_full(st.ghost, st.pos)
            if u_full == math.inf:
                return
            log_acc = (
                log_z
                + math.log(self.volume)
                - math.log(n + 1)
                + w[0]
                - w[m]
                - self.beta * (u_full - u_ghost)
                - (self._tail(n + 1) - self._tail(n))
            )
            if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
                st.pos = (
                    np.vstack([st.pos, st.ghost[None, :]])
                    if n
                    else st.ghost[None, :].copy()
                )
                st.ghost = rng.uniform(0.0, self.box, self.dim)
                st.stage = 0
                self.counters["up_accept"] += 1
                self.counters["insert"] += 1
            return

        if m - 1 >= 0:
            du = self._u_ghost(st, m - 1) - self._u_ghost(st, m)
            if du == math.inf:
                return
            if math.log(rng.uniform(0.0, 1.0)) <= w[m - 1] - w[m] - self.beta * du:
                st.stage = m - 1
                self.counters["down_accept"] += 1
            return

        # m = 0 -> (N-1, M-1): one full molecule, chosen uniformly, is demoted
        # to a ghost at lambda_{M-1}; the old decoupled ghost is discarded. The
        # 1/(N) proposal probability is what puts log(n) in the acceptance and
        # makes this the exact reverse of the boundary insertion above.
        if n == 0:
            return
        j = int(rng.integers(n))
        cand = st.pos[j].copy()
        rest = np.delete(st.pos, j, axis=0)
        sig = self.sigma * (M - 1) / M
        u_dem = self._u_pair(self._minimage_d2(cand, rest), sig)
        if u_dem == math.inf:
            return
        u_was = self._u_full(cand, rest)
        log_acc = (
            math.log(n)
            - log_z
            - math.log(self.volume)
            + w[M - 1]
            - w[0]
            - self.beta * (u_dem - u_was)
            - (self._tail(n - 1) - self._tail(n))
        )
        if math.log(rng.uniform(0.0, 1.0)) <= log_acc:
            st.pos = rest
            st.ghost = cand
            st.stage = M - 1
            self.counters["down_accept"] += 1
            self.counters["delete"] += 1

    # -- the driver -------------------------------------------------------
    def run(
        self,
        st: ExpandedState,
        rng: RNG,
        n_steps: int,
        *,
        record_every: int = 1,
    ) -> FloatArray:
        """Advance `n_steps` proposals and return the recorded N series.

        ONLY m = 0 CONFIGURATIONS ARE RECORDED, which is what makes the series a
        muVT sample of N (see the module docstring). The return value is
        therefore shorter than `n_steps // record_every`, and how much shorter
        is itself the diagnostic: a chain that spends almost no time at stage 0
        is one whose stage weights need tuning.
        """
        out: list[int] = []
        for i in range(n_steps):
            self.stage_visits[st.stage] += 1.0
            r = rng.uniform(0.0, 1.0)
            if r < 0.4:
                self._translate(st, rng)
            elif r < 0.5:
                self._move_ghost(st, rng)
            elif r < 0.75:
                self._stage_step(st, rng, up=True)
            else:
                self._stage_step(st, rng, up=False)
            if st.stage == 0 and i % record_every == 0:
                out.append(len(st.pos))
        return np.asarray(out, dtype=np.float64)

    def tune_stage_weights(self, *, damping: float = 1.0) -> FloatArray:
        """Set w_m = -log(visits_m) from the pilot histogram, anchored at w_0.

        The same rule as `FlatHistogram.refresh`, applied to the stage ladder
        instead of the macrostate ladder: a flat stage histogram means the
        chain climbs and descends freely, which is the whole reason the method
        beats direct insertion at high density.
        """
        v = np.maximum(self.stage_visits, 1.0)
        target = -np.log(v / v.sum())
        self.stage_weights = (1.0 - damping) * self.stage_weights + damping * target
        self.stage_weights = self.stage_weights - self.stage_weights[0]
        self.stage_visits = np.zeros(self.n_stages)
        return self.stage_weights


def fresh_state(box: float, dim: int, pos: FloatArray, rng: RNG) -> ExpandedState:
    """An extended state from an ordinary configuration: stage 0, ghost placed
    uniformly. Stage 0 is the muVT-equivalent slice, so this is the state a
    plain configuration corresponds to exactly."""
    return ExpandedState(
        pos=np.asarray(pos, float).copy(),
        ghost=rng.uniform(0.0, box, dim),
        stage=0,
    )
