"""Phase 1 (A1.1-A1.4): the Lennard-Jones potential with per-pair sigma and eps.

The frozen engine's LJ is sigma = eps = 1 everywhere: `_lj_from_ir6`,
`lj_pair_energy`, `lj_pair_energy_sq` are the single source of that shape and are
locked-adjacent (the parity suite reads them). This module adds the table-aware
forms the MOF work needs WITHOUT touching them, and proves at one type that the
new forms ARE the old ones, bit for bit.

The reduction is the whole design. A `PairTable` built from a single guest type
with sigma* = eps* = 1 must make `table_pair_energy_sq(d2, 0, 0)` return exactly
`lj_pair_energy_sq(d2)` -- same float, same bits -- so that a single-species Xe
fluid with no framework is the existing "lj"/"ljbranch" run and nothing else.
Everything downstream (the sweep, the reduction tests, the schema-2 seeds)
depends on that identity holding to the last bit, so it is asserted here and in
the test suite rather than assumed.

Conventions, fixed by the plan:
  - types cover guests and framework atoms in one table; guests take the low
    ids, which is the order units.py already emits.
  - the overlap floor is per pair: c * sigma_ij, not a global sigma. The
    `overlaps` helper in geometry.py is unchanged; callers hand it the pair
    floor. The convention d <= sigma_ij*c - 1e-12 is the same convention, only
    the sigma handed to it now depends on the pair.
  - tail correction is meaningless in a pore and hard-disabled in this mode:
    a run that asks for it is a bug, so it raises rather than silently dropping
    the term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._types import FloatArray, IntArray
from . import overlap
from .lennard_jones import _lj_from_ir6


@dataclass(frozen=True)
class PairTable:
    """Precomputed per-pair Lennard-Jones parameters.

    Built once at from_params time from the reduced tables of P0.4. Every array is
    (n_types, n_types) and symmetric; guests occupy the low ids. `core_floor`
    holds c * sigma_ij, the per-pair lower cutoff the overlap test receives.
    """

    types: tuple[str, ...]
    n_guests: int
    sigma: np.ndarray  # sigma*_ij (reduced)
    eps: np.ndarray  # eps*_ij (reduced)
    sigma2: np.ndarray  # sigma*_ij ** 2, so the sq path never re-squares
    core_floor: np.ndarray  # c * sigma*_ij, the per-pair overlap radius argument

    @classmethod
    def from_reduced(
        cls,
        types: tuple[str, ...],
        n_guests: int,
        sigma_star: np.ndarray,
        eps_star: np.ndarray,
        core_floor_c: float,
    ) -> PairTable:
        """From units.ReducedTables' sigma_star / eps_star and the P0.6 floor.

        Kept independent of the units module (which imports yaml and ASE) so a
        potential can be built from bare arrays; sim_setup does the wiring.
        """
        sigma = np.asarray(sigma_star, float)
        eps = np.asarray(eps_star, float)
        n = len(types)
        if sigma.shape != (n, n) or eps.shape != (n, n):
            raise ValueError(f"sigma/eps must be ({n}, {n}) for {n} types")
        if not (np.allclose(sigma, sigma.T) and np.allclose(eps, eps.T)):
            raise ValueError("sigma and eps tables must be symmetric")
        if core_floor_c <= 0 or core_floor_c > 1:
            raise ValueError(
                f"core floor fraction must be in (0, 1], got {core_floor_c}"
            )
        return cls(
            types=types,
            n_guests=n_guests,
            sigma=sigma,
            eps=eps,
            sigma2=sigma * sigma,
            core_floor=core_floor_c * sigma,
        )

    def index(self, type_name: str) -> int:
        """Row/column index of a type name in every table here.

        Raises ValueError on an unknown name rather than returning -1: a
        negative index would silently address the LAST type, giving a plausible
        energy for the wrong pair.
        """
        return self.types.index(type_name)


def table_pair_energy_sq(
    d2: FloatArray, ti: int, tj: int, table: PairTable
) -> FloatArray:
    """Truncated 12-6 pair energy from squared distance, per pair type.

    ir6 = (sigma_ij^2 / d2)^3, and the energy is 4 eps_ij (ir6^2 - ir6). The
    single-type sigma = eps = 1 case is (1/d2)^3 fed to `_lj_from_ir6`, which is
    `lj_pair_energy_sq` exactly -- same operations in the same order, so same
    bits. That identity is the reduction test in test_pair_table.py.
    """
    ir6 = (table.sigma2[ti, tj] / np.asarray(d2, float)) ** 3
    return table.eps[ti, tj] * _lj_from_ir6(ir6)


def table_pair_energy(d: FloatArray, ti: int, tj: int, table: PairTable) -> FloatArray:
    """Same energy from a distance rather than its square.

    Mirrors `lj_pair_energy`: ir6 = (sigma_ij / d)^6. At one type it is
    (1/d)^6 into `_lj_from_ir6`, i.e. `lj_pair_energy` bit for bit.
    """
    ir6 = (table.sigma[ti, tj] / np.asarray(d, float)) ** 6
    return table.eps[ti, tj] * _lj_from_ir6(ir6)


def table_pair_energy_vec(
    d: FloatArray, si: IntArray, sj: IntArray, table: PairTable
) -> FloatArray:
    """Per-pair energy for ARRAYS of type ids, one pair per element.

    `table_pair_energy` takes two scalar type ids, which forces a caller with a
    mixed set of pairs to group by type combination and loop. That is fine once
    per trial cloud and expensive once per CLIQUE: `ljbranch_weight_sum`
    enumerates every k-clique and asks for its intra-clique energy, so a Python
    loop over type pairs there is a loop inside the clique enumeration. Measured
    on the feature matrix, the grouped version cost 53 s on one depth-4 QA point;
    this gather brings the whole tier-2 run back under the budget.

    `si` and `sj` are INDEX arrays and were annotated `FloatArray`, which is
    not merely imprecise: numpy refuses a float array as a fancy index, so the
    annotation described a call that cannot run. Corrected when the
    typed dual-cutoff correction became the second caller and mypy said so.

    `sigma[si, sj]` is a fancy-index gather, so the pair parameters come out as
    arrays and the arithmetic is the same expression `table_pair_energy` uses, in
    the same order -- which is what makes it bit-identical rather than merely
    close. `test_pair_table.py` asserts that against the scalar form.
    """
    ir6 = (table.sigma[si, sj] / np.asarray(d, float)) ** 6
    return table.eps[si, sj] * _lj_from_ir6(ir6)


def pair_floor(ti: int, tj: int, table: PairTable) -> float:
    """The per-pair overlap radius c * sigma_ij to hand geometry.overlaps.

    Pass this where the sigma=eps=1 code passed `diameter`. A pair at exactly
    c*sigma_ij is admissible (the -1e-12 in overlaps excludes only strictly
    below), its energy huge but finite, matching ref_energy.py, which has no
    floor at all and just evaluates the steep repulsion.
    """
    return float(table.core_floor[ti, tj])


def overlaps_table(
    d: FloatArray | float, ti: int, tj: int, table: PairTable
) -> np.ndarray:
    """Per-pair overlap indicator, delegating to `geometry.pair_overlaps`.

    Kept as a thin wrapper rather than reimplemented: the invariant is that the
    overlap comparison exists once, in geometry.py, and the pair floor changes
    only which number is handed to it. This module used to repeat the tolerance
    as a local constant to avoid importing geometry into a potential; geometry
    imports nothing but numpy at module scope (its one scipy use is a local
    import), so that cost was imaginary and the second copy was not.
    """
    return overlap.pair_overlaps(d, ti, tj, table.core_floor)
