"""Back-compat re-export. The loader itself now lives in `gcmc/forcefield/species_file.py`.

It moved because `gcmc.run.sim_setup` validates a species file at `from_params` time
(P2 of multispecies_plan.md, M18), and the engine package must not import
`scripts/`: `scripts` is not in `[tool.setuptools.packages.find]`, so a typed run
from an installed copy of gcmc would have failed at import.

Nothing about the MOF pipeline changes. Every name it used is re-exported here,
`from scripts.mof.units import ...` keeps working unedited, and
`python -m scripts.mof.units` still runs the same CLI.
"""

from __future__ import annotations

import sys

from gcmc.forcefield.species_file import (
    ANGSTROM_M,
    AUX_FIRST_SHELL,
    KB_J_PER_K,
    UNIT_SYSTEMS,
    ForceField,
    Fugacity,
    ReducedTables,
    _main,
    core_floor_scan,
    lj_energy_kelvin,
    load_forcefield,
    peng_robinson,
    reduce_tables,
)

__all__ = [
    "ANGSTROM_M",
    "AUX_FIRST_SHELL",
    "KB_J_PER_K",
    "UNIT_SYSTEMS",
    "ForceField",
    "Fugacity",
    "ReducedTables",
    "core_floor_scan",
    "lj_energy_kelvin",
    "load_forcefield",
    "peng_robinson",
    "reduce_tables",
]

if __name__ == "__main__":
    sys.exit(_main())
