"""The potential registry, and the import that populates it.

`base` holds the `Potential` protocol and the `register` decorator; the modules
imported below are the implementations. They are imported HERE and nowhere else
for a reason: a registry is only useful once every mode has registered itself, so
importing the package is what makes `make_potential("lj_table")` resolvable. A
caller that reached for a module directly would get a registry missing whichever
modes it had not happened to touch.

`auxiliary` is spelled out rather than abbreviated to `aux` because `AUX` is a
reserved DOS device name that Windows still honours: `aux.py` cannot be created,
checked out or unpacked on a Windows filesystem, so the tree was unusable there.
The config vocabulary is unaffected -- the potential is still registered and
selected as `"aux"`, `"aux2d"`, `"ljaux"`; only the module file was renamed.
"""

from __future__ import annotations

from . import auxiliary, hard_sphere, lennard_jones, lj_table
from .base import Potential, available, make_potential, register

__all__ = ["Potential", "available", "make_potential", "register"]
