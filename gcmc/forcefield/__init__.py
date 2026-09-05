"""The potential registry, and the import that populates it.

`base` holds the `Potential` protocol and the `register` decorator; the modules
imported below are the implementations. They are imported HERE and nowhere else
for a reason: a registry is only useful once every mode has registered itself, so
importing the package is what makes `make_potential("lj_table")` resolvable. A
caller that reached for a module directly would get a registry missing whichever
modes it had not happened to touch.
"""

from __future__ import annotations

from . import aux, hard_sphere, lennard_jones, lj_table
from .base import Potential, available, make_potential, register

__all__ = ["Potential", "available", "make_potential", "register"]
