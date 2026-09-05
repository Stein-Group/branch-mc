"""Selection policies and clique machinery for the multi-trial exchange moves.

Re-exports the weight policies so callers can say ``from gcmc.sampling import
EnergyWeigher``; the sampler modules themselves (``rosenbluth``, ``branch``,
``channel``, ``reweighting``) are imported for side-effect-free access under
one namespace.

This file also makes ``gcmc.sampling`` a regular package rather than an
implicit namespace package, so ``setuptools.find_packages(include=["gcmc*"])``
collects it and a built wheel ships the samplers.
"""

from __future__ import annotations

from . import branch, channel, reweighting, rosenbluth
from .base import EnergyWeigher, ReweightWeigher, UniformWeigher

__all__ = [
    "EnergyWeigher",
    "ReweightWeigher",
    "UniformWeigher",
    "branch",
    "channel",
    "reweighting",
    "rosenbluth",
]
