"""R12: imports may only go DOWN the layer ladder.

WHY THIS TEST EXISTS. The package has no import cycles today -- that was
measured, not assumed -- but nothing prevents one, and nothing prevented the
coupling that is already there. `Config` is imported by modules at every level,
which is why "one library, one job" is currently impossible: every module
transitively depends on the union of all configuration.

Making the tree lean once is easy. Keeping it lean needs a mechanism, and this
is it. The ladder is held here AS DATA, so the intended architecture is a thing
you can read in thirty lines rather than infer from forty files.

HOW TO READ A FAILURE. The message names the importing module, its layer, the
imported module and its layer. There are exactly three honest responses:

  1. The import is genuinely backwards -- move the shared thing DOWN to a layer
     both can see. This is right roughly always.
  2. The two modules belong at the same layer -- change LAYERS below, in a
     commit that says why.
  3. The dependency is type-only -- put it under `if TYPE_CHECKING:`, which this
     test ignores, because a type annotation is not a runtime dependency.

Adding a module to `LAYERS` is the ONLY way to make this test pass for new code.
That is deliberate: a new file must be placed on the ladder by a human who
thought about where it goes, not defaulted onto it.

NOTE ON THE CURRENT TREE. The layer assignments below used to describe a TARGET
architecture mapped onto flat filenames, because the tree was flat and the ladder
was the only place the architecture existed. At S5.10 the files moved into it:
`gcmc/forcefield/`, `space/`, `system/`, `proposals/`, `sampling/`, `engine/`,
`run/`, and `analysis/` off to one side. The directory a module sits in is now
its layer, so this table and the tree say the same thing twice -- which is the
point, because a second statement of the same fact is what makes a drift between
them fail rather than pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2] / "gcmc"

# ---------------------------------------------------------------- the ladder
#
# L0  types        vocabulary; imports nothing in-package
# L1  forcefield   what interacts and how strongly; knows no geometry
# L2  space        where things are; knows no physics
# L3  system       the configuration being sampled; where L1 and L2 meet
# L4  proposals    what move to try
# L5  sampling     how to weight a trial cloud
# L6  engine       the move loop
# L7  run          making a params.json runnable
#
# `analysis` is deliberately OFF the ladder (layer 3): it consumes a finished
# chain and must never be reachable from one. Giving it a high layer would let
# the engine import a plotting helper, which is how analysis code ends up in a
# hot loop.
LAYERS: dict[str, int] = {
    # -- L0 -----------------------------------------------------------------
    "_types": 0,
    # The Config field partition (R8) and the sub-config dataclasses it names.
    #
    # MOVED HERE FROM L7 at the parallel migration, and the ladder is what forced
    # it: `metropolis_dynamics` (L6) taking a `KernelSpec` was an upward import,
    # and so was
    # every other migration R8 exists to enable. The old placement reasoned "they
    # describe Config, and Config is L7" -- true, and beside the point. What they
    # ARE is the vocabulary in which a layer states which slice of configuration
    # it needs, and a vocabulary only works if every layer can speak it. That is
    # the definition of L0.
    #
    # They qualify structurally, not just by argument: `config_fields` imports
    # nothing in-package, `config_specs` imports only `config_fields`, and the
    # `Config` they are typed against is a TYPE_CHECKING forward reference with
    # no runtime edge. Nothing had to be bent to put them here.
    "config_fields": 0,
    "config_specs": 0,
    # -- L1 forcefield ------------------------------------------------------
    "forcefield.base": 1,
    "forcefield.pair_table": 1,
    "forcefield.lennard_jones": 1,
    "forcefield.lj_table": 1,  # S5.4: absorbed mof_field (the framework half)
    "forcefield.hard_sphere": 1,
    "forcefield.auxiliary": 1,
    "forcefield.overlap": 1,  # S2: the overlap convention, moved down from L2
    "forcefield.species": 1,
    "forcefield.species_file": 1,
    "forcefield.topology": 1,
    # -- L2 space -----------------------------------------------------------
    "space.geometry": 2,
    "space.neighbors": 2,
    "space.body": 2,  # S2: oriented body geometry, moved down from samplers.channel
    "space.cell_plan": 2,
    "space.decomposition": 2,
    # -- L3 system ----------------------------------------------------------
    "system.state": 3,
    # -- L4 proposals -------------------------------------------------------
    "proposals.moves": 4,
    "proposals.trials": 4,
    "proposals.channel_move": 4,  # S5.7: absorbed the framework window + its sweep factory
    # -- L5 sampling --------------------------------------------------------
    "sampling.base": 5,
    "sampling.rosenbluth": 5,
    "sampling.branch": 5,
    "sampling.channel": 5,
    # The Suwa-Todo geometric allocation. LAYER 5 AND NOT 6, although only the
    # L6 Glauber engine calls it: it is a weighting policy over an occupancy
    # ladder and imports nothing but numpy. A module's layer is what it depends
    # on, never who happens to call it.
    "sampling.suwa_todo": 5,
    "sampling.reweighting": 5,
    # -- L6 engine ----------------------------------------------------------
    "engine.metropolis_dynamics": 6,
    "engine.cell_parallel": 6,
    "engine.rigid_framework": 6,  # S5.5: its metropolis loop merged into metropolis_dynamics
    # The rejection-free Glauber engine: a second DYNAMICS at the same layer
    # as `metropolis_dynamics`, not a branch inside it. `advance` is 2912 lines and its
    # acceptance algebra is built for a different proposal; the Glauber engine has no
    # acceptance algebra at all, so the two share the geometry (L4), the sweep
    # (L5) and the cell list (L2) and nothing else.
    "engine.glauber_dynamics": 6,
    # The Suwa-Todo dynamics: the THIRD dynamics, at the same layer as the
    # other two because it is one of them and not a helper to one. It holds
    # no loop of its own -- the ladder move is shared and picks its kernel
    # from `engine` -- so its only import is its sibling, which is why it
    # sits beside rather than above it.
    "engine.suwa_todo_dynamics": 6,
    # Expanded-ensemble staged insertion: the G0 INCUMBENT BASELINE the heat
    # bath is measured against (Siderius & Shen 2013). A separate loop rather
    # than a branch in `metropolis_dynamics` for two reasons, in order: a new move type
    # inside `advance` would be a change to a live chain, which the plan's
    # group 2C must stay empty of; and a baseline computed by the code under
    # test shares that code's mistakes, so its pair energy, minimum image and
    # acceptance are written out and share nothing with the samplers.
    # -- L7 run -------------------------------------------------------------
    "run.sim_setup": 7,
    "run.guard_audit": 7,
    "run.hash_fingerprint": 7,
    "run.io": 7,
    "run.io_framework": 7,
    "run.runner": 7,
    "run.launcher": 7,
    # Guests in a rigid framework (MOF): a params file with a `framework` block,
    # read, run and resumed beside the bulk runner. Same layer as `run`: the
    # driver reuses the runner's lock and log columns and `sim_setup`'s window
    # rules, and only `__main__` (also L7) dispatches to it.
    "framework.params": 7,
    "framework.structure": 7,
    "framework.system": 7,
    "framework.driver": 7,
    "framework.raspa2": 7,
    "framework.raspa2_parse": 7,
    "__main__": 7,
    # -- off-ladder ---------------------------------------------------------
    "analysis.rdf": 3,
    # Integrated autocorrelation time of a chain observable. Squarely analysis:
    # it consumes a finished series and the engine must never reach it.
    "analysis.autocorr": 3,
    "analysis.widom": 3,
    # Analysis over a configuration: reads positions, a species column and the
    # neighbour structure (L2), and nothing above. Sits with `rdf` because it is
    # the same kind of thing -- a correlation measured on a frame -- and the two
    # are marked in `demixing`'s docstring as an S5 merge candidate.
    "analysis.demixing": 3,
    # U of a logged frame, rebuilt from positions: reads L1 pair energies and
    # the L2 cell list, and the runner is its only caller.
    "analysis.energy": 3,
}

# Packages whose __init__ is a re-export shim. They are excluded rather than
# layered because a shim's imports describe its children, not a dependency of
# its own, and layering them would force every child up to the parent's level.
_SHIMS = {
    "",
    "forcefield",
    "sampling",
    "space",
    "system",
    "proposals",
    "engine",
    "run",
    "analysis",
    "framework",
}


# Inversions that exist TODAY and are scheduled to be removed. Listed
# explicitly, with the stage that removes each, so the debt is enumerated rather
# than hidden: a NEW inversion still fails this test, and an entry that stops
# being needed fails `test_no_stale_allowances` below.
#
# WHY THESE FOUR. Three of them are one finding: `max_core_floor` and the
# overlap predicate live in `geometry` (L2) but are functions of sigma, which
# makes them FORCE FIELD concepts (L1). restructure_plan.md section 5 placed
# `overlap.py` at L2; that was wrong, and this test is what caught it. The fix
# at S2 is to move both down to L1, after which L2 computes distances, L1 owns
# the floor, and L3 applies the test -- no module needs to reach up.
#
# The fourth is the P4 extraction still importing from the sampler it was
# extracted from. The shared pieces move down into `channel_move`; `channel`
# then imports them, which is the direction P4 intended.
#
# The table below holds three of them now: the two `max_core_floor`/overlap
# entries went away at S2 when those functions moved down to L1, which is what
# an allowance is supposed to do, and the cube-cavity sweep joined the factory
# pair it belongs with.
KNOWN_INVERSIONS: dict[tuple[str, str], str] = {
    # What remains after S2 moved the body GEOMETRY down to `body` (L2): the
    # sweep CLASSES. `ChannelBody.sweep()` is a factory picking one of four
    # sweep classes, which is a layer-5 decision living in a layer-4 module.
    #
    # Deferred to S5 on purpose, not overlooked. R7 collapses those four classes
    # (six, counting the multi-species pair) into ONE dynamic program selected by
    # a mask policy. After that, `sweep()` needs no sampler class at all -- it
    # hands a policy object down. Resolving the edge now would build a seam that
    # the collapse then removes.
    (
        "proposals.channel_move",
        "sampling.channel",
    ): "S5: R7 collapses the sweeps; the factory goes with them",
    # THE THIRD MEMBER OF THE SAME FINDING, not a new one. `ChannelBody.sweep()`
    # picks `CavitySweep` for the cube geometry, where the transfer matrix it
    # would otherwise build has an eviction window the size of the pool and a
    # 2^M state count. Same factory, same layer-5 decision in a layer-4 module,
    # and the same resolution: R7 collapses every sweep into one dynamic program
    # selected by a mask policy, after which `sweep()` hands a policy object
    # down and imports no sampler class at all.
    #
    # The import is DEFERRED (inside the method) rather than at module scope,
    # which keeps the cycle out of import time but is still a runtime edge, and
    # this test is right to count it as one.
    (
        "proposals.channel_move",
        "sampling.branch",
    ): "S5: R7 collapses the sweeps; the cube-cavity factory goes with them",
}


def _module_name(path: Path) -> str:
    """Dotted name of a module relative to the package root."""
    rel = path.relative_to(_PKG).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _in_package_imports(path: Path, name: str) -> set[str]:
    """Every in-package module this module imports AT RUNTIME.

    `if TYPE_CHECKING:` blocks are skipped: a forward-referenced annotation
    creates no runtime dependency and no cycle, and forbidding it would push
    people toward untyped signatures, which is a worse outcome than a
    nominally-backwards type import.
    """
    tree = ast.parse(path.read_text())

    # Collect the line ranges of TYPE_CHECKING blocks so their imports can be
    # ignored without re-walking the tree.
    skip: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_tc:
            skip.append((node.lineno, node.end_lineno or node.lineno))

    def skipped(lineno: int) -> bool:
        return any(a <= lineno <= b for a, b in skip)

    own_parts = name.split(".")
    found: set[str] = set()

    for node in ast.walk(tree):
        if skipped(getattr(node, "lineno", 0)):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = own_parts[: len(own_parts) - node.level]
                target = ".".join([*base, node.module] if node.module else base)
                if node.module:
                    found.add(target)
                else:
                    # `from . import x` -- each name is a sibling module
                    found.update(".".join([*base, a.name]) for a in node.names)
            elif node.module and node.module.startswith("gcmc"):
                found.add(node.module[len("gcmc") :].lstrip("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gcmc."):
                    found.add(alias.name[len("gcmc.") :])

    return {f for f in found if f and f not in _SHIMS}


def _modules() -> list[tuple[str, Path]]:
    """Every module in the package, excluding re-export shims."""
    out = []
    for path in sorted(_PKG.rglob("*.py")):
        name = _module_name(path)
        if name in _SHIMS:
            continue
        out.append((name, path))
    return out


def test_every_module_is_on_the_ladder() -> None:
    """A new module must be placed on the ladder deliberately.

    Defaulting an unknown module to some layer would let new code slot itself in
    anywhere, which is exactly the drift R12 exists to stop.
    """
    unplaced = [name for name, _ in _modules() if name not in LAYERS]
    assert not unplaced, (
        f"modules not on the ladder: {unplaced}\n"
        "Add each to LAYERS in this file, at the layer its CONTENT belongs to. "
        "See restructure_plan.md section 5 for what each layer means."
    )


def test_the_ladder_has_no_dead_entries() -> None:
    """A merged-away module must leave the ladder with its code.

    Without this, `LAYERS` accumulates names of files that no longer exist and
    stops being a readable description of the architecture.
    """
    real = {name for name, _ in _modules()}
    stale = sorted(set(LAYERS) - real)
    assert not stale, (
        f"LAYERS names modules that do not exist: {stale}\n"
        "Delete them; the ladder must describe the tree as it is."
    )


def test_imports_only_go_down() -> None:
    """No module imports one at a strictly higher layer."""
    violations: list[str] = []
    for name, path in _modules():
        if name not in LAYERS:
            continue  # reported by test_every_module_is_on_the_ladder
        here = LAYERS[name]
        for target in sorted(_in_package_imports(path, name)):
            there = LAYERS.get(target)
            if there is None:
                continue  # reported by test_every_module_is_on_the_ladder
            if there > here:
                if (name, target) in KNOWN_INVERSIONS:
                    continue
                violations.append(f"  L{here} {name}  ->  L{there} {target}")

    assert not violations, (
        "imports going UP the ladder:\n"
        + "\n".join(violations)
        + "\n\nMove the shared thing down to a layer both can see, or -- if the "
        "two really belong together -- change LAYERS and say why in the commit."
    )


def test_no_stale_allowances() -> None:
    """Every allowed inversion must still exist.

    An allowlist nobody prunes stops being a list of debt and becomes a list of
    permissions. When a merge removes an inversion, this fails and the entry
    goes with it -- so the allowlist can only ever shrink.
    """
    live: set[tuple[str, str]] = set()
    for name, path in _modules():
        if name not in LAYERS:
            continue
        for target in _in_package_imports(path, name):
            if LAYERS.get(target, -1) > LAYERS[name]:
                live.add((name, target))

    stale = sorted(set(KNOWN_INVERSIONS) - live)
    assert not stale, (
        f"KNOWN_INVERSIONS lists inversions that no longer exist: {stale}\n"
        "Delete the entries; the debt is paid."
    )


@pytest.mark.parametrize(
    "name,path", _modules(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_layer_zero_depends_only_on_layer_zero(name: str, path: Path) -> None:
    """L0 is the vocabulary everything shares, so it reaches nothing above it.

    This said "depends on nothing" while L0 held exactly one module, and the two
    readings were indistinguishable. They stopped being so when `config_fields`
    and `config_specs` moved down: `config_specs` imports `config_fields`, which
    is an L0 module, and forbidding that would mean either merging two files
    that say different things or pushing the sub-configs back above the engines
    that need them.

    So the rule is stated as what it always meant -- nothing above L0 -- and the
    guarantee it was really protecting, that L0 cannot be part of a cycle, is
    asserted directly by `test_layer_zero_is_acyclic` rather than approximated
    by a blanket ban. Strictly more is checked than before, not less.

    Parametrised per module so a violation names the file directly rather than
    the whole layer.
    """
    if LAYERS.get(name) != 0:
        pytest.skip("not a layer-0 module")
    above = sorted(d for d in _in_package_imports(path, name) if LAYERS.get(d, 99) > 0)
    assert not above, (
        f"{name} is L0 but imports {above}, which are not.\n"
        "L0 is what every other layer is allowed to speak; a dependency upward "
        "from it makes that impossible for whoever sits between."
    )


def test_layer_zero_is_acyclic() -> None:
    """No cycle among the L0 modules.

    The guarantee the old blanket ban gave for free, now that L0 modules may
    import each other. A cycle here would be the worst place to have one: L0 is
    imported by everything, so an import-order-dependent failure would surface
    as an unrelated module failing to load.
    """
    zero = {n: p for n, p in _modules() if LAYERS.get(n) == 0}
    edges = {n: _in_package_imports(p, n) & set(zero) for n, p in zero.items()}

    state: dict[str, int] = {}

    def walk(node: str, trail: list[str]) -> list[str] | None:
        if state.get(node) == 1:
            return [*trail, node]  # back edge: the trail IS the cycle
        if state.get(node) == 2:
            return None
        state[node] = 1
        for nxt in sorted(edges[node]):
            found = walk(nxt, [*trail, node])
            if found:
                return found
        state[node] = 2
        return None

    for start in sorted(edges):
        cycle = walk(start, [])
        assert cycle is None, f"import cycle inside L0: {' -> '.join(cycle)}"
