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

NOTE ON THE CURRENT TREE. The layer assignments below describe the TARGET
architecture (restructure_plan.md section 5) mapped onto today's filenames, so
the test is useful before the S2 restructure rather than after it. Where a
module is scheduled to be merged away (the `mof_*` family, the duplicate
potentials), it is placed at the layer its content belongs to, not the layer its
filename suggests.
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
    # it: `MC_engine` (L6) taking a `KernelSpec` was an upward import, and so was
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
    "potentials.base": 1,
    "potentials.pair_table": 1,
    "potentials.lennard_jones": 1,
    "potentials.lj_table": 1,
    "potentials.hard_sphere": 1,
    "potentials.aux": 1,
    "potentials.overlap": 1,  # S2: the overlap convention, moved down from L2
    "species": 1,
    "species_file": 1,
    "topology": 1,
    "mof_field": 1,  # -> merges into potentials.pair_table (S5.4)
    # -- L2 space -----------------------------------------------------------
    "geometry": 2,
    "neighbors": 2,
    "body": 2,  # S2: oriented body geometry, moved down from samplers.channel
    "cell_plan": 2,
    "decomposition": 2,
    # -- L3 system ----------------------------------------------------------
    "state": 3,
    # -- L4 proposals -------------------------------------------------------
    "moves": 4,
    "trials": 4,
    "channel_move": 4,
    # -- L5 sampling --------------------------------------------------------
    "samplers.base": 5,
    "samplers.rosenbluth": 5,
    "samplers.branch": 5,
    "samplers.channel": 5,
    "samplers.channel_cyclic": 5,  # R16: shared DP, own mask policy
    "samplers.reweighting": 5,
    "mof_channel": 5,  # -> merges into channel_move (S5.7)
    # -- L6 engine ----------------------------------------------------------
    "MC_engine": 6,
    "parallel": 6,
    "mof_metropolis": 6,  # -> merges into the one metropolis loop (S5.5)
    # -- L7 run -------------------------------------------------------------
    "sim_setup": 7,
    "hash_fingerprint": 7,
    "io": 7,
    "io_framework": 7,
    "runner": 7,
    "launcher": 7,
    "migrate": 7,
    "__main__": 7,
    # -- off-ladder ---------------------------------------------------------
    "rdf": 3,
    "widom": 3,
    # Analysis over a configuration: reads positions, a species column and the
    # neighbour structure (L2), and nothing above. Sits with `rdf` because it is
    # the same kind of thing -- a correlation measured on a frame -- and the two
    # are marked in `demixing`'s docstring as an S5 merge candidate.
    "demixing": 3,
}

# Packages whose __init__ is a re-export shim. They are excluded rather than
# layered because a shim's imports describe its children, not a dependency of
# its own, and layering them would force every child up to the parent's level.
_SHIMS = {"", "potentials", "samplers"}


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
        "channel_move",
        "samplers.channel",
    ): "S5: R7 collapses the sweeps; the factory goes with them",
    (
        "channel_move",
        "samplers.channel_cyclic",
    ): "S5: R7 collapses the sweeps; the factory goes with them",
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
