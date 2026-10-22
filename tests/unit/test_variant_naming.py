"""Two rules about variants, held as gates so they cannot drift back.

Both come from the same finding. `channel.py` had six sweep classes -- {hard
core, LJ} x {single, mixture, closed-ring} -- and two problems with them:

  1. FOUR of the six overrode `__init__` and nothing else. None of those axes is
     a different algorithm; the dynamic program is one implementation and the
     axes are mask policies. Two of the four were the one-species case of the
     other two, written out again: 69% and 74% identical constructors, and
     agreeing exactly at one species.

  2. The hard-core kernel was called `ChannelSweep` while its sibling was
     `ChannelSweepLJ`. One of two peers wore the generic name, so the reader had
     to already know that the bare one meant hard cores.

A one-time cleanup fixes the instances. A test fixes the class of mistake, which
is why these are here rather than in a report.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2] / "gcmc"

_LJ = re.compile(r"(?i)(?:^|[^a-z])(?:lj|lennard)")

#: `name` mentions LJ and `bare` is a definition whose name is `name` with the
#: LJ marker removed. Listed here only when the two are NOT a variant pair, with
#: the reason -- an entry is a claim that a reviewer can check, not a mute.
_NOT_A_PAIR: dict[tuple[str, str], str] = {
    ("_lj_energy", "energy"): (
        "`energy` is FrameworkField.energy, a method on the static host; "
        "`_lj_energy` is the engine's guest-guest sum. Unrelated."
    ),
    ("lj_pair_energy", "pair_energy"): (
        "`pair_energy` is a METHOD on LennardJones and on MetropolisModel, so "
        "its class already says which potential it belongs to."
    ),
}

#: Subclasses allowed to override only `__init__`, each with the measured reason.
#: A mask policy earns a class when it shares little with its parent; when it
#: shares most of it, it should be a keyword instead.
#: EMPTY, AND THAT IS THE RESULT. The two entries here were the closed-ring
#: sweeps, allowed because they really were a different mask policy: they lifted
#: the locality guard and paid a full minimum-image pass over the pool. Leaving
#: a frozen gap in that body removes the reason -- a rod with a gap in it is an
#: open rod, swept by the parent class on local coordinates -- so the policy and
#: its two subclasses went together. The gate below is what noticed; the entries
#: did not have to be remembered.
_MASK_POLICY_SUBCLASSES: dict[str, str] = {}


def _definitions() -> dict[str, tuple[str, ast.AST]]:
    """Every module-level and class-level definition in the package, by name."""
    out: dict[str, tuple[str, ast.AST]] = {}
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                out.setdefault(node.name, (rel, node))
    return out


def test_a_potential_variant_never_wears_the_generic_name() -> None:
    """If one of two peers says LJ, the other must say hard core.

    The failure this prevents is not aesthetic. A reader who meets
    `ChannelSweep` and `ChannelSweepLJ` has to guess whether the first is the
    hard-core kernel or a base class both share -- and it was the former, which
    is the reading that makes the code look like it has a hierarchy it does not.
    """
    defs = _definitions()
    names = set(defs)
    offenders: list[str] = []
    for name in sorted(names):
        if not _LJ.search(name):
            continue
        stripped = {
            re.sub(r"(?i)_?lj_?", "", name),
            name.replace("LJ", ""),
            re.sub(r"(?i)lennardjones?", "", name),
        }
        for bare in sorted(s for s in stripped if s and s != name and s in names):
            if (name, bare) in _NOT_A_PAIR:
                continue
            if re.search(r"(?i)hard", bare):
                continue
            offenders.append(
                f"  {name} ({defs[name][0]})  vs  {bare} ({defs[bare][0]})"
            )
    assert not offenders, (
        "a variant pair where only one side names its potential:\n"
        + "\n".join(offenders)
        + "\n\nRename the bare one to say `HardCore`, or -- if the two are not a "
        "variant pair at all -- add them to _NOT_A_PAIR with the reason."
    )


def test_a_subclass_that_overrides_only_init_is_justified() -> None:
    """A subclass whose only difference is its constructor is a policy, not a type.

Sometimes that is right: a sweep that builds its masks from a different
    geometry shares almost nothing with the parent's constructor. Sometimes it
    is a copy: the two mixture sweeps were 69% and 74% the same as the
    constructors they inherited, and became a keyword. This test forces the
    distinction to be made and written down rather than assumed. All six
    classes are now two: the mixture axis became a keyword and the closed-ring
    axis went with the rod that needed it.
    """
    unjustified: list[str] = []
    for path in sorted(_PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef) or not node.bases:
                continue
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            if methods == ["__init__"] and node.name not in _MASK_POLICY_SUBCLASSES:
                unjustified.append(
                    f"  {node.name} in {path.relative_to(_PKG).as_posix()}"
                )
    assert not unjustified, (
        "subclasses overriding only __init__:\n"
        + "\n".join(unjustified)
        + "\n\nMeasure it against the parent's constructor. Mostly the same "
        "means it should be a KEYWORD on the parent, not a class; genuinely "
        "different means it belongs in _MASK_POLICY_SUBCLASSES with the number."
    )


@pytest.mark.parametrize("entry", sorted(_MASK_POLICY_SUBCLASSES))
def test_no_stale_mask_policy_allowance(entry: str) -> None:
    """An allowlist nobody prunes stops being a list of reasons and becomes a
    list of permissions. When a class is merged away, this fails and the entry
    goes with it."""
    assert entry in _definitions(), (
        f"_MASK_POLICY_SUBCLASSES names {entry}, which no longer exists. "
        "Delete the entry."
    )
