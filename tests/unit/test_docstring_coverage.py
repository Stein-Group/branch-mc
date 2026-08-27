"""R14: a function without a contract does not ship.

Gate S2 asks for 100% docstring coverage on public functions, and the tier-0 row
in the CI table lists "docstring coverage" beside ruff, mypy and the ladder. The
check itself was never written, so the requirement had no teeth: coverage sat at
a measured 96% and nothing said which four were missing or whether it was drifting
up or down.

WHAT COUNTS AS PUBLIC, and why the answer is not "every def". A closure inside a
function is an implementation detail of that function -- `rand`, `alive_push`,
`to_global` -- and its contract is the enclosing docstring. Demanding one for each
would push people to write "helper for the loop above" a hundred times, which is
noise that makes the real contracts harder to find. So the surface is
MODULE-LEVEL and CLASS-LEVEL definitions whose names do not start with an
underscore: what another module can import and call.

Underscore-private definitions are exempt from the count but NOT from the
practice: most of the ones in this package carry long docstrings, because a
private function with a subtle invariant is exactly where a contract earns its
keep. The exemption is about what a gate can sensibly demand, not about what is
worth writing.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "gcmc"


def _public_defs() -> list[tuple[str, str, bool]]:
    """(module, name, has_docstring) for every public API definition."""
    out: list[tuple[str, str, bool]] = []
    for path in sorted(_PKG.rglob("*.py")):
        rel = str(path.relative_to(_PKG.parent))
        tree = ast.parse(path.read_text())

        def visit(node: ast.AST, rel: str = rel) -> None:
            for child in getattr(node, "body", []):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if not child.name.startswith("_"):
                        out.append(
                            (rel, child.name, ast.get_docstring(child) is not None)
                        )
                    if isinstance(child, ast.ClassDef):
                        visit(child, rel)

        visit(tree)
    return out


def test_every_public_definition_has_a_contract() -> None:
    """100%, per Gate S2. Not a ratchet: the target is the whole surface."""
    defs = _public_defs()
    missing = [f"{m}:{n}" for m, n, ok in defs if not ok]
    assert not missing, (
        f"{len(missing)} of {len(defs)} public definitions have no docstring:\n  "
        + "\n  ".join(missing)
        + "\n\nR14: a function without a contract does not ship. If the thing is "
        "genuinely internal, give it a leading underscore -- that is a real "
        "statement about its surface, and it is exempt from the count. Adding an "
        "empty or restating docstring to pass this is worse than failing it."
    )


def test_module_docstrings_exist() -> None:
    """Every module says what it is for.

    Checked separately from the per-definition rule because a module docstring
    does different work: it is where the layer, the units convention and the
    reason the file exists at all are stated, and it is the first thing anyone
    reading the tree sees.
    """
    missing = []
    for path in sorted(_PKG.rglob("*.py")):
        if path.name == "__init__.py" and path.stat().st_size < 200:
            continue  # a tiny re-export shim documents itself by being one
        if not ast.get_docstring(ast.parse(path.read_text())):
            missing.append(str(path.relative_to(_PKG.parent)))
    assert not missing, "modules with no docstring:\n  " + "\n  ".join(missing)


def test_the_units_convention_is_stated_where_numbers_have_units() -> None:
    """R14 asks for Units alongside Invariants, and this is the honest state of it.

    The package is entirely in engine-reduced units with ONE conversion boundary
    (R10), so the convention is a property of the tree rather than of each
    function, and repeating "reduced units" 300 times would be noise. What must be
    true is that the modules carrying dimensional quantities SAY so somewhere in
    their docstring, and that is what this asserts.

    Stated as a floor rather than a total: the modules listed are the ones where a
    reader could plausibly assume Angstroms or Kelvin and be wrong. Widening the
    list is welcome; narrowing it needs a reason.
    """
    must_state_units = [
        "gcmc/potentials/pair_table.py",
        "gcmc/potentials/lennard_jones.py",
        "gcmc/species.py",
        "gcmc/species_file.py",
        "gcmc/config_specs.py",
    ]
    silent = []
    for rel in must_state_units:
        text = (_PKG.parent / rel).read_text()
        doc = ast.get_docstring(ast.parse(text)) or ""
        if "reduced" not in doc.lower() and "unit" not in doc.lower():
            silent.append(rel)
    assert not silent, (
        "these carry dimensional quantities but their module docstring does not "
        "state the units convention:\n  " + "\n  ".join(silent)
    )
