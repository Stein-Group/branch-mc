"""Command-line options for the generated feature matrix.

The fixtures that used to live here (`matrix_dir`, `equilibrated_library`) moved
up to `tests/conftest.py` at S2: a conftest is visible only to its own directory
and below, and the sub-config round-trip test in `tests/unit` needs the same
generated QA matrices this tier does. One generator, one library of obstacles.
"""

from __future__ import annotations

import json

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Options that switch the matrix between its per-push and nightly shapes."""
    parser.addoption(
        "--matrix-full",
        action="store_true",
        default=False,
        help="Run every accepted point (tier 3) rather than the pairwise cover (tier 2).",
    )
    parser.addoption(
        "--matrix-regen",
        action="store_true",
        default=False,
        help="Rewrite digests.json and expected.json from the current code. "
        "Deliberate act: the diff is the review.",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Under `--matrix-regen`, rewrite the two committed files.

    THIS DID NOT EXIST. The option was declared in `conftest.py`, its help text
    described what it would rewrite, and four separate failure messages told the
    reader to run it -- and nothing anywhere read `config.getoption`. So the
    documented way to re-baseline a digest was inert, and the only way to move
    one was to hand-edit JSON, which is the habit every comment in this file is
    trying to prevent.

    `expected.json` was in the same state from the other direction: committed,
    described in `_enumerate` as the file that makes a validator change show up
    as a diff, with `load_expected` and `dump_expected` written for the purpose
    and called by nothing. It is compared now, in
    `test_the_committed_partition_is_still_the_live_one`, and regenerated here.

    Written only when regenerating, so an ordinary run never touches the tree.
    """
    if not session.config.getoption("--matrix-regen"):
        return
    # Imported here, not at module scope: importing the test module from its
    # own package's conftest at collection time is a circular import.
    from . import _enumerate as E
    from . import test_smoke as TS

    # Digests: MERGED, not replaced. A regen run under the default pairwise
    # cover visits only the cover, and replacing the file would silently drop
    # every pinned id outside it -- turning a re-baseline into a coverage cut.
    if TS._REGEN:
        merged = {**TS._load_digests(), **TS._REGEN}
        with TS._DIGESTS.open("w") as fh:
            json.dump(dict(sorted(merged.items())), fh, indent=2)
            fh.write("\n")
    # The partition is enumerated in full whatever the run shape, so it is
    # replaced rather than merged: a point that stopped existing must leave.
    expected = E.dump_expected(E.enumerate_points())
    with E._EXPECTED.open("w") as fh:
        json.dump(expected, fh, indent=1, sort_keys=True)
        fh.write("\n")
