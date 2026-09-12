"""Where the frozen historical artifacts live, now that they are history not tree.

R1 of restructure_plan.md builds the repository's first commits as its real
lineage, each tagged, and deletes them from the working tree at HEAD:

    v0-original   the original monolith          (was orig/,               5,116 LOC)
    v1-engine0    the first modular engine       (was tests/ref/engine0,  ~7,300 LOC)
    v2-engineA    the pre-species engine         (was tests/ref/engineA,  ~7,400 LOC)

About 19,800 lines leave the tree; every oracle stays retrievable in one command:

    git worktree add /tmp/engineA v2-engineA
    ENGINEA_ROOT=/tmp/engineA pytest -m characterisation

WHAT DOES *NOT* MOVE. `tests/ref/*.npz` (frozen trajectories) and the timing
JSONs stay in the tree. They are DATA the oracles compare against, not code, and
they are 620 KB total. The same reasoning kept `g3_analysis_data.csv` out of the
v0 tag: an artifact read at test or run time is shipped data, whatever its age.

Eleven scripts previously each computed their own `../../orig`. Eleven copies of
one path is eleven places to edit, so resolution lives here instead.

WHY A SKIP AND NOT A FAILURE. A developer running the default suite has no
worktree and does not need one -- tiers 0 to 2 do not touch the original at all.
Failing there would train people to ignore red. But a skip that says nothing is
just as bad, so `require_orig` raises a skip carrying the exact command that
would make it run.

CI does the worktree in tier 3 and sets ORIG_ROOT, so the nightly job exercises
these for real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple


class Frozen(NamedTuple):
    """One historical artifact: its tag, its env override, its legacy location.

    `legacy` is where the artifact used to sit in the tree. Keeping it in the
    resolution order means an older checkout, or a tree mid-migration, keeps
    working without anyone setting an environment variable -- the migration does
    not have to be atomic across everyone's working copies.
    """

    tag: str
    env: str
    legacy: str


ORIG = Frozen("v0-original", "ORIG_ROOT", "orig")
ENGINE0 = Frozen("v1-engine0", "ENGINE0_ROOT", "tests/ref/engine0")
ENGINEA = Frozen("v2-engineA", "ENGINEA_ROOT", "tests/ref/engineA")

#: Retained for the eleven check scripts that name it directly.
ORIG_TAG = ORIG.tag

_REPO = Path(__file__).resolve().parents[2]


def root(which: Frozen) -> Path | None:
    """The artifact's directory, or None if it is not available.

    Resolution order, most explicit first:

    1. the artifact's environment variable -- what CI sets after
       ``git worktree add``.
    2. its legacy path in the tree -- for checkouts that still carry it.

    Returns
    -------
    Path or None
        A directory that exists, or None. Never a path that does not exist:
        callers should not have to re-check.
    """
    env = os.environ.get(which.env)
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None

    legacy = _REPO / which.legacy
    return legacy if legacy.is_dir() else None


def orig_root() -> Path | None:
    """The frozen original monolith's directory, or None."""
    return root(ORIG)


def instructions(which: Frozen) -> str:
    """The exact two commands that make `which` available.

    Generated rather than written out so a message can never name one tag and
    one environment variable that do not belong together.
    """
    slug = which.tag.replace("-", "_")
    return (
        f"{which.legacy} is not on disk. It lives in tag {which.tag}, not in the "
        f"tree (R1). To make it available:\n"
        f"    git worktree add /tmp/{slug} {which.tag}\n"
        f"    {which.env}=/tmp/{slug} pytest -m characterisation"
    )


#: The original's message, named for the eleven scripts that import it directly.
INSTRUCTIONS = instructions(ORIG)


class OrigUnavailable(RuntimeError):
    """Raised when a frozen artifact is needed and absent.

    A dedicated exception rather than `pytest.skip`, because the check scripts
    run as SUBPROCESSES: a skip raised inside one is just a non-zero exit that
    the harness reports as a failure. The harness (tests/characterisation/
    test_check_scripts.py) checks `orig_root()` once and skips the whole module,
    which is the only place that can skip correctly. Running a script directly
    from a shell gets this exception's message, which is what a human wants.
    """


def require(which: Frozen) -> Path:
    """`which`'s directory, or raise with instructions.

    Returns
    -------
    Path
        The directory, guaranteed to exist.

    Raises
    ------
    OrigUnavailable
        Carrying the exact two commands that would make the caller run.
    """
    got = root(which)
    if got is None:
        raise OrigUnavailable(instructions(which))
    return got


def require_orig() -> Path:
    """The frozen original's directory, or raise with instructions."""
    return require(ORIG)


def add_to_path(which: Frozen) -> Path:
    """Put `which` on `sys.path` so its modules import by bare name.

    The check scripts do `import cav_n_d_3d_g3_LJ as orig`, and the engine
    oracles do `import gcmcA.metropolis_dynamics` -- both need the directory on the path
    rather than a package import. Idempotent: repeated calls do not grow
    `sys.path`, which matters because these modules are imported once per script
    and there are eleven of them.
    """
    got = require(which)
    entry = str(got)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return got


def add_orig_to_path() -> Path:
    """Put the frozen original on `sys.path`. See `add_to_path`."""
    return add_to_path(ORIG)


def skip_module_without(which: Frozen) -> None:
    """Skip the calling module unless `which` is on disk.

    Call this at module level, immediately after the imports. It is the only
    correct place to skip for a frozen artifact:

    * Skipping inside a test would still pay collection and import cost for a
      module whose oracle cannot exist.
    * Skipping inside a subprocess script does not work at all -- a skip there
      is just a non-zero exit the harness reports as a failure.

    The message names the artifact and the two commands that would make the
    module run, so a skipped run is actionable rather than merely quiet.
    """
    import pytest

    if root(which) is None:
        pytest.skip(instructions(which), allow_module_level=True)
