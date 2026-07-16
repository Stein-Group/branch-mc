"""Command line entry point.

    python -m gcmc run params.json [--root runs]

One command runs or resumes a single simulation. It is idempotent: pointed at a
finished run it returns at once, and pointed at an interrupted one it continues
from the checkpoint. This is the command each single-core sweep job invokes and
the one a parallel job runs directly.
"""

from __future__ import annotations

import argparse
import logging

from . import runner


def main(argv: list[str] | None = None) -> int:
    """Parse the CLI (currently the single `run` subcommand) and dispatch to
    runner.run; prints the terminal step/N/directory line the sweep scripts
    grep for."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="python -m gcmc")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume one simulation from params.json")
    r.add_argument("params", help="path to params.json")
    r.add_argument("--root", default="runs", help="directory holding run folders")

    args = p.parse_args(argv)
    if args.cmd == "run":
        state, paths = runner.run(args.params, root=args.root)
        # Compatibility contract: this exact "done: ..." line goes to stdout;
        # cluster sweep scripts grep it. Keep it a print, not a log record.
        print(f"done: step={state.step} N={len(state.positions)} dir={paths.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
