from __future__ import annotations

import argparse
import sys
from typing import Sequence

from pypyrus.cli.commands.batches import cmd_batches_show
from pypyrus.cli.commands.compare import cmd_compare
from pypyrus.cli.commands.runs import cmd_runs_list, cmd_runs_show


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypyrus",
        description="Inspect and compare PyPyrus provenance runs.",
    )
    parser.add_argument(
        "--db",
        help="Path to the PyPyrus SQLite database. Defaults to $PYPYRUS_DB or ./pypyrus.db.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text.",
    )

    subparsers = parser.add_subparsers(dest="command")

    runs_parser = subparsers.add_parser("runs", help="List or inspect recorded runs.")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")

    runs_list_parser = runs_subparsers.add_parser("list", help="List recorded runs.")
    runs_list_parser.set_defaults(handler=cmd_runs_list)

    runs_show_parser = runs_subparsers.add_parser("show", help="Show one recorded run.")
    runs_show_parser.add_argument("run_id", help="Run identifier to inspect.")
    runs_show_parser.set_defaults(handler=cmd_runs_show)

    compare_parser = subparsers.add_parser("compare", help="Compare two runs.")
    compare_parser.add_argument("run_id_a", help="Baseline run ID.")
    compare_parser.add_argument("run_id_b", help="Candidate run ID.")
    compare_parser.set_defaults(handler=cmd_compare)

    batches_parser = subparsers.add_parser("batches", help="Inspect delivered batches.")
    batches_subparsers = batches_parser.add_subparsers(dest="batches_command")

    batches_show_parser = batches_subparsers.add_parser(
        "show",
        help="Show one batch for a run/global_step pair.",
    )
    batches_show_parser.add_argument("run_id", help="Run identifier.")
    batches_show_parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Per-loader global_step to inspect.",
    )
    batches_show_parser.add_argument(
        "--role",
        help="Optional role to disambiguate batches with the same global_step.",
    )
    batches_show_parser.add_argument(
        "--dataset-id",
        help="Optional dataset_id to disambiguate batches with the same global_step.",
    )
    batches_show_parser.add_argument(
        "--loader-id",
        help="Optional loader_id to disambiguate batches with the same global_step.",
    )
    batches_show_parser.add_argument(
        "--no-sample-ids",
        action="store_true",
        help="Do not decode sample IDs from the stored blob.",
    )
    batches_show_parser.set_defaults(handler=cmd_batches_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return int(handler(args) or 0)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
