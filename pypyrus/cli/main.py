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
        description=(
            "Inspect PyPyrus provenance runs from the command line.\n\n"
            "Use this CLI to list runs, inspect one run in detail, compare two runs,\n"
            "or inspect the exact batch delivered at a specific step."
        ),
        epilog=(
            "Examples:\n"
            "  pypyrus runs list\n"
            "  pypyrus runs show <run_id>\n"
            "  pypyrus compare <run_a> <run_b>\n"
            "  pypyrus batches show <run_id> --step 12\n"
            "  pypyrus --json runs show <run_id>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        help=(
            "Path to the PyPyrus SQLite database. "
            "Defaults to $PYPYRUS_DB or ./pypyrus.db."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of human-readable terminal output.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{runs,compare,batches}",
    )

    runs_parser = subparsers.add_parser(
        "runs",
        help="List runs or inspect one run in detail.",
        description=(
            "Inspect recorded runs.\n\n"
            "Use 'runs list' to find runs worth inspecting, then 'runs show' to\n"
            "see datasets, loaders, transforms, environment details, and batch counts."
        ),
        epilog=(
            "Examples:\n"
            "  pypyrus runs list\n"
            "  pypyrus runs show <run_id>\n"
            "  pypyrus --json runs show <run_id>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")

    runs_list_parser = runs_subparsers.add_parser(
        "list",
        help="List recorded runs.",
        description=(
            "List recorded runs in reverse chronological order.\n\n"
            "This is the default entry point for finding runs to inspect or compare."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runs_list_parser.set_defaults(handler=cmd_runs_list)

    runs_show_parser = runs_subparsers.add_parser(
        "show",
        help="Show one recorded run.",
        description=(
            "Show a run overview with its datasets, loaders, transforms,\n"
            "environment summary, and batch counts."
        ),
        epilog=(
            "Example:\n"
            "  pypyrus runs show <run_id>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runs_show_parser.add_argument("run_id", help="Run identifier to inspect.")
    runs_show_parser.set_defaults(handler=cmd_runs_show)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two runs.",
        description=(
            "Compare two runs at the dataset and batch-stream level.\n\n"
            "The comparison is role-aware: train batches are compared to train,\n"
            "val to val, and so on."
        ),
        epilog=(
            "Examples:\n"
            "  pypyrus compare <run_a> <run_b>\n"
            "  pypyrus --json compare <run_a> <run_b>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument("run_id_a", help="Baseline run ID.")
    compare_parser.add_argument("run_id_b", help="Candidate run ID.")
    compare_parser.set_defaults(handler=cmd_compare)

    batches_parser = subparsers.add_parser(
        "batches",
        help="Inspect delivered batches.",
        description=(
            "Inspect the exact batches delivered during a run.\n\n"
            "Use this when compare output points to a divergence and you want the\n"
            "concrete batch identity at a specific step."
        ),
        epilog=(
            "Example:\n"
            "  pypyrus batches show <run_id> --step 12"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batches_subparsers = batches_parser.add_subparsers(dest="batches_command")

    batches_show_parser = batches_subparsers.add_parser(
        "show",
        help="Show one batch for a run/global batch step.",
        description=(
            "Show one delivered batch for a given run and step.\n\n"
            "The step is the run-global batch position (`global_sequence`),\n"
            "so each step identifies at most one batch within a run."
        ),
        epilog=(
            "Examples:\n"
            "  pypyrus batches show <run_id> --step 3\n"
            "  pypyrus --json batches show <run_id> --step 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batches_show_parser.add_argument("run_id", help="Run identifier.")
    batches_show_parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Run-global batch position (`global_sequence`) to inspect.",
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
