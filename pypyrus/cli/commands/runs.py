from __future__ import annotations

from pypyrus.cli.render import format_json, render_run_overview, render_runs_table
from pypyrus.cli.store import open_query_store
from pypyrus.reporting import build_run_overview, list_run_summaries


def cmd_runs_list(args) -> int:
    with open_query_store(args.db) as store:
        runs = list_run_summaries(store)

    if args.json:
        print(format_json(runs))
    else:
        print(render_runs_table(runs))
    return 0


def cmd_runs_show(args) -> int:
    with open_query_store(args.db) as store:
        overview = build_run_overview(store, args.run_id)

    if overview is None:
        raise ValueError(f"Run not found: {args.run_id}")

    if args.json:
        print(format_json(overview))
    else:
        print(render_run_overview(overview))
    return 0
