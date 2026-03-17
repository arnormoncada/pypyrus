from __future__ import annotations

from pypyrus.cli.render import format_json
from pypyrus.cli.store import open_query_store
from pypyrus.reporting import compare_runs, format_run_comparison


def cmd_compare(args) -> int:
    with open_query_store(args.db) as store:
        result = compare_runs(store, args.run_id_a, args.run_id_b)

    if args.json:
        print(format_json(result))
    else:
        print(format_run_comparison(result))
    return 0
