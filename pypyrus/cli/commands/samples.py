from __future__ import annotations

from pypyrus.cli.render import format_json, render_sample_find
from pypyrus.cli.store import open_query_store
from pypyrus.reporting import find_sample_occurrences


def cmd_samples_find(args) -> int:
    if not args.sample_id:
        raise ValueError("--sample-id is required.")

    with open_query_store(args.db) as store:
        result = find_sample_occurrences(
            store,
            args.run_id,
            args.sample_id,
        )

    if args.json:
        print(format_json(result))
    else:
        print(render_sample_find(result))
    return 0
