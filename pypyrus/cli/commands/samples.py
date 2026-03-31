from __future__ import annotations

from pypyrus.cli.render import format_json, render_sample_find
from pypyrus.cli.store import open_query_store
from pypyrus.reporting import find_sample_occurrences, resolve_file_query_for_run


def cmd_samples_find(args) -> int:
    direct = args.sample_id is not None
    by_file = args.file is not None
    if direct == by_file:
        raise ValueError("Pass exactly one of --sample-id or --file.")
    if by_file and not args.dataset_path:
        raise ValueError("--dataset-path is required when using --file.")
    if by_file and args.dataset_id:
        raise ValueError("--dataset-id cannot be combined with --file.")

    with open_query_store(args.db) as store:
        if args.sample_id is not None:
            sample_id = args.sample_id
            resolution = None
            dataset_ids = [args.dataset_id] if args.dataset_id else None
        else:
            resolution = resolve_file_query_for_run(
                store,
                args.run_id,
                dataset_path=args.dataset_path,
                file_path=args.file,
            )
            sample_id = resolution["sample_id"]
            dataset_ids = resolution.get("matching_dataset_ids")

        result = find_sample_occurrences(
            store,
            args.run_id,
            sample_id,
            dataset_ids=dataset_ids,
        )

    if resolution is not None:
        result["query_resolution"] = resolution

    if args.json:
        print(format_json(result))
    else:
        print(render_sample_find(result))
    return 0
