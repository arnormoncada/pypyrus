from __future__ import annotations

from pypyrus.cli.render import format_json, render_batch
from pypyrus.cli.store import open_query_store
from pypyrus.reporting import get_batch_for_run_step


def cmd_batches_show(args) -> int:
    with open_query_store(args.db) as store:
        batch = get_batch_for_run_step(
            store,
            args.run_id,
            args.step,
            role=args.role,
            dataset_id=args.dataset_id,
            loader_id=args.loader_id,
            include_sample_ids=not args.no_sample_ids,
        )

    if batch is None:
        raise ValueError(
            f"Batch not found for run_id={args.run_id}, global_step={args.step}"
        )

    if args.json:
        print(format_json(batch))
    else:
        print(render_batch(batch))
    return 0
