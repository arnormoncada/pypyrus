from __future__ import annotations

import gzip
import json
from typing import Any

from pypyrus.storage.sqlite_store import SQLiteStore


def decode_sample_ids_blob(blob: bytes | None) -> list[str] | None:
    """
    Decode a gzip-compressed sample_ids blob back into a Python list.

    Returns None if no blob is present.
    """
    if blob is None:
        return None

    raw = gzip.decompress(blob)
    return json.loads(raw.decode("utf-8"))


def get_batches_for_run(
    store: SQLiteStore,
    run_id: str,
    include_sample_ids: bool = True,
) -> list[dict[str, Any]]:
    """
    Return all batch_delivered rows for a run, ordered by global_step.

    If include_sample_ids=True, sample_ids_blob is decoded into sample_ids.
    """
    rows = store.get_events(run_id=run_id, event_type="batch_delivered")

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)

        if include_sample_ids:
            item["sample_ids"] = decode_sample_ids_blob(item.get("sample_ids_blob"))

        results.append(item)

    return results