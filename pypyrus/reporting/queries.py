"""
High-level query helpers for the PyPyrus reporting layer.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

from pypyrus.storage.store import Store


def decode_sample_ids_blob(blob: bytes | None) -> list[str] | None:
    """
    Decode a gzip-compressed sample_ids blob back into a Python list.
    Returns None if no blob is present.
    """
    if blob is None:
        return None
    raw: bytes | str
    if isinstance(blob, (bytes, bytearray)):
        try:
            raw = gzip.decompress(blob)
        except (gzip.BadGzipFile, OSError):
            # Backward-compatible path for plain JSON bytes.
            raw = blob
    else:
        raw = blob
    return json.loads(raw)


def get_run(store: Store, run_id: str) -> dict[str, Any] | None:
    """Return metadata for a single run, or None if not found."""
    rows = store.get_events(run_id)
    return rows[0] if rows else None


def list_runs(store: Store) -> list[str]:
    """Return all known run IDs ordered by start time (newest first)."""
    return store.list_runs()


def get_datasets_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return dataset descriptors registered during a run."""
    return store.get_events(run_id, event_type="dataset_registered")


def get_transforms_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return transform declarations recorded during a run."""
    rows = store.get_events(run_id, event_type="transform_declared")
    for row in rows:
        raw = row.get("transform_list_json")
        if raw is not None:
            try:
                row["transform_list"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                row["transform_list"] = raw
    return rows


def get_batches_for_run(
    store: Store,
    run_id: str,
    include_sample_ids: bool = True,
    role: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return batch delivery records for a run ordered by global_sequence.

    Parameters
    ----------
    store:
        An open Store instance.
    run_id:
        The run identifier.
    include_sample_ids:
        When True, decode ``sample_ids_blob`` into ``sample_ids`` (list of str).
    role:
        Optional filter — only return batches belonging to the dataset
        with this role.  Requires a join through run_datasets.
    """
    rows = store.get_events(run_id=run_id, event_type="batch_delivered")

    if role is not None:
        role_dataset_ids = {
            d["dataset_id"]
            for d in store.get_events(run_id, event_type="dataset_registered")
            if d.get("role") == role
        }
        rows = [r for r in rows if r["dataset_id"] in role_dataset_ids]

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sample_ids"] = (
            decode_sample_ids_blob(item.get("sample_ids_blob"))
            if include_sample_ids
            else None
        )
        results.append(item)

    return results


def get_batch_for_run_step(
    store: Store,
    run_id: str,
    global_step: int,
    *,
    role: str | None = None,
    dataset_id: str | None = None,
    include_sample_ids: bool = True,
) -> dict[str, Any] | None:
    """
    Return one batch for (run_id, global_step), with optional disambiguation.

    Notes
    -----
    ``global_step`` is per-dataset/per-loader, not run-global. If a run has
    multiple datasets/loaders, multiple rows can share the same step. Use
    ``role`` or ``dataset_id`` to disambiguate.
    """
    rows = get_batches_for_run(
        store,
        run_id,
        include_sample_ids=include_sample_ids,
        role=role,
    )

    matches = [row for row in rows if row.get("global_step") == global_step]

    if dataset_id is not None:
        matches = [row for row in matches if row.get("dataset_id") == dataset_id]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            "Multiple batches match (run_id, global_step). "
            "Pass role=... or dataset_id=... to disambiguate."
        )

    return matches[0]


def get_environment_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return environment snapshots recorded during a run."""
    return store.get_events(run_id, event_type="environment_snapshot")
