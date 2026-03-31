"""
High-level query helpers for the PyPyrus reporting layer.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pypyrus.core.dataset_identity import fingerprint_local_path
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


def list_run_summaries(store: Store) -> list[dict[str, Any]]:
    """Return CLI-friendly run summaries ordered by start time."""
    summaries: list[dict[str, Any]] = []
    for run_id in store.list_runs():
        run = get_run(store, run_id)
        if run is None:
            continue

        datasets = get_datasets_for_run(store, run_id)
        loaders = get_loaders_for_run(store, run_id)
        batches = get_batches_for_run(store, run_id, include_sample_ids=False)

        roles = sorted(
            {
                str(role)
                for role in (
                    loader.get("role") for loader in loaders
                )
                if role
            }
        )

        summary = dict(run)
        summary["duration_seconds"] = _compute_duration_seconds(
            run.get("start_time"),
            run.get("end_time"),
        )
        summary["dataset_count"] = len(datasets)
        summary["loader_count"] = len(loaders)
        summary["batch_count"] = len(batches)
        summary["roles"] = roles
        summaries.append(summary)
    return summaries


def _compute_duration_seconds(
    start_time: Any,
    end_time: Any,
) -> float | None:
    """Return run duration in seconds when both timestamps are present."""
    if not start_time or not end_time:
        return None

    try:
        start = datetime.fromisoformat(str(start_time))
        end = datetime.fromisoformat(str(end_time))
    except ValueError:
        return None

    return max((end - start).total_seconds(), 0.0)


def get_datasets_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return dataset descriptors registered during a run."""
    return store.get_events(run_id, event_type="dataset_registered")


def get_loaders_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return loader registrations recorded during a run."""
    return store.get_events(run_id, event_type="loader_registered")


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
        Optional filter — only return batches emitted by loaders with this role.
    """
    rows = store.get_events(run_id=run_id, event_type="batch_delivered")

    if role is not None:
        rows = [r for r in rows if r.get("role") == role]

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
    global_sequence: int,
    *,
    include_sample_ids: bool = True,
) -> dict[str, Any] | None:
    """
    Return one batch for (run_id, global_sequence).

    This is the run-global batch position used by the CLI batch-inspection path.
    """
    rows = get_batches_for_run(store, run_id, include_sample_ids=include_sample_ids)
    for row in rows:
        if row.get("global_sequence") == global_sequence:
            return row
    return None


def get_environment_for_run(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return environment snapshots recorded during a run."""
    return store.get_events(run_id, event_type="environment_snapshot")


def build_run_overview(store: Store, run_id: str) -> dict[str, Any] | None:
    """
    Return a CLI-friendly overview for a single run.

    This combines run metadata with related dataset, transform, environment,
    and batch summary information.
    """
    run = get_run(store, run_id)
    if run is None:
        return None

    run = dict(run)
    run["duration_seconds"] = _compute_duration_seconds(
        run.get("start_time"),
        run.get("end_time"),
    )

    datasets = get_datasets_for_run(store, run_id)
    loaders = get_loaders_for_run(store, run_id)
    transforms = get_transforms_for_run(store, run_id)
    environments = get_environment_for_run(store, run_id)
    batches = get_batches_for_run(store, run_id, include_sample_ids=False)

    batches_by_role: dict[str, int] = {}
    for loader in loaders:
        role = loader.get("role")
        if not role:
            continue
        batches_by_role.setdefault(role, 0)
    for batch in batches:
        role = batch.get("role")
        if role:
            batches_by_role[role] = batches_by_role.get(role, 0) + 1

    return {
        "run": run,
        "datasets": datasets,
        "loaders": loaders,
        "transforms": transforms,
        "environment": environments,
        "batch_count": len(batches),
        "batches_by_role": batches_by_role,
    }


def find_sample_occurrences(
    store: Store,
    run_id: str,
    sample_id: str,
    *,
    dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    batches = get_batches_for_run(store, run_id, include_sample_ids=True)
    occurrences: list[dict[str, Any]] = []
    dataset_scope = {str(item) for item in dataset_ids or []}

    for batch in batches:
        batch_dataset_id = batch.get("dataset_id")
        if dataset_scope and str(batch_dataset_id) not in dataset_scope:
            continue
        sample_ids = batch.get("sample_ids") or []
        normalized_ids = [str(item) for item in sample_ids]
        if sample_id not in normalized_ids:
            continue
        occurrences.append(
            {
                "run_id": batch.get("run_id"),
                "dataset_id": batch.get("dataset_id"),
                "loader_id": batch.get("loader_id"),
                "role": batch.get("role"),
                "global_step": batch.get("global_step"),
                "global_sequence": batch.get("global_sequence"),
                "timestamp": batch.get("timestamp"),
                "batch_fingerprint": batch.get("batch_fingerprint"),
            }
        )

    first_occurrence = occurrences[0] if occurrences else None
    last_occurrence = occurrences[-1] if occurrences else None

    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "sample_id_scheme": _sample_id_scheme(sample_id),
        "query_scope": "dataset" if dataset_scope else "run",
        "scoped_dataset_ids": sorted(dataset_scope),
        "found": bool(occurrences),
        "occurrence_count": len(occurrences),
        "first_occurrence": first_occurrence,
        "last_occurrence": last_occurrence,
        "matching_steps": [item["global_sequence"] for item in occurrences],
        "matching_roles": sorted({str(item["role"]) for item in occurrences if item.get("role")}),
        "matching_loader_ids": sorted(
            {str(item["loader_id"]) for item in occurrences if item.get("loader_id")}
        ),
        "matching_dataset_ids": sorted(
            {str(item["dataset_id"]) for item in occurrences if item.get("dataset_id")}
        ),
        "occurrences": occurrences,
    }


def resolve_file_query_for_run(
    store: Store,
    run_id: str,
    *,
    dataset_path: str | Path,
    file_path: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    target_path = Path(file_path).expanduser()
    if not target_path.is_absolute():
        candidate_path = dataset_root / target_path
        try:
            relative_path = candidate_path.relative_to(dataset_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"File path {candidate_path} is not inside dataset path {dataset_root}"
            ) from exc
        target_path = candidate_path
    else:
        target_path = target_path.resolve(strict=False)
        try:
            relative_path = target_path.relative_to(dataset_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"File path {target_path} is not inside dataset path {dataset_root}"
            ) from exc

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not target_path.exists():
        raise FileNotFoundError(f"File path not found: {target_path}")

    fingerprint = fingerprint_local_path(dataset_root)
    datasets = get_datasets_for_run(store, run_id)
    matching_datasets = [
        dataset
        for dataset in datasets
        if dataset.get("sample_id_scheme") == "filepath"
        and dataset.get("fingerprint") == fingerprint.fingerprint
    ]

    if not matching_datasets:
        raise ValueError(
            "No file-collection dataset in this run matches the provided dataset path fingerprint."
        )

    return {
        "sample_id": f"filepath:{relative_path}",
        "sample_id_scheme": "filepath",
        "sample_id_resolver": "file_collection",
        "query_scope": "dataset",
        "dataset_fingerprint": fingerprint.fingerprint,
        "dataset_fingerprint_method": fingerprint.fingerprint_method,
        "matching_dataset_ids": [
            str(dataset["dataset_id"])
            for dataset in matching_datasets
            if dataset.get("dataset_id") is not None
        ],
        "matching_datasets": matching_datasets,
    }


def _sample_id_scheme(sample_id: str) -> str:
    if ":" not in sample_id:
        return "custom"
    return sample_id.split(":", 1)[0]
