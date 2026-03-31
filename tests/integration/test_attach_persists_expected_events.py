from __future__ import annotations

import json

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run

from tests.helpers import (
    ComposeLike,
    OffsetFeatures,
    ScaleFeatures,
    TinyMapDataset,
    fetch_all,
    fetch_one,
)


def test_attached_loader_persists_run_dataset_transform_and_batch_events(
    db_path,
    store,
) -> None:
    transform = ComposeLike([ScaleFeatures(2.0), OffsetFeatures(1.5)])
    dataset = TinyMapDataset(n=10, transform=transform)
    loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        attached_loader = attach(loader, run, role="train")
        consumed_batches = list(attached_loader)

    run_row = fetch_one(
        db_path,
        "SELECT run_id, status, start_time, end_time FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["status"] == "success"
    assert run_row["start_time"] is not None
    assert run_row["end_time"] is not None

    environment_rows = fetch_all(
        db_path,
        "SELECT python_version FROM environment_snapshot WHERE run_id = ?",
        (run.run_id,),
    )
    assert len(environment_rows) == 1
    assert environment_rows[0]["python_version"]

    dataset_row = fetch_one(
        db_path,
        """
        SELECT d.dataset_id, d.fingerprint, d.fingerprint_method,
               d.sample_id_scheme, d.sample_id_resolver, rd.role
        FROM datasets d
        JOIN run_datasets rd ON rd.dataset_id = d.dataset_id
        WHERE rd.run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["role"] == "train"
    assert dataset_row["fingerprint"]
    assert dataset_row["fingerprint_method"] == "in_memory_deterministic_v1"
    assert dataset_row["sample_id_scheme"] == "index"
    assert dataset_row["sample_id_resolver"] == "fallback_index"

    transform_row = fetch_one(
        db_path,
        """
        SELECT transform_list_json, introspection_level
        FROM transform_declared
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    transform_list = json.loads(transform_row["transform_list_json"])
    assert [item["name"] for item in transform_list] == [
        "ScaleFeatures",
        "OffsetFeatures",
    ]
    assert transform_row["introspection_level"] == "full"

    loader_row = fetch_one(
        db_path,
        """
        SELECT loader_id, dataset_id, role
        FROM loaders
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert loader_row["dataset_id"] == dataset_row["dataset_id"]
    assert loader_row["role"] == "train"
    assert loader_row["loader_id"]

    batch_rows = fetch_all(
        db_path,
        """
        SELECT loader_id, global_step, global_sequence, batch_size, sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    assert len(batch_rows) == len(consumed_batches) == 4
    assert all(row["loader_id"] == loader_row["loader_id"] for row in batch_rows)
    assert [row["global_step"] for row in batch_rows] == [0, 1, 2, 3]
    assert [row["global_sequence"] for row in batch_rows] == [0, 1, 2, 3]
    assert [row["batch_size"] for row in batch_rows] == [3, 3, 3, 1]
    assert all(row["sample_ids_blob"] is not None for row in batch_rows)
