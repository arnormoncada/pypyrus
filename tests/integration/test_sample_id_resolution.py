from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.dataset_identity import resolve_dataset_identity
from pypyrus.core.run import Run
from pypyrus.instrumentation.pytorch.dataset import wrap_dataset
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import (
    TinyFileCollectionDataset,
    TinyMapDataset,
    TinyRecordsPayloadIdDataset,
    TinyRecordIdsDataset,
    TinyRecordsDataset,
    TinyRowsDataset,
    custom_sample_id_resolver,
    fetch_all,
    fetch_one,
)


def test_file_collection_dataset_persists_filepath_sample_ids(db_path, store, tmp_path) -> None:
    dataset_root = _build_file_dataset(tmp_path / "dataset")
    dataset = TinyFileCollectionDataset(dataset_root)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "filepath"
    assert dataset_row["sample_id_resolver"] == "file_collection"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["filepath:class_a/item_0.txt", "filepath:class_a/item_1.txt"],
        ["filepath:class_b/item_2.txt"],
    ]


def test_custom_sample_id_resolver_override_wins(db_path, store) -> None:
    loader = DataLoader(TinyMapDataset(n=4), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        attached = attach(
            loader,
            run,
            role="train",
            sample_id_resolver=custom_sample_id_resolver,
        )
        list(attached)

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "record_id"
    assert dataset_row["sample_id_resolver"] == "user_override"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["record_id:custom_0", "record_id:custom_1"],
        ["record_id:custom_2", "record_id:custom_3"],
    ]


def test_custom_sample_id_resolver_override_wins_over_structured_record_builtin(
    db_path, store
) -> None:
    loader = DataLoader(TinyRecordsDataset(), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        list(
            attach(
                loader,
                run,
                role="train",
                sample_id_resolver=custom_sample_id_resolver,
            )
        )

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "record_id"
    assert dataset_row["sample_id_resolver"] == "user_override"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["record_id:custom_0", "record_id:custom_1"],
        ["record_id:custom_2", "record_id:custom_3"],
    ]


def test_structured_record_dataset_uses_record_ids(db_path, store) -> None:
    loader = DataLoader(TinyRecordIdsDataset(), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        list(attach(loader, run, role="train"))

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "record_id"
    assert dataset_row["sample_id_resolver"] == "structured_record"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["record_id:alpha", "record_id:beta"],
        ["record_id:gamma", "record_id:delta"],
    ]


def test_structured_record_dataset_uses_record_object_keys(db_path, store) -> None:
    loader = DataLoader(TinyRecordsDataset(), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        list(attach(loader, run, role="train"))

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "record_id"
    assert dataset_row["sample_id_resolver"] == "structured_record"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["record_id:cust_001", "record_id:cust_002"],
        ["record_id:cust_003", "record_id:cust_004"],
    ]


def test_structured_record_dataset_without_keys_falls_back_to_row_ids(db_path, store) -> None:
    loader = DataLoader(TinyRowsDataset(), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        list(attach(loader, run, role="train"))

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "row"
    assert dataset_row["sample_id_resolver"] == "structured_record"

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]
    assert decoded == [
        ["row:0", "row:1"],
        ["row:2", "row:3"],
    ]


def test_structured_row_ids_stay_attached_to_their_samples_when_shuffled(db_path, store) -> None:
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        TinyRowsDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        consumed_batches = list(attached)

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded_ids = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]

    persisted_ids: list[str] = [sample_id for batch in decoded_ids for sample_id in batch]
    payload_row_indices: list[int] = []
    for batch in consumed_batches:
        features, _labels = batch
        payload_row_indices.extend(int(value) for value in features[:, 0].tolist())

    assert persisted_ids == [f"row:{row_index}" for row_index in payload_row_indices]
    assert payload_row_indices != [0, 1, 2, 3]


def test_structured_record_ids_stay_attached_to_their_samples_when_shuffled(db_path, store) -> None:
    dataset = TinyRecordIdsDataset()
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        consumed_batches = list(attached)

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded_ids = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]

    persisted_ids: list[str] = [sample_id for batch in decoded_ids for sample_id in batch]
    payload_row_indices: list[int] = []
    for batch in consumed_batches:
        features, _labels = batch
        payload_row_indices.extend(int(value) for value in features[:, 0].tolist())

    expected_ids = [f"record_id:{dataset.record_ids[row_index]}" for row_index in payload_row_indices]
    assert persisted_ids == expected_ids
    assert payload_row_indices != [0, 1, 2, 3]


def test_structured_record_object_keys_stay_attached_to_their_samples_when_shuffled(
    db_path, store
) -> None:
    dataset = TinyRecordsDataset()
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        consumed_batches = list(attached)

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded_ids = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]

    persisted_ids: list[str] = [sample_id for batch in decoded_ids for sample_id in batch]
    payload_row_indices: list[int] = []
    for batch in consumed_batches:
        features, _labels = batch
        payload_row_indices.extend(int(value) for value in features[:, 0].tolist())

    expected_ids = [f"record_id:{dataset.records[row_index]['id']}" for row_index in payload_row_indices]
    assert persisted_ids == expected_ids
    assert payload_row_indices != [0, 1, 2, 3]


def test_structured_record_payload_ids_match_persisted_ids_when_shuffled(db_path, store) -> None:
    dataset = TinyRecordsPayloadIdDataset()
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        consumed_batches = list(attached)

    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    decoded_ids = [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]

    persisted_ids: list[str] = [sample_id for batch in decoded_ids for sample_id in batch]
    emitted_records: list[tuple[str, int]] = []
    for batch in consumed_batches:
        record_ids, values = batch
        batch_record_ids = [str(record_id) for record_id in record_ids]
        batch_values = [int(value) for value in values.tolist()]
        emitted_records.extend(list(zip(batch_record_ids, batch_values)))
    emitted_record_ids = [record_id for record_id, _value in emitted_records]
    print("Emitted Records:", emitted_records)
    print("Emitted Record IDs:", emitted_record_ids)
    print("Persisted IDs:", persisted_ids)

    assert persisted_ids == [f"record_id:{record_id}" for record_id in emitted_record_ids]
    assert emitted_record_ids != ["cust_001", "cust_002", "cust_003", "cust_004"]


def test_same_dataset_can_use_different_sample_id_resolvers_across_runs(db_path, store) -> None:
    loader_a = DataLoader(TinyRecordsDataset(), batch_size=2, shuffle=False, num_workers=0)
    loader_b = DataLoader(TinyRecordsDataset(), batch_size=2, shuffle=False, num_workers=0)

    with Run(store=store) as run_a:
        list(attach(loader_a, run_a, role="train"))

    with Run(store=store) as run_b:
        list(
            attach(
                loader_b,
                run_b,
                role="train",
                sample_id_resolver=custom_sample_id_resolver,
            )
        )

    dataset_rows = fetch_all(
        db_path,
        """
        SELECT run_id, dataset_id, sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id IN (?, ?)
        ORDER BY run_id
        """,
        (run_a.run_id, run_b.run_id),
    )
    assert len(dataset_rows) == 2
    by_run_id = {row["run_id"]: row for row in dataset_rows}
    assert by_run_id[run_a.run_id]["dataset_id"] == by_run_id[run_b.run_id]["dataset_id"]
    assert (
        by_run_id[run_a.run_id]["sample_id_scheme"],
        by_run_id[run_a.run_id]["sample_id_resolver"],
    ) == (
        "record_id",
        "structured_record",
    )
    assert (
        by_run_id[run_b.run_id]["sample_id_scheme"],
        by_run_id[run_b.run_id]["sample_id_resolver"],
    ) == (
        "record_id",
        "user_override",
    )


def test_wrapped_structured_record_datasets_use_underlying_identity_contract() -> None:
    records_descriptor, records_fingerprint, _ = resolve_dataset_identity(
        wrap_dataset(TinyRecordsDataset())
    )
    rows_descriptor, rows_fingerprint, _ = resolve_dataset_identity(
        wrap_dataset(TinyRowsDataset())
    )

    assert records_descriptor.name == "TinyRecordsDataset"
    assert rows_descriptor.name == "TinyRowsDataset"
    assert records_fingerprint.fingerprint_method == "in_memory_deterministic_v1"
    assert rows_fingerprint.fingerprint_method == "in_memory_deterministic_v1"
    assert records_descriptor.dataset_id != rows_descriptor.dataset_id
    assert records_fingerprint.fingerprint != rows_fingerprint.fingerprint


def _build_file_dataset(root: Path) -> Path:
    (root / "class_a").mkdir(parents=True, exist_ok=True)
    (root / "class_b").mkdir(parents=True, exist_ok=True)
    (root / "class_a" / "item_0.txt").write_text("a0")
    (root / "class_a" / "item_1.txt").write_text("a1")
    (root / "class_b" / "item_2.txt").write_text("b2")
    return root
