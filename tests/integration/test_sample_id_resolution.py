from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import (
    TinyFileCollectionDataset,
    TinyMapDataset,
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
        SELECT d.sample_id_scheme, d.sample_id_resolver
        FROM datasets d
        JOIN run_datasets rd ON rd.dataset_id = d.dataset_id
        WHERE rd.run_id = ?
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
        SELECT d.sample_id_scheme, d.sample_id_resolver
        FROM datasets d
        JOIN run_datasets rd ON rd.dataset_id = d.dataset_id
        WHERE rd.run_id = ?
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


def _build_file_dataset(root: Path) -> Path:
    (root / "class_a").mkdir(parents=True, exist_ok=True)
    (root / "class_b").mkdir(parents=True, exist_ok=True)
    (root / "class_a" / "item_0.txt").write_text("a0")
    (root / "class_a" / "item_1.txt").write_text("a1")
    (root / "class_b" / "item_2.txt").write_text("b2")
    return root
