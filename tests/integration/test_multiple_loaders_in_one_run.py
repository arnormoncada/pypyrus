from __future__ import annotations

from copy import deepcopy

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.reporting import get_batch_for_run_step, get_batches_for_run

from tests.helpers import TinyMapDataset, fetch_all


def test_multiple_loaders_share_run_sequence_and_keep_roles_distinct(
    db_path,
    store,
) -> None:
    train_loader = DataLoader(TinyMapDataset(n=5, start=0), batch_size=2, shuffle=False)
    val_loader = DataLoader(TinyMapDataset(n=4, start=100), batch_size=2, shuffle=False)

    with Run(store=store) as run:
        attached_train = attach(train_loader, run, role="train")
        attached_val = attach(val_loader, run, role="val")

        list(attached_train)
        list(attached_val)

    role_rows = fetch_all(
        db_path,
        """
        SELECT role
        FROM run_datasets
        WHERE run_id = ?
        ORDER BY role
        """,
        (run.run_id,),
    )
    assert [row["role"] for row in role_rows] == ["train", "val"]

    batch_rows = fetch_all(
        db_path,
        """
        SELECT loader_id, dataset_id, global_step, global_sequence
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    assert [row["global_sequence"] for row in batch_rows] == [0, 1, 2, 3, 4]

    steps_by_dataset: dict[str, list[int]] = {}
    for row in batch_rows:
        steps_by_dataset.setdefault(row["dataset_id"], []).append(row["global_step"])

    assert sorted(steps_by_dataset.values()) == [[0, 1], [0, 1, 2]]


def test_multiple_loaders_can_share_one_dataset_identity_without_batch_collisions(
    db_path,
    store,
) -> None:
    shared_dataset = TinyMapDataset(n=6, start=50)
    same_identity_dataset = TinyMapDataset(n=6, start=50)
    same_identity_dataset.data = deepcopy(shared_dataset.data)

    train_loader = DataLoader(shared_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(same_identity_dataset, batch_size=3, shuffle=False)

    with Run(store=store) as run:
        attached_train = attach(train_loader, run, role="train")
        attached_val = attach(val_loader, run, role="val")

        list(attached_train)
        list(attached_val)

    loader_rows = fetch_all(
        db_path,
        """
        SELECT loader_id, dataset_id, role
        FROM loaders
        WHERE run_id = ?
        ORDER BY role
        """,
        (run.run_id,),
    )
    assert len(loader_rows) == 2
    assert loader_rows[0]["dataset_id"] == loader_rows[1]["dataset_id"]
    assert [row["role"] for row in loader_rows] == ["train", "val"]
    assert loader_rows[0]["loader_id"] != loader_rows[1]["loader_id"]

    batch_rows = fetch_all(
        db_path,
        """
        SELECT loader_id, global_step, global_sequence
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run.run_id,),
    )
    assert [row["global_sequence"] for row in batch_rows] == [0, 1, 2, 3, 4]

    steps_by_loader: dict[str, list[int]] = {}
    for row in batch_rows:
        steps_by_loader.setdefault(row["loader_id"], []).append(row["global_step"])

    assert sorted(steps_by_loader.values()) == [[0, 1], [0, 1, 2]]

    train_batches = get_batches_for_run(store, run.run_id, role="train")
    val_batches = get_batches_for_run(store, run.run_id, role="val")
    assert [row["global_step"] for row in train_batches] == [0, 1, 2]
    assert [row["global_step"] for row in val_batches] == [0, 1]

    first_batch = get_batch_for_run_step(store, run.run_id, 0)
    second_batch = get_batch_for_run_step(store, run.run_id, 1)
    assert first_batch is not None
    assert second_batch is not None
    assert first_batch["global_sequence"] == 0
    assert second_batch["global_sequence"] == 1
    assert first_batch["loader_id"] != second_batch["loader_id"]
