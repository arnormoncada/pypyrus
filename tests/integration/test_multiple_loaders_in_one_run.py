from __future__ import annotations

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run

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
        SELECT dataset_id, global_step, global_sequence
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
