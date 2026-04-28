from __future__ import annotations

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import TinyMapDataset, fetch_all


def _fetch_ids(db_path, run_id: str) -> list[list[str]]:
    rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run_id,),
    )
    return [decode_sample_ids_blob(row["sample_ids_blob"]) for row in rows]


def test_batch_size_one_records_single_sample_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=3)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)

    assert _fetch_ids(db_path, run.run_id) == [
        ["index:0"],
        ["index:1"],
        ["index:2"],
    ]


def test_auto_batching_disabled_records_single_sample_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=3)
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        shuffle=False,
        num_workers=0,
    )

    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)

    assert _fetch_ids(db_path, run.run_id) == [
        ["index:0"],
        ["index:1"],
        ["index:2"],
    ]
