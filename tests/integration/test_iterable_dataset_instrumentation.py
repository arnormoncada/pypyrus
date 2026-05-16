from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import (
    DuckTypedMapDataset,
    TinyIterableDataset,
    fetch_all,
    fetch_one,
    iterable_sample_id_resolver,
)


def test_attach_rejects_iterable_datasets_without_sample_id_resolver(store) -> None:
    loader = DataLoader(TinyIterableDataset(), batch_size=2)

    with Run(store=store) as run:
        with pytest.raises(ValueError, match="requires sample_id_resolver"):
            attach(loader, run, role="train")


def test_attach_rejects_datasets_that_do_not_inherit_torch_dataset_base(store) -> None:
    loader = DataLoader(DuckTypedMapDataset(), batch_size=2, shuffle=False)

    with Run(store=store) as run:
        with pytest.raises(TypeError, match="inherit torch.utils.data.Dataset"):
            attach(loader, run, role="train")


def test_iterable_dataset_persists_sample_ids_from_user_resolver(db_path, store) -> None:
    loader = DataLoader(TinyIterableDataset(n=4), batch_size=2)

    with Run(store=store) as run:
        attached = attach(
            loader,
            run,
            role="train",
            sample_id_resolver=iterable_sample_id_resolver,
        )
        consumed_batches = list(attached)

    assert len(consumed_batches) == 2

    dataset_row = fetch_one(
        db_path,
        """
        SELECT sample_id_scheme, sample_id_resolver
        FROM dataset_registrations
        WHERE run_id = ?
        """,
        (run.run_id,),
    )
    assert dataset_row["sample_id_scheme"] == "custom"
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
        ["record_id:stream_0", "record_id:stream_1"],
        ["record_id:stream_2", "record_id:stream_3"],
    ]
