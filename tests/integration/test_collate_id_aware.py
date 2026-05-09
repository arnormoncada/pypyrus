from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import TinyMapDataset, fetch_all


def reorder_collate(
    samples: list[tuple[torch.Tensor, int]],
    sample_ids: list[Any],
) -> tuple[list[torch.Tensor], list[Any]]:
    paired = list(zip(samples, sample_ids))
    paired = list(reversed(paired))

    remapped_samples = [sample for sample, _ in paired]
    remapped_ids = [sample_id for _, sample_id in paired]

    features = torch.stack([sample[0] for sample in remapped_samples])
    labels = torch.tensor([sample[1] for sample in remapped_samples], dtype=torch.long)
    return [features, labels], remapped_ids


def filter_collate(
    samples: list[tuple[torch.Tensor, int]],
    sample_ids: list[Any],
) -> tuple[list[torch.Tensor], list[Any]]:
    paired = list(zip(samples, sample_ids))
    kept = [pair for pair in paired if pair[0][1] == 0]

    remapped_samples = [sample for sample, _ in kept]
    remapped_ids = [sample_id for _, sample_id in kept]

    features = torch.stack([sample[0] for sample in remapped_samples])
    labels = torch.tensor([sample[1] for sample in remapped_samples], dtype=torch.long)
    return [features, labels], remapped_ids


def bad_collate(
    samples: list[tuple[torch.Tensor, int]],
    sample_ids: list[Any],
) -> list[torch.Tensor]:
    features = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    return [features, labels]


def _fetch_batch_ids(db_path, run_id: str) -> list[list[str]]:
    batch_rows = fetch_all(
        db_path,
        """
        SELECT sample_ids_blob
        FROM batch_delivered
        WHERE run_id = ?
        ORDER BY global_sequence
        """,
        (run_id,),
    )
    return [decode_sample_ids_blob(row["sample_ids_blob"]) for row in batch_rows]


def test_id_aware_collate_reorders_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=reorder_collate)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train", id_aware_collate=True)
        list(attached)

    decoded = _fetch_batch_ids(db_path, run.run_id)
    assert decoded == [["index:3", "index:2", "index:1", "index:0"]]


def test_id_aware_collate_filters_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=filter_collate)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train", id_aware_collate=True)
        list(attached)

    decoded = _fetch_batch_ids(db_path, run.run_id)
    assert decoded == [["index:0", "index:3"]]


def test_id_aware_collate_requires_remapped_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=bad_collate)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train", id_aware_collate=True)
        with pytest.raises(TypeError, match=r"return \(batch, remapped_ids\)"):
            list(attached)

# Negative test to show that if id_aware_collate=false and collate_fn DOES reorder/filter, 
# without taking the id's into account then the sample_ids are still returned as they were, without remapping. 
# 
def test_id_aware_collate_false_returns_unmapped_ids(db_path, store) -> None:
    dataset = TinyMapDataset(n=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=reorder_collate)

    with Run(store=store) as run:
        attached = attach(loader, run, role="train", id_aware_collate=False)
        list(attached)

    decoded = _fetch_batch_ids(db_path, run.run_id)
    
    assert decoded == [["index:0", "index:1", "index:2", "index:3"]] 