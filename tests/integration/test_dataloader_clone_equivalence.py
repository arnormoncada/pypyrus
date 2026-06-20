from __future__ import annotations

import inspect

import torch
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.instrumentation.pytorch import dataloader as dataloader_module
from pypyrus.reporting.queries import decode_sample_ids_blob

from tests.helpers import (
    EvenOddBatchSampler,
    TinyMapDataset,
    assert_loader_settings_preserved,
    custom_collate_with_metadata,
    equal_payload,
    fetch_all,
    seed_worker,
)


def test_attached_loader_preserves_map_style_behavior(store) -> None:
    # Baseline contract: for the simple single-worker map-style path,
    # attaching must not change the delivered batch payloads.
    dataset = TinyMapDataset(n=25)
    original_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        timeout=0,
    )

    with Run(store=store) as run:
        attached_loader = attach(original_loader, run, role="train")

        assert_loader_settings_preserved(original_loader, attached_loader.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached_loader)

    assert len(original_batches) == len(attached_batches)
    for original_batch, attached_batch in zip(original_batches, attached_batches):
        assert equal_payload(original_batch, attached_batch)


def test_attached_loader_preserves_multiworker_behavior(store) -> None:
    # Multi-worker loading is a high-risk path because the wrapped dataset and
    # wrapped collate function both need to remain spawn/pickle-safe.
    original_loader = DataLoader(
        TinyMapDataset(n=18),
        batch_size=3,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    comparison_loader = DataLoader(
        TinyMapDataset(n=18),
        batch_size=3,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )

    with Run(store=store) as run:
        attached_loader = attach(comparison_loader, run, role="train")
        assert_loader_settings_preserved(comparison_loader, attached_loader.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached_loader)

    assert len(original_batches) == len(attached_batches)
    for original_batch, attached_batch in zip(original_batches, attached_batches):
        assert equal_payload(original_batch, attached_batch)


def test_attached_loader_preserves_custom_collate_behavior(db_path, store) -> None:
    # Custom collate functions can compute additional metadata; attaching
    # should preserve the full return structure, not just tensors/labels.
    # We also assert that sample IDs still make it through collation into the
    # persisted provenance rows.
    original_loader = DataLoader(
        TinyMapDataset(n=9),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_with_metadata,
    )
    comparison_loader = DataLoader(
        TinyMapDataset(n=9),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_with_metadata,
    )

    with Run(store=store) as run:
        attached_loader = attach(comparison_loader, run, role="train")
        assert_loader_settings_preserved(comparison_loader, attached_loader.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached_loader)

    assert len(original_batches) == len(attached_batches)
    for original_batch, attached_batch in zip(original_batches, attached_batches):
        assert equal_payload(original_batch, attached_batch)

    batch_rows = fetch_all(
        db_path,
        """
        SELECT global_step, sample_ids_blob
        FROM batch_delivered
        ORDER BY global_sequence
        """,
    )
    assert [row["global_step"] for row in batch_rows] == [0, 1, 2]
    assert [
        decode_sample_ids_blob(row["sample_ids_blob"])
        for row in batch_rows
    ] == [
        ["index:0", "index:1", "index:2", "index:3"],
        ["index:4", "index:5", "index:6", "index:7"],
        ["index:8"],
    ]


def test_attached_loader_preserves_custom_batch_sampler_behavior(store) -> None:
    # Batch samplers define grouping semantics directly, so cloning must
    # preserve them exactly or the training batches change.
    original_loader = DataLoader(
        TinyMapDataset(n=8),
        batch_sampler=EvenOddBatchSampler(TinyMapDataset(n=8)),
        num_workers=0,
    )
    comparison_loader = DataLoader(
        TinyMapDataset(n=8),
        batch_sampler=EvenOddBatchSampler(TinyMapDataset(n=8)),
        num_workers=0,
    )

    with Run(store=store) as run:
        attached_loader = attach(comparison_loader, run, role="train")
        assert_loader_settings_preserved(comparison_loader, attached_loader.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached_loader)

    assert len(original_batches) == len(attached_batches)
    for original_batch, attached_batch in zip(original_batches, attached_batches):
        assert equal_payload(original_batch, attached_batch)


def test_attached_loader_preserves_seeded_shuffle_order(store) -> None:
    # Two loaders started from the same seed should yield the same order after
    # attach(); otherwise the wrapper is perturbing shuffle behavior.
    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)

    original_loader = DataLoader(
        TinyMapDataset(n=20),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        generator=generator_a,
    )
    comparison_loader = DataLoader(
        TinyMapDataset(n=20),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        generator=generator_b,
    )

    with Run(store=store) as run:
        attached_loader = attach(comparison_loader, run, role="train")
        assert_loader_settings_preserved(comparison_loader, attached_loader.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached_loader)

    assert len(original_batches) == len(attached_batches)
    for original_batch, attached_batch in zip(original_batches, attached_batches):
        assert equal_payload(original_batch, attached_batch)