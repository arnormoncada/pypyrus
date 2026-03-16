from __future__ import annotations

from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run

from tests.helpers import (
    TinyMapDataset,
    assert_loader_settings_preserved,
    equal_payload,
)


def test_attached_loader_preserves_map_style_behavior(store) -> None:
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
