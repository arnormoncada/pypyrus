from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, IterableDataset

from pypyrus.core.attach import attach
from pypyrus.core.run import Run


class TinyIterableDataset(IterableDataset):
    def __iter__(self):
        yield from range(4)


def test_attach_rejects_iterable_datasets(store) -> None:
    loader = DataLoader(TinyIterableDataset(), batch_size=2)

    with Run(store=store) as run:
        with pytest.raises(TypeError, match="map-style datasets only"):
            attach(loader, run, role="train")
