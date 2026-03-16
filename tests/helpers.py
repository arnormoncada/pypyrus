from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ComposeLike:
    def __init__(self, transforms: list[Any]):
        self.transforms = transforms

    def __call__(self, sample: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ScaleFeatures:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, sample: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
        features, label = sample
        return features * self.scale, label


class OffsetFeatures:
    def __init__(self, offset: float):
        self.offset = offset

    def __call__(self, sample: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
        features, label = sample
        return features + self.offset, label


class TinyMapDataset(Dataset):
    def __init__(
        self,
        n: int = 12,
        *,
        start: int = 0,
        transform: Any | None = None,
    ):
        self.transform = transform
        self.data = [
            {
                "features": [float(start + idx), float(start + idx + 0.5)],
                "label": (start + idx) % 3,
            }
            for idx in range(n)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data[idx]
        sample = (
            torch.tensor(row["features"], dtype=torch.float32),
            int(row["label"]),
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def equal_payload(lhs: Any, rhs: Any) -> bool:
    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, torch.Tensor):
        return torch.equal(lhs, rhs)

    if isinstance(lhs, (list, tuple)):
        return len(lhs) == len(rhs) and all(
            equal_payload(left_item, right_item)
            for left_item, right_item in zip(lhs, rhs)
        )

    if isinstance(lhs, dict):
        return lhs.keys() == rhs.keys() and all(
            equal_payload(lhs[key], rhs[key]) for key in lhs
        )

    return lhs == rhs


def assert_loader_settings_preserved(original: Any, cloned: Any) -> None:
    def effective_batch_size(loader: Any) -> int | None:
        if loader.batch_size is not None:
            return loader.batch_size
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "batch_size"):
            return getattr(batch_sampler, "batch_size")
        return None

    checks = {
        "effective_batch_size": (
            effective_batch_size(original),
            effective_batch_size(cloned),
        ),
        "drop_last": (original.drop_last, cloned.drop_last),
        "num_workers": (original.num_workers, cloned.num_workers),
        "pin_memory": (original.pin_memory, cloned.pin_memory),
        "timeout": (original.timeout, cloned.timeout),
        "has_batch_sampler": (
            original.batch_sampler is not None,
            cloned.batch_sampler is not None,
        ),
        "sampler_type": (
            type(original.sampler).__name__,
            type(cloned.sampler).__name__,
        ),
    }

    mismatches = [
        f"{name}: original={left!r}, cloned={right!r}"
        for name, (left, right) in checks.items()
        if left != right
    ]
    if mismatches:
        raise AssertionError("Loader settings mismatch:\n" + "\n".join(mismatches))


def fetch_all(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def fetch_one(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row:
    rows = fetch_all(db_path, query, params)
    if len(rows) != 1:
        raise AssertionError(f"Expected exactly one row, got {len(rows)}")
    return rows[0]
