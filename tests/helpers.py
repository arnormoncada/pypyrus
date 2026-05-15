from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import BatchSampler, Dataset, IterableDataset


class ComposeLike:
    """Minimal Compose-like container for transform declaration tests."""

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


class TinyFileCollectionDataset(Dataset):
    def __init__(self, root: Path):
        self.root = str(root)
        self.samples = []
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        for class_index, class_dir in enumerate(class_dirs):
            for file_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                self.samples.append((str(file_path), class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        _, class_index = self.samples[idx]
        sample = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample, class_index


class TinyRecordIdsDataset(Dataset):
    def __init__(self):
        self.record_ids = ["alpha", "beta", "gamma", "delta"]

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample, idx % 2


class TinyRecordsDataset(Dataset):
    def __init__(self):
        self.records = [
            {"id": "cust_001", "value": 1},
            {"id": "cust_002", "value": 2},
            {"id": "cust_003", "value": 3},
            {"id": "cust_004", "value": 4},
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample, idx % 2


class TinyRecordsPayloadIdDataset(Dataset):
    def __init__(self):
        self.records = [
            {"id": "cust_001", "value": 1},
            {"id": "cust_002", "value": 2},
            {"id": "cust_003", "value": 3},
            {"id": "cust_004", "value": 4},
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        record = self.records[idx]
        return record["id"], record["value"]


class TinyRowsDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"value": 10},
            {"value": 11},
            {"value": 12},
            {"value": 13},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample, idx % 2


class TinyIterableDataset(IterableDataset):
    def __init__(self, n: int = 4):
        self.n = n

    def __iter__(self):
        for idx in range(self.n):
            yield {
                "record_id": f"stream_{idx}",
                "features": torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32),
                "label": idx % 2,
            }


class DuckTypedMapDataset:
    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample, idx % 2


def custom_sample_id_resolver(dataset: Any, index: int, sample: Any) -> str:
    return f"record_id:custom_{index}"


def iterable_sample_id_resolver(dataset: Any, index: int, sample: Any) -> str:
    return f"record_id:{sample['record_id']}"


def custom_collate_with_metadata(
    samples: list[tuple[torch.Tensor, int]],
) -> dict[str, Any]:
    """Return a richer batch object to verify custom collate preservation."""
    features = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    return {
        "features": features,
        "labels": labels,
        "label_sum": int(labels.sum().item()),
    }


class EvenOddBatchSampler(BatchSampler):
    """Group even and odd indices separately to stress custom batch sampling."""

    def __init__(self, data_source: Dataset):
        self._batches = [
            [idx for idx in range(len(data_source)) if idx % 2 == 0],
            [idx for idx in range(len(data_source)) if idx % 2 == 1],
        ]

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def seed_worker(worker_id: int) -> None:
    """Top-level worker seeding helper so multi-worker tests stay pickle-safe."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def iter_payloads(loader: Any) -> list[Any]:
    return list(loader)


def equal_payload(lhs: Any, rhs: Any) -> bool:
    """Recursively check equality of batch payloads, handling tensors and nested structures."""
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
    """Check clone-time settings that should remain behaviorally equivalent."""

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
        "worker_init_fn": (original.worker_init_fn, cloned.worker_init_fn),
        "generator_is_same": (original.generator is cloned.generator, True),
        "has_batch_sampler": (
            original.batch_sampler is not None,
            cloned.batch_sampler is not None,
        ),
        "persistent_workers": (
            getattr(original, "persistent_workers", False),
            getattr(cloned, "persistent_workers", False),
        ),
        "prefetch_factor": (
            getattr(original, "prefetch_factor", None),
            getattr(cloned, "prefetch_factor", None),
        ),
    }
    # With an explicit batch_sampler, PyTorch may rewrite `.sampler`
    # internally, so compare the batch-sampler contract instead.
    if original.batch_sampler is None and cloned.batch_sampler is None:
        checks["sampler_type"] = (
            type(original.sampler).__name__,
            type(cloned.sampler).__name__,
        )
    else:
        checks["batch_sampler_type"] = (
            type(original.batch_sampler).__name__,
            type(cloned.batch_sampler).__name__,
        )

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
