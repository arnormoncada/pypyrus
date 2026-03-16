"""
Validate that PyPyrus DataLoader cloning preserves map-style loader behavior.

Run:
    python examples/validate_dataloader_clone_equivalence.py

What this checks:
1. Attached loader yields the same batch payloads as the original loader.
2. Important DataLoader settings are preserved in the cloned loader.

This script intentionally uses a map-style dataset and num_workers=0 for a
clean deterministic comparison.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from pypyrus.core.attach import attach
from pypyrus.core.run import Run
from pypyrus.storage.sqlite_store import SQLiteStore


class TinyMapDataset(Dataset):
    def __init__(self, n: int = 20):
        self.x = torch.arange(n * 2, dtype=torch.float32).view(n, 2)
        self.y = torch.arange(n, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _equal_payload(a, b) -> bool:
    if type(a) is not type(b):
        return False

    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_equal_payload(x, y) for x, y in zip(a, b))

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_equal_payload(a[k], b[k]) for k in a)

    return a == b


def _assert_loader_settings_preserved(original: DataLoader, cloned: DataLoader) -> None:
    def _effective_batch_size(loader: DataLoader) -> int | None:
        if loader.batch_size is not None:
            return loader.batch_size
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "batch_size"):
            return getattr(batch_sampler, "batch_size")
        return None

    checks = {
        "effective_batch_size": (_effective_batch_size(original), _effective_batch_size(cloned)),
        "drop_last": (original.drop_last, cloned.drop_last),
        "num_workers": (original.num_workers, cloned.num_workers),
        "pin_memory": (original.pin_memory, cloned.pin_memory),
        "timeout": (original.timeout, cloned.timeout),
        "has_batch_sampler": (original.batch_sampler is not None, cloned.batch_sampler is not None),
        "sampler_type": (type(original.sampler).__name__, type(cloned.sampler).__name__),
    }

    mismatches = [
        f"{name}: original={lhs!r}, cloned={rhs!r}"
        for name, (lhs, rhs) in checks.items()
        if lhs != rhs
    ]
    if mismatches:
        raise AssertionError("Loader settings mismatch:\n" + "\n".join(mismatches))


def main() -> None:
    torch.manual_seed(42)

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

    db_path = Path("/tmp/pypyrus_clone_equivalence.db")
    if db_path.exists():
        db_path.unlink()

    store = SQLiteStore(db_path)
    with Run(store=store) as run:
        attached = attach(original_loader, run, role="train")

        # Compare core settings on cloned loader inside proxy.
        _assert_loader_settings_preserved(original_loader, attached.loader)

        original_batches = list(original_loader)
        attached_batches = list(attached)

        if len(original_batches) != len(attached_batches):
            raise AssertionError(
                f"Batch count mismatch: original={len(original_batches)}, attached={len(attached_batches)}"
            )

        for i, (orig, proxied) in enumerate(zip(original_batches, attached_batches)):
            if not _equal_payload(orig, proxied):
                raise AssertionError(f"Batch payload mismatch at batch index {i}")

    print("Clone equivalence check passed.")
    print(f"Compared {len(original_batches)} batches with identical payloads.")
    print("Key loader settings are preserved for the map-style path.")


if __name__ == "__main__":
    main()
