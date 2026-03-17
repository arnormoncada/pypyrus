"""
Framework-specific instrumentation.

This package contains PyTorch-facing wrappers (dataset, dataloader, collate) that
extract sample IDs and emit batch-level events via the Run/store.
"""

from __future__ import annotations

__all__ = [
    "wrap_dataset",
    "wrap_dataloader",
    "wrap_collate",
]


def __getattr__(name: str):
    if name == "wrap_dataset":
        from pypyrus.instrumentation.dataset import wrap_dataset

        return wrap_dataset
    if name == "wrap_dataloader":
        from pypyrus.instrumentation.dataloader import wrap_dataloader

        return wrap_dataloader
    if name == "wrap_collate":
        from pypyrus.instrumentation.collate import wrap_collate

        return wrap_collate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
