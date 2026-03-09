"""
Framework-specific instrumentation.

This package contains PyTorch-facing wrappers (dataset, dataloader, collate) that
extract sample IDs and emit batch-level events via the Run/store.
"""

from __future__ import annotations

from .dataset import wrap_dataset
from .dataloader import wrap_dataloader
from .collate import wrap_collate

__all__ = [
    "wrap_dataset",
    "wrap_dataloader",
    "wrap_collate",
]