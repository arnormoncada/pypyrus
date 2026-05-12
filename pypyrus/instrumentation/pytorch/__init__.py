"""
PyTorch-specific instrumentation helpers.

This package contains the concrete wrappers that integrate PyPyrus with
PyTorch DataLoader and Dataset APIs while keeping provenance semantics and
storage concerns outside the framework-facing layer.
"""

from __future__ import annotations

from .collate import wrap_collate
from .dataloader import wrap_dataloader
from .dataset import wrap_dataset

__all__ = [
    "wrap_dataset",
    "wrap_dataloader",
    "wrap_collate",
]
