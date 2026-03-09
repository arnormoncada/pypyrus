"""
Provenance domain model: event schemas, fingerprint logic, and semantics.

No framework-specific imports should live here.
"""

from __future__ import annotations

from .events import (
    RunStartEvent,
    RunEndEvent,
    DatasetRegisteredEvent,
    TransformDeclaredEvent,
    BatchDeliveredEvent,
    EnvironmentSnapshotEvent,
)
from .fingerprints import (
    hash_bytes,
    hash_ordered_ids,
)

__all__ = [
    # Events
    "RunStartEvent",
    "RunEndEvent",
    "DatasetRegisteredEvent",
    "TransformDeclaredEvent",
    "BatchDeliveredEvent",
    "EnvironmentSnapshotEvent",
    # Fingerprints
    "hash_bytes",
    "hash_ordered_ids",
]