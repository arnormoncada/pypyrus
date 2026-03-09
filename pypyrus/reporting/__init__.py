"""
PyPyrus reporting layer.

Provides high-level query helpers over a Store for inspecting
provenance data collected during training runs.
"""

from pypyrus.reporting.queries import (
    get_batches_for_run,
)

__all__ = [
    "get_batches_for_run",
]
