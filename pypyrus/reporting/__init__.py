"""
PyPyrus reporting layer.

Provides high-level query helpers and cross-run comparison utilities.
"""

from pypyrus.reporting.compare import compare_runs, format_run_comparison
from pypyrus.reporting.queries import (
    get_batch_for_run_step,
    decode_sample_ids_blob,
    get_batches_for_run,
    get_datasets_for_run,
    get_environment_for_run,
    get_run,
    get_transforms_for_run,
    list_runs,
)

__all__ = [
    "compare_runs",
    "decode_sample_ids_blob",
    "format_run_comparison",
    "get_batch_for_run_step",
    "get_batches_for_run",
    "get_datasets_for_run",
    "get_environment_for_run",
    "get_run",
    "get_transforms_for_run",
    "list_runs",
]
