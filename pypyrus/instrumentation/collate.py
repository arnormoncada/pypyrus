from __future__ import annotations

from typing import Any, Callable

from torch.utils.data._utils.collate import default_collate

from pypyrus.instrumentation.dataset import (
    PYPYRUS_ID_KEY,
    PYPYRUS_PAYLOAD_KEY,
)


def _split_wrapped_samples(samples: list[Any]) -> tuple[list[Any], list[Any]] | None:
    """
    If samples are PyPyrus-wrapped dataset items, split them into:
    - payloads: original samples
    - sample_ids: ordered sample IDs

    Returns None if the batch does not look like a wrapped PyPyrus batch.
    """
    if not samples:
        return None

    if not all(
        isinstance(sample, dict)
        and PYPYRUS_ID_KEY in sample
        and PYPYRUS_PAYLOAD_KEY in sample
        for sample in samples
    ):
        return None

    payloads = [sample[PYPYRUS_PAYLOAD_KEY] for sample in samples]
    sample_ids = [sample[PYPYRUS_ID_KEY] for sample in samples]
    return payloads, sample_ids


def collate_with_ids(
    samples: list[Any],
    user_collate_fn: Callable[[list[Any]], Any] | None = None,
) -> tuple[Any, list[Any]]:
    """
    Collate samples while preserving ordered PyPyrus sample IDs.

    Returns:
        (batch_payload, sample_ids)

    Behavior:
    - If samples are wrapped by PyPyrus, IDs are extracted first.
    - The payloads are then collated using the user's collate function
      or PyTorch's default_collate.
    - If samples are not wrapped, the full samples are collated and IDs
      are returned as an empty list.
    """
    collate_fn = user_collate_fn or default_collate

    split = _split_wrapped_samples(samples)
    if split is None:
        return collate_fn(samples), []

    payloads, sample_ids = split
    batch = collate_fn(payloads)
    return batch, sample_ids


class CollateWithIdsWrapper:
    """Pickle-safe callable wrapper for DataLoader collate instrumentation."""

    def __init__(self, collate_fn: Callable[[list[Any]], Any] | None):
        self.collate_fn = collate_fn

    def __call__(self, samples: list[Any]) -> tuple[Any, list[Any]]:
        return collate_with_ids(samples, user_collate_fn=self.collate_fn)


def wrap_collate(
    collate_fn: Callable[[list[Any]], Any] | None,
) -> Callable[[list[Any]], tuple[Any, list[Any]]]:
    """
    Wrap a collate function so that PyPyrus sample IDs survive collation.

    The returned function always returns:
        (batch_payload, sample_ids)
    """
    return CollateWithIdsWrapper(collate_fn)
