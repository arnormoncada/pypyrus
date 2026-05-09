from __future__ import annotations

import inspect
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


def _accepts_sample_ids(collate_fn: Callable[..., Any]) -> bool:
    """Return True if collate_fn can be called with (payloads, sample_ids)."""
    try:
        signature = inspect.signature(collate_fn)
    except (TypeError, ValueError):
        return False

    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
        return True

    positional_count = sum(
        param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for param in params
    )
    return positional_count >= 2


def collate_with_ids(
    samples: list[Any],
    user_collate_fn: Callable[[list[Any]], Any] | None = None,
    *,
    id_aware_collate: bool = False,
) -> tuple[Any, list[Any]]:
    """
    Collate samples while preserving ordered PyPyrus sample IDs.

    Returns:
        (batch_payload, sample_ids)

        Behavior:
        - If samples are wrapped by PyPyrus, IDs are extracted first.
        - By default, the payloads are collated and IDs are returned in
            original order.
        - If id_aware_collate is enabled, the collate function is called
            with (payloads, sample_ids) and must return (batch, remapped_ids).
        - If samples are not wrapped, the full samples are collated and IDs
            are returned as an empty list.
    """
    if id_aware_collate and user_collate_fn is None:
        raise ValueError(
            "id_aware_collate=True requires a user-provided collate_fn "
            "that accepts (payloads, sample_ids)."
        )

    collate_fn = user_collate_fn or default_collate

    split = _split_wrapped_samples(samples)
    if split is None:
        return collate_fn(samples), []

    payloads, sample_ids = split

    if id_aware_collate:
        result = collate_fn(payloads, sample_ids)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "id_aware_collate=True requires collate_fn to return "
                "(batch, remapped_ids)."
            )
        batch, remapped_ids = result
        if not isinstance(remapped_ids, (list, tuple)):
            raise TypeError(
                "id_aware_collate=True requires remapped_ids to be a list or tuple."
            )
        return batch, list(remapped_ids)

    if _accepts_sample_ids(collate_fn):
        # In non-id-aware mode, allow flexible signatures but keep original IDs.
        result = collate_fn(payloads, sample_ids)
        batch = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    else:
        batch = collate_fn(payloads)
    return batch, sample_ids


class CollateWithIdsWrapper:
    """Pickle-safe callable wrapper for DataLoader collate instrumentation."""

    def __init__(
        self,
        collate_fn: Callable[[list[Any]], Any] | None,
        *,
        id_aware_collate: bool = False,
    ):
        self.collate_fn = collate_fn
        self.id_aware_collate = id_aware_collate

    def __call__(self, samples: list[Any]) -> tuple[Any, list[Any]]:
        return collate_with_ids(
            samples,
            user_collate_fn=self.collate_fn,
            id_aware_collate=self.id_aware_collate,
        )


def wrap_collate(
    collate_fn: Callable[[list[Any]], Any] | None,
    *,
    id_aware_collate: bool = False,
) -> Callable[[list[Any]], tuple[Any, list[Any]]]:
    """
    Wrap a collate function so that PyPyrus sample IDs survive collation.

    The returned function always returns:
        (batch_payload, sample_ids)

    If id_aware_collate is enabled, the collate function must accept
    (payloads, sample_ids) and return (batch, remapped_ids).
    """
    return CollateWithIdsWrapper(
        collate_fn,
        id_aware_collate=id_aware_collate,
    )
