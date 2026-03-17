from __future__ import annotations

from collections.abc import Sequence
from typing import Any


PYPYRUS_ID_KEY = "__pypyrus_id__"
PYPYRUS_PAYLOAD_KEY = "__pypyrus_payload__"


class DatasetWrapper:
    """
    Wrap a dataset so each returned sample carries a stable sample_id.

    This wrapper is intended for map-style datasets that implement
    __getitem__ and __len__.

    The wrapped sample is returned as:

        {
            "__pypyrus_id__": <sample_id>,
            "__pypyrus_payload__": <original_sample>,
        }

    The default sample_id is the dataset index.
    """

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]

        return {
            PYPYRUS_ID_KEY: self._make_sample_id(index, sample),
            PYPYRUS_PAYLOAD_KEY: sample,
        }

    def _make_sample_id(self, index: int, sample: Any) -> int | str:
        """
        Return a stable identifier for a sample.

        MVP behavior:
        - use dataset index for map-style datasets
        """
        return index

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to the wrapped dataset.

        This helps preserve compatibility with code that accesses dataset
        properties such as `.transform`, `.targets`, `.classes`, etc.
        """
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)


def is_wrapped_dataset(dataset: Any) -> bool:
    """Return True if the dataset is already wrapped by PyPyrus."""
    return isinstance(dataset, DatasetWrapper)


def wrap_dataset(dataset: Any) -> DatasetWrapper:
    """
    Wrap a dataset with DatasetWrapper if it is not already wrapped.
    """
    if is_wrapped_dataset(dataset):
        return dataset
    return DatasetWrapper(dataset)
