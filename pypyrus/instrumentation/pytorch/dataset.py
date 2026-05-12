from __future__ import annotations

from typing import Any

from pypyrus.core.sample_id import SampleIdResolution, SampleIdResolver, infer_sample_id_metadata, resolve_sample_id


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

    def __init__(
        self,
        dataset: Any,
        *,
        sample_id_resolver: SampleIdResolver | None = None,
    ):
        self.dataset = dataset
        self._sample_id_resolver = sample_id_resolver
        self._sample_id_scheme, self._sample_id_resolver_name = infer_sample_id_metadata(
            dataset,
            user_resolver=sample_id_resolver,
        )

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
        - use resolver-driven normalized sample IDs
        """
        resolution = self._resolve_sample_id(index, sample)
        return resolution.sample_id

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

    def sample_id_metadata(self) -> tuple[str, str]:
        return self._sample_id_scheme, self._sample_id_resolver_name

    def _resolve_sample_id(self, index: int, sample: Any) -> SampleIdResolution:
        resolution = resolve_sample_id(
            self.dataset,
            index,
            sample,
            user_resolver=self._sample_id_resolver,
        )
        self._sample_id_scheme = resolution.sample_id_scheme
        self._sample_id_resolver_name = resolution.sample_id_resolver
        return resolution


def is_wrapped_dataset(dataset: Any) -> bool:
    """Return True if the dataset is already wrapped by PyPyrus."""
    return isinstance(dataset, DatasetWrapper)


def wrap_dataset(
    dataset: Any,
    *,
    sample_id_resolver: SampleIdResolver | None = None,
) -> DatasetWrapper:
    """
    Wrap a dataset with DatasetWrapper if it is not already wrapped.
    """
    if is_wrapped_dataset(dataset):
        return dataset
    return DatasetWrapper(dataset, sample_id_resolver=sample_id_resolver)
