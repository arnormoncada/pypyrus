from __future__ import annotations

from typing import Any

from pypyrus.core.run import Run
from pypyrus.core.sample_id import SampleIdResolver
from pypyrus.instrumentation.pytorch.dataloader import wrap_dataloader


def attach(
    loader: Any,
    run: Run,
    *,
    role: str,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
    dataset_name: str | None = None,
    dataset_uri: str | None = None,
    dataset_version_hint: str | None = None,
) -> Any:
    """
    Attach PyPyrus instrumentation to a DataLoader.

    This wraps the loader so that batch deliveries emit provenance events.

    Parameters
    ----------
    loader:
        A PyTorch DataLoader instance whose dataset inherits
        ``torch.utils.data.Dataset`` or
        ``torch.utils.data.IterableDataset``.

    run:
        The active PyPyrus Run.

    role:
        A label identifying the purpose of this loader within the run
        (e.g. ``'train'``, ``'val'``, ``'test'``).  Required so that
        multiple loaders in the same run can be distinguished unambiguously.

    id_aware_collate:
        Set to True when your custom collate function reorders or filters
        samples and returns (batch, remapped_ids).

    sample_id_resolver:
        Optional for map-style datasets. Required for
        ``torch.utils.data.IterableDataset``.

    dataset_name, dataset_uri, dataset_version_hint:
        Optional explicit dataset provenance metadata. Use these when PyPyrus
        cannot infer a meaningful source name/path from your dataset object.

    Returns
    -------
    Wrapped DataLoader proxy.
    """
    if not role or not role.strip():
        raise ValueError(
            "attach() requires a non-empty 'role' argument "
            "(e.g. role='train', role='val').  "
            "This uniquely identifies the loader within the run."
        )

    return wrap_dataloader(
        loader,
        run,
        role=role,
        sample_id_resolver=sample_id_resolver,
        id_aware_collate=id_aware_collate,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
        dataset_version_hint=dataset_version_hint,
    )
