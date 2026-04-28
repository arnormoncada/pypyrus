from __future__ import annotations

from typing import Any

from pypyrus.core.run import Run
from pypyrus.core.sample_id import SampleIdResolver
from pypyrus.instrumentation.dataloader import wrap_dataloader


def attach(
    loader: Any,
    run: Run,
    *,
    role: str,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
) -> Any:
    """
    Attach PyPyrus instrumentation to a DataLoader.

    This wraps the loader so that batch deliveries emit provenance events.

    Parameters
    ----------
    loader:
        A PyTorch DataLoader instance.

    run:
        The active PyPyrus Run.

    role:
        A label identifying the purpose of this loader within the run
        (e.g. ``'train'``, ``'val'``, ``'test'``).  Required so that
        multiple loaders in the same run can be distinguished unambiguously.

    id_aware_collate:
        Set to True when your custom collate function reorders or filters
        samples and returns (batch, remapped_ids).

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
    )
