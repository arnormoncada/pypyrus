from __future__ import annotations

from typing import Any

from pypyrus.core.run import Run
from pypyrus.instrumentation.dataloader import wrap_dataloader


def attach(loader: Any, run: Run) -> Any:
    """
    Attach PyPyrus instrumentation to a DataLoader.

    This wraps the loader so that batch deliveries emit provenance events.

    Parameters
    ----------
    loader:
        A PyTorch DataLoader instance.

    run:
        The active PyPyrus Run.

    Returns
    -------
    Wrapped DataLoader proxy.
    """

    return wrap_dataloader(loader, run)