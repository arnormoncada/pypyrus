from __future__ import annotations

import gzip
import inspect
import warnings
from typing import Any, Iterator
from uuid import uuid4

from torch.utils.data import DataLoader, IterableDataset

from pypyrus.core.run import Run
from pypyrus.core.dataset_identity import resolve_dataset_identity
from pypyrus.core.transform_identity import extract_transform_declaration, transform_chain_id
from pypyrus.instrumentation.collate import wrap_collate
from pypyrus.instrumentation.dataset import wrap_dataset
from pypyrus.provenance.events import (
    BatchDeliveredEvent,
    DatasetRegisteredEvent,
    LoaderRegisteredEvent,
    TransformDeclaredEvent,
)
from pypyrus.provenance.fingerprints import encode_sample_ids, hash_json, hash_ordered_ids


def _compress_sample_ids(sample_ids: list[int | str]) -> bytes:
    raw = encode_sample_ids(sample_ids)
    return gzip.compress(raw)


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, dict) and batch:
        first_value = next(iter(batch.values()))
        if hasattr(first_value, "__len__"):
            return len(first_value)

    if isinstance(batch, (list, tuple)) and batch:
        first_value = batch[0]
        if hasattr(first_value, "__len__"):
            return len(first_value)

    raise ValueError("Unable to infer batch size from batch payload")


def _clone_dataloader_with_wrapped_dataset(loader: DataLoader) -> DataLoader:
    """
    Create a new DataLoader with the same configuration, but using:
    - wrapped dataset
    - wrapped collate_fn

    Note:
    - This path currently supports map-style datasets only.
    """
    if isinstance(loader.dataset, IterableDataset):
        raise TypeError(
            "PyPyrus attach currently supports map-style datasets only. "
            "Received an IterableDataset, which is not yet supported by "
            "the wrapper-based sample ID injection path."
        )

    wrapped_dataset = wrap_dataset(loader.dataset)
    wrapped_collate_fn = wrap_collate(loader.collate_fn)

    kwargs: dict[str, Any] = {
        "dataset": wrapped_dataset,
        "batch_size": loader.batch_size,
        "shuffle": False,  # sampler/batch_sampler already carries ordering logic
        "sampler": loader.sampler,
        "batch_sampler": loader.batch_sampler,
        "num_workers": loader.num_workers,
        "collate_fn": wrapped_collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader.generator,
        "prefetch_factor": getattr(loader, "prefetch_factor", None),
        "persistent_workers": getattr(loader, "persistent_workers", False),
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
    }

    # DataLoader forbids setting both batch_sampler and batch_size/shuffle/sampler/drop_last
    if loader.batch_sampler is not None:
        kwargs.pop("batch_size", None)
        kwargs.pop("shuffle", None)
        kwargs.pop("sampler", None)
        kwargs.pop("drop_last", None)

    # Some args are invalid when num_workers == 0
    if loader.num_workers == 0:
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)

    # Remove None values that some torch versions dislike
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    safe_kwargs, dropped = _filter_supported_dataloader_kwargs(kwargs)
    if dropped:
        dropped_str = ", ".join(sorted(dropped))
        raise TypeError(
            "Failed to clone DataLoader for PyPyrus instrumentation. "
            "Dropping DataLoader kwargs is not allowed because it may change "
            f"training behavior. Unsupported kwargs: {dropped_str}"
        )

    try:
        return DataLoader(**safe_kwargs)
    except TypeError as exc:
        used_keys = ", ".join(sorted(safe_kwargs.keys()))
        raise TypeError(
            "Failed to clone DataLoader for PyPyrus instrumentation. "
            f"Used kwargs: {used_keys}. "
            f"Original error: {exc}"
        ) from exc


def _filter_supported_dataloader_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """
    Keep only kwargs supported by the current torch DataLoader constructor.

    This makes cloning more robust across torch versions where optional
    DataLoader parameters differ.
    """
    signature = inspect.signature(DataLoader.__init__)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )

    if accepts_kwargs:
        return dict(kwargs), set()

    allowed = {name for name in parameters if name != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = {k for k in kwargs if k not in allowed}
    return filtered, dropped


class DataLoaderProxy:
    """
    Wrap a DataLoader so PyPyrus can observe delivered batches.

    The proxy:
    - wraps the dataset with sample IDs
    - wraps the collate function so IDs survive collation
    - emits DatasetRegisteredEvent / TransformDeclaredEvent once
    - emits BatchDeliveredEvent for each delivered batch
    - yields the original batch payload to user code
    """

    def __init__(self, loader: DataLoader, run: Run, role: str):
        self.run = run
        self.role = role
        self.loader_id = str(uuid4())
        self._registered = False
        self._dataset_id: str | None = None
        self._global_step = 0
        self._source_loader = loader
        self.loader = _clone_dataloader_with_wrapped_dataset(loader)

    def _emit_registration_events_once(self) -> None:
        if self._registered:
            return

        dataset = self.loader.dataset
        descriptor, dataset_fingerprint, warning = resolve_dataset_identity(dataset)

        if warning is not None:
            warnings.warn(
                (
                    "PyPyrus dataset fingerprint fallback used "
                    f"for dataset_id={descriptor.dataset_id}, "
                    f"name={descriptor.name}, reason={warning}"
                ),
                stacklevel=2,
            )

        self._dataset_id = descriptor.dataset_id

        self.run.emit(
            DatasetRegisteredEvent(
                run_id=self.run.run_id,
                dataset_id=descriptor.dataset_id,
                name=descriptor.name,
                role=self.role,
                uri=descriptor.uri,
                version_hint=descriptor.version_hint,
                fingerprint=dataset_fingerprint.fingerprint,
                fingerprint_method=dataset_fingerprint.fingerprint_method,
            )
        )

        self.run.emit(
            LoaderRegisteredEvent(
                run_id=self.run.run_id,
                loader_id=self.loader_id,
                dataset_id=descriptor.dataset_id,
                role=self.role,
            )
        )

        declared_transform = getattr(dataset, "transform", None)
        transform_decl = extract_transform_declaration(declared_transform)
        if transform_decl is not None:
            self.run.emit(
                TransformDeclaredEvent(
                    run_id=self.run.run_id,
                    dataset_id=descriptor.dataset_id,
                    transform_chain_id=transform_chain_id(transform_decl),
                    transform_list=transform_decl["transform_list"],
                    params_hash=transform_decl["params_hash"],
                    introspection_level=transform_decl["introspection_level"],
                )
            )

        self._registered = True

    def __iter__(self) -> Iterator[Any]:
        self._emit_registration_events_once()

        dataset_id = self._dataset_id
        if dataset_id is None:
            descriptor, _, _ = resolve_dataset_identity(self.loader.dataset)
            dataset_id = descriptor.dataset_id
            self._dataset_id = dataset_id

        for batch_payload, sample_ids in self.loader:
            batch_size = len(sample_ids) if sample_ids else _infer_batch_size(batch_payload)
            batch_fingerprint = (
                hash_ordered_ids(sample_ids)
                if sample_ids
                else hash_json({"missing_sample_ids": True, "step": self._global_step})
            )
            sample_ids_blob = _compress_sample_ids(sample_ids) if sample_ids else None
            # Uncomment to save sample IDs as json instead of compressed binary (easier for debugging, but takes more space)
            # sample_ids_blob = json.dumps(sample_ids).encode("utf-8") if sample_ids else None

            self.run.emit(
                BatchDeliveredEvent(
                    run_id=self.run.run_id,
                    loader_id=self.loader_id,
                    dataset_id=dataset_id,
                    global_step=self._global_step,
                    global_sequence=self.run.next_batch_sequence(),
                    batch_size=batch_size,
                    batch_fingerprint=batch_fingerprint,
                    sample_ids_blob=sample_ids_blob,
                    rng_state_hash=None,
                )
            )

            self._global_step += 1
            yield batch_payload

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def wrap_dataloader(loader: DataLoader, run: Run, role: str) -> DataLoaderProxy:
    return DataLoaderProxy(loader, run, role=role)
