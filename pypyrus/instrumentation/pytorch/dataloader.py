from __future__ import annotations

import gzip
import inspect
import warnings
from typing import Any, Iterator
from uuid import uuid4

from torch.utils.data import DataLoader, Dataset, IterableDataset

from pypyrus.core.run import Run
from pypyrus.core.dataset_identity import resolve_dataset_identity
from pypyrus.core.sample_id import SampleIdResolver
from pypyrus.core.transform_identity import extract_transform_declaration
from pypyrus.instrumentation.pytorch.collate import wrap_collate
from pypyrus.instrumentation.pytorch.dataset import (
    wrap_dataset,
    wrap_iterable_dataset,
)
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


def _classify_dataset(dataset: Any) -> str:
    if isinstance(dataset, IterableDataset):
        return "iterable"
    if isinstance(dataset, Dataset):
        return "map"
    raise TypeError(
        "PyPyrus attach requires loader.dataset to inherit "
        "torch.utils.data.Dataset or torch.utils.data.IterableDataset."
    )


def _supported_dataloader_constructor_kwargs() -> set[str]:
    signature = inspect.signature(DataLoader.__init__)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def _build_clone_dataloader_kwargs(
    loader: DataLoader,
    *,
    wrapped_dataset: Any,
    wrapped_collate_fn: Any,
    dataset_kind: str,
) -> dict[str, Any]:
    """
    Reconstruct clone kwargs from the current DataLoader constructor surface.

    We preserve any constructor-backed loader attributes the installed torch
    version exposes, then apply the small set of PyPyrus-specific overrides
    and DataLoader legality fixups.
    """
    supported_kwargs = _supported_dataloader_constructor_kwargs()
    kwargs: dict[str, Any] = {}
    iterable_excluded = {"sampler", "batch_sampler", "shuffle"}

    for name in supported_kwargs:
        if dataset_kind == "iterable" and name in iterable_excluded:
            continue

        if name == "dataset":
            kwargs[name] = wrapped_dataset
            continue

        if name == "collate_fn":
            kwargs[name] = wrapped_collate_fn
            continue

        if dataset_kind == "map" and name == "shuffle":
            # Preserve the original sampler/batch_sampler objects rather than
            # letting DataLoader synthesize a fresh shuffle strategy.
            kwargs[name] = False
            continue

        if hasattr(loader, name):
            kwargs[name] = getattr(loader, name)

    # DataLoader forbids setting both batch_sampler and
    # batch_size/shuffle/sampler/drop_last.
    if kwargs.get("batch_sampler") is not None:
        kwargs.pop("batch_size", None)
        kwargs.pop("shuffle", None)
        kwargs.pop("sampler", None)
        kwargs.pop("drop_last", None)

    # Some args are invalid when num_workers == 0.
    if kwargs.get("num_workers") == 0:
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)

    # Remove None values that some torch versions dislike.
    return {k: v for k, v in kwargs.items() if v is not None}


def _clone_dataloader_with_wrapped_map_dataset(
    loader: DataLoader,
    *,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
) -> DataLoader:
    """
    Create a new DataLoader with the same configuration, but using:
    - wrapped dataset
    - wrapped collate_fn

    Note:
    - This path supports map-style datasets only.
    """
    wrapped_dataset = wrap_dataset(
        loader.dataset,
        sample_id_resolver=sample_id_resolver,
    )
    wrapped_collate_fn = wrap_collate(
        loader.collate_fn,
        id_aware_collate=id_aware_collate,
    )
    safe_kwargs = _build_clone_dataloader_kwargs(
        loader,
        wrapped_dataset=wrapped_dataset,
        wrapped_collate_fn=wrapped_collate_fn,
        dataset_kind="map",
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


def _clone_dataloader_with_wrapped_iterable_dataset(
    loader: DataLoader,
    *,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
) -> DataLoader:
    if sample_id_resolver is None:
        raise ValueError(
            "PyPyrus requires sample_id_resolver=... for IterableDataset "
            "instrumentation."
        )

    wrapped_dataset = wrap_iterable_dataset(
        loader.dataset,
        sample_id_resolver=sample_id_resolver,
    )
    wrapped_collate_fn = wrap_collate(
        loader.collate_fn,
        id_aware_collate=id_aware_collate,
    )
    safe_kwargs = _build_clone_dataloader_kwargs(
        loader,
        wrapped_dataset=wrapped_dataset,
        wrapped_collate_fn=wrapped_collate_fn,
        dataset_kind="iterable",
    )

    try:
        return DataLoader(**safe_kwargs)
    except TypeError as exc:
        used_keys = ", ".join(sorted(safe_kwargs.keys()))
        raise TypeError(
            "Failed to clone IterableDataset DataLoader for PyPyrus "
            f"instrumentation. Used kwargs: {used_keys}. Original error: {exc}"
        ) from exc


def _clone_dataloader_with_wrapped_dataset(
    loader: DataLoader,
    *,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
) -> DataLoader:
    dataset_kind = _classify_dataset(loader.dataset)
    if dataset_kind == "iterable":
        return _clone_dataloader_with_wrapped_iterable_dataset(
            loader,
            sample_id_resolver=sample_id_resolver,
            id_aware_collate=id_aware_collate,
        )

    return _clone_dataloader_with_wrapped_map_dataset(
        loader,
        sample_id_resolver=sample_id_resolver,
        id_aware_collate=id_aware_collate,
    )


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

    def __init__(
        self,
        loader: DataLoader,
        run: Run,
        role: str,
        sample_id_resolver: SampleIdResolver | None = None,
        id_aware_collate: bool = False,
        dataset_name: str | None = None,
        dataset_uri: str | None = None,
        dataset_version_hint: str | None = None,
    ):
        self.run = run
        self.role = role
        self.loader_id = str(uuid4())
        self._registered = False
        self._dataset_id: str | None = None
        self._dataset_registration_event_id: str | None = None
        self._global_step = 0
        self._dataset_name_override = dataset_name
        self._dataset_uri_override = dataset_uri
        self._dataset_version_hint_override = dataset_version_hint
        self._source_loader = loader
        self.loader = _clone_dataloader_with_wrapped_dataset(
            loader,
            sample_id_resolver=sample_id_resolver,
            id_aware_collate=id_aware_collate,
        )

    def _emit_registration_events_once(self) -> None:
        if self._registered:
            return

        dataset = self.loader.dataset
        descriptor, dataset_fingerprint, warning = resolve_dataset_identity(
            dataset,
            name_override=self._dataset_name_override,
            uri_override=self._dataset_uri_override,
            version_hint_override=self._dataset_version_hint_override,
        )

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

        sample_id_scheme, sample_id_resolver_name = ("index", "fallback_index")
        if hasattr(dataset, "sample_id_metadata"):
            sample_id_scheme, sample_id_resolver_name = dataset.sample_id_metadata()

        dataset_registration = DatasetRegisteredEvent(
            run_id=self.run.run_id,
            dataset_id=descriptor.dataset_id,
            name=descriptor.name,
            role=self.role,
            uri=descriptor.uri,
            version_hint=descriptor.version_hint,
            fingerprint=dataset_fingerprint.fingerprint,
            fingerprint_method=dataset_fingerprint.fingerprint_method,
            sample_id_scheme=sample_id_scheme,
            sample_id_resolver=sample_id_resolver_name,
        )
        self.run.emit(dataset_registration)
        self._dataset_registration_event_id = dataset_registration.event_id

        self.run.emit(
            LoaderRegisteredEvent(
                run_id=self.run.run_id,
                loader_id=self.loader_id,
                dataset_registration_event_id=dataset_registration.event_id,
            )
        )

        declared_transform = getattr(dataset, "transform", None)
        transform_decl = extract_transform_declaration(declared_transform)
        if transform_decl is not None:
            self.run.emit(
                TransformDeclaredEvent(
                    run_id=self.run.run_id,
                    dataset_registration_event_id=dataset_registration.event_id,
                    transform_list=transform_decl["transform_list"],
                    params_hash=transform_decl["params_hash"],
                    introspection_level=transform_decl["introspection_level"],
                )
            )

        self._registered = True

    def __iter__(self) -> Iterator[Any]:
        self._emit_registration_events_once()

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
                    global_step=self._global_step,
                    global_sequence=self.run.next_batch_sequence(),
                    batch_size=batch_size,
                    batch_fingerprint=batch_fingerprint,
                    sample_ids_blob=sample_ids_blob,
                )
            )

            self._global_step += 1
            yield batch_payload

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def wrap_dataloader(
    loader: DataLoader,
    run: Run,
    role: str,
    sample_id_resolver: SampleIdResolver | None = None,
    id_aware_collate: bool = False,
    dataset_name: str | None = None,
    dataset_uri: str | None = None,
    dataset_version_hint: str | None = None,
) -> DataLoaderProxy:
    return DataLoaderProxy(
        loader,
        run,
        role=role,
        sample_id_resolver=sample_id_resolver,
        id_aware_collate=id_aware_collate,
        dataset_name=dataset_name,
        dataset_uri=dataset_uri,
        dataset_version_hint=dataset_version_hint,
    )
