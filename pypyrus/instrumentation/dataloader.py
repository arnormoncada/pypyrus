from __future__ import annotations

import gzip
import json
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from pypyrus.core.run import Run
from pypyrus.instrumentation.collate import wrap_collate
from pypyrus.instrumentation.dataset import wrap_dataset
from pypyrus.provenance.events import (
    BatchDeliveredEvent,
    DatasetRegisteredEvent,
    TransformDeclaredEvent,
)
from pypyrus.provenance.fingerprints import (
    encode_sample_ids,
    hash_json,
    hash_ordered_ids,
)


def _best_effort_dataset_name(dataset: Any) -> str:
    return dataset.__class__.__name__


def _best_effort_dataset_uri(dataset: Any) -> str | None:
    for attr in ("root", "path", "data_dir", "directory"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if value is not None:
                return str(value)
    return None


def _best_effort_dataset_id(dataset: Any) -> str:
    descriptor = {
        "class_name": dataset.__class__.__name__,
        "name": _best_effort_dataset_name(dataset),
        "uri": _best_effort_dataset_uri(dataset),
        "length": len(dataset) if hasattr(dataset, "__len__") else None,
    }
    return hash_json(descriptor)


def _best_effort_transform_decl(dataset: Any) -> dict[str, Any] | None:
    transforms_found: list[str] = []

    for attr in ("transform", "transforms", "target_transform"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if value is not None:
                transforms_found.append(repr(value))

    if not transforms_found:
        return None

    return {
        "transform_list": transforms_found,
        "deterministic_flag": False,
        "seed_policy": "unknown",
    }


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
    """
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

    return DataLoader(**kwargs)


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

    def __init__(self, loader: DataLoader, run: Run):
        self.run = run
        self._registered = False
        self._global_step = 0
        self._source_loader = loader
        self.loader = _clone_dataloader_with_wrapped_dataset(loader)

    def _emit_registration_events_once(self) -> None:
        if self._registered:
            return

        dataset = self.loader.dataset
        dataset_id = _best_effort_dataset_id(dataset)

        self.run.emit(
            DatasetRegisteredEvent(
                run_id=self.run.run_id,
                dataset_id=dataset_id,
                name=_best_effort_dataset_name(dataset),
                uri=_best_effort_dataset_uri(dataset),
                version_hint=None,
                fingerprint=None,
                fingerprint_method=None,
            )
        )

        transform_decl = _best_effort_transform_decl(dataset)
        if transform_decl is not None:
            self.run.emit(
                TransformDeclaredEvent(
                    run_id=self.run.run_id,
                    dataset_id=dataset_id,
                    transform_chain_id=hash_json(transform_decl["transform_list"]),
                    transform_list=transform_decl["transform_list"],
                    params_hash=hash_json(transform_decl),
                    deterministic_flag=transform_decl["deterministic_flag"],
                    seed_policy=transform_decl["seed_policy"],
                )
            )

        self._registered = True

    def __iter__(self) -> Iterator[Any]:
        self._emit_registration_events_once()

        dataset_id = _best_effort_dataset_id(self.loader.dataset)

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
                    dataset_id=dataset_id,
                    global_step=self._global_step,
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


def wrap_dataloader(loader: DataLoader, run: Run) -> DataLoaderProxy:
    return DataLoaderProxy(loader, run)