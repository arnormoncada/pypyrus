from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


KNOWN_SCHEMES = ("filepath", "record_id", "row", "logical", "index")


@dataclass(slots=True, kw_only=True)
class SampleIdResolution:
    sample_id: str
    sample_id_scheme: str
    sample_id_resolver: str


SampleIdResolver = Callable[[Any, int, Any], SampleIdResolution | str | int]


def resolve_sample_id(
    dataset: Any,
    index: int,
    sample: Any,
    *,
    user_resolver: SampleIdResolver | None = None,
) -> SampleIdResolution:
    if user_resolver is not None:
        return _normalize_user_resolution(
            user_resolver(dataset, index, sample),
            default_resolver="user_override",
        )

    file_resolution = _resolve_file_collection_sample_id(dataset, index)
    if file_resolution is not None:
        return file_resolution

    logical_resolution = _resolve_framework_logical_sample_id(dataset, index)
    if logical_resolution is not None:
        return logical_resolution

    return SampleIdResolution(
        sample_id=f"index:{index}",
        sample_id_scheme="index",
        sample_id_resolver="fallback_index",
    )


def infer_sample_id_metadata(
    dataset: Any,
    *,
    user_resolver: SampleIdResolver | None = None,
) -> tuple[str, str]:
    if user_resolver is not None:
        if hasattr(dataset, "__len__") and len(dataset) > 0 and hasattr(dataset, "__getitem__"):
            sample = dataset[0]
            resolution = _normalize_user_resolution(
                user_resolver(dataset, 0, sample),
                default_resolver="user_override",
            )
            return resolution.sample_id_scheme, resolution.sample_id_resolver
        return "custom", "user_override"

    file_resolution = _resolve_file_collection_sample_id(dataset, 0, allow_missing_index=True)
    if file_resolution is not None:
        return file_resolution.sample_id_scheme, file_resolution.sample_id_resolver

    logical_resolution = _resolve_framework_logical_sample_id(dataset, 0)
    if logical_resolution is not None:
        return logical_resolution.sample_id_scheme, logical_resolution.sample_id_resolver

    return "index", "fallback_index"


def _resolve_file_collection_sample_id(
    dataset: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> SampleIdResolution | None:
    sample_path = _get_indexed_file_path(dataset, index, allow_missing_index=allow_missing_index)
    if sample_path is None:
        return None

    dataset_root = getattr(dataset, "root", None)
    path = Path(sample_path)
    if dataset_root is not None:
        root_path = Path(dataset_root)
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError:
            relative = path.name
    else:
        relative = path.name

    return SampleIdResolution(
        sample_id=f"filepath:{relative}",
        sample_id_scheme="filepath",
        sample_id_resolver="file_collection",
    )


def _get_indexed_file_path(
    dataset: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> str | None:
    for attr in ("samples", "imgs"):
        values = getattr(dataset, attr, None)
        if values is None:
            continue
        try:
            if index >= len(values):
                if allow_missing_index and len(values) > 0:
                    index = 0
                else:
                    return None
            item = values[index]
        except Exception:
            return None

        if isinstance(item, (list, tuple)) and item:
            candidate = item[0]
            if isinstance(candidate, (str, Path)):
                return str(candidate)
        if isinstance(item, (str, Path)):
            return str(item)

    return None


def _resolve_framework_logical_sample_id(
    dataset: Any,
    index: int,
) -> SampleIdResolution | None:
    class_name = dataset.__class__.__name__
    if class_name in {"MNIST", "FashionMNIST", "KMNIST", "QMNIST"}:
        split = "train" if getattr(dataset, "train", False) else "test"
        return SampleIdResolution(
            sample_id=f"logical:{split}#{index}",
            sample_id_scheme="logical",
            sample_id_resolver="framework_logical",
        )
    return None


def _normalize_user_resolution(
    value: SampleIdResolution | str | int,
    *,
    default_resolver: str,
) -> SampleIdResolution:
    if isinstance(value, SampleIdResolution):
        return value

    if isinstance(value, int):
        return SampleIdResolution(
            sample_id=f"index:{value}",
            sample_id_scheme="index",
            sample_id_resolver=default_resolver,
        )

    if not isinstance(value, str):
        raise TypeError(
            "sample_id_resolver must return SampleIdResolution, str, or int"
        )

    scheme = _infer_scheme_from_sample_id(value)
    sample_id = value if ":" in value else f"{scheme}:{value}"
    return SampleIdResolution(
        sample_id=sample_id,
        sample_id_scheme=scheme,
        sample_id_resolver=default_resolver,
    )


def _infer_scheme_from_sample_id(sample_id: str) -> str:
    for scheme in KNOWN_SCHEMES:
        prefix = f"{scheme}:"
        if sample_id.startswith(prefix):
            return scheme
    return "custom"
