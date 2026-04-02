from __future__ import annotations

"""
PyPyrus sample-ID resolution conventions.

These are PyPyrus-supported dataset conventions, not PyTorch-wide standards.
Each built-in family has a canonical contract plus a small set of
compatibility aliases. Datasets outside these conventions should use
`sample_id_resolver=` at attach time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


KNOWN_SCHEMES = ("filepath", "record_id", "row", "logical", "index")

# File collection contract:
# - canonical sample container attr: `samples`
# - compatibility alias: `imgs`
# - supporting dataset root attr: `root`
FILE_COLLECTION_SAMPLE_ATTRS = ("samples", "imgs")
FILE_COLLECTION_ROOT_ATTR = "root"

# Structured record contract:
# - canonical record container attr: `records`
# - compatibility aliases: `rows`
# - canonical record id attr: `record_ids`
# - compatibility alias: `ids`
STRUCTURED_RECORD_ID_ATTRS = ("record_ids", "ids")
STRUCTURED_RECORD_CONTAINER_ATTRS = ("records", "rows")
STRUCTURED_RECORD_KEY_FIELDS = ("record_id", "id", "uuid", "key")

# Framework/logical compatibility contract:
# - current built-in support is intentionally narrow
FRAMEWORK_LOGICAL_CLASS_NAMES = ("MNIST", "FashionMNIST", "KMNIST", "QMNIST")


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
    """Resolve one sample using the built-in PyPyrus contract order."""
    if user_resolver is not None:
        return _normalize_user_resolution(
            user_resolver(dataset, index, sample),
            default_resolver="user_override",
        )

    file_resolution = _resolve_file_collection_sample_id(dataset, index)
    if file_resolution is not None:
        return file_resolution

    record_resolution = _resolve_structured_record_sample_id(dataset, index)
    if record_resolution is not None:
        return record_resolution

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
    """Infer resolver metadata using the same contract order as runtime resolution."""
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

    record_resolution = _resolve_structured_record_sample_id(
        dataset,
        0,
        allow_missing_index=True,
    )
    if record_resolution is not None:
        return record_resolution.sample_id_scheme, record_resolution.sample_id_resolver

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
    """Resolve file-backed datasets that expose the file-collection contract."""
    sample_path = _get_indexed_file_path(dataset, index, allow_missing_index=allow_missing_index)
    if sample_path is None:
        return None

    dataset_root = getattr(dataset, FILE_COLLECTION_ROOT_ATTR, None)
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
    """Read the canonical file-path attrs plus the supported compatibility alias."""
    for attr in FILE_COLLECTION_SAMPLE_ATTRS:
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


def _resolve_structured_record_sample_id(
    dataset: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> SampleIdResolution | None:
    """Resolve datasets that expose the structured-record contract."""
    record_id = _get_indexed_record_id(
        dataset,
        index,
        allow_missing_index=allow_missing_index,
    )
    if record_id is not None:
        return SampleIdResolution(
            sample_id=f"record_id:{record_id}",
            sample_id_scheme="record_id",
            sample_id_resolver="structured_record",
        )

    record = _get_indexed_record(
        dataset,
        index,
        allow_missing_index=allow_missing_index,
    )
    if record is None:
        return None

    record_key = _extract_record_key(record)
    if record_key is not None:
        return SampleIdResolution(
            sample_id=f"record_id:{record_key}",
            sample_id_scheme="record_id",
            sample_id_resolver="structured_record",
        )

    return SampleIdResolution(
        sample_id=f"row:{index}",
        sample_id_scheme="row",
        sample_id_resolver="structured_record",
    )


def _get_indexed_record_id(
    dataset: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> str | None:
    """Read the canonical record-id attr plus the supported compatibility alias."""
    for attr in STRUCTURED_RECORD_ID_ATTRS:
        values = getattr(dataset, attr, None)
        if values is None:
            continue
        item = _get_indexed_value(values, index, allow_missing_index=allow_missing_index)
        if item is None:
            continue
        return str(item)
    return None


def _get_indexed_record(
    dataset: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> Any | None:
    """Read the canonical record container attr plus the supported compatibility alias."""
    for attr in STRUCTURED_RECORD_CONTAINER_ATTRS:
        values = getattr(dataset, attr, None)
        if values is None:
            continue
        item = _get_indexed_value(values, index, allow_missing_index=allow_missing_index)
        if item is not None:
            return item
    return None


def _extract_record_key(record: Any) -> str | None:
    """Read a stable record key from the canonical field names in priority order."""
    for key in STRUCTURED_RECORD_KEY_FIELDS:
        if isinstance(record, dict):
            value = record.get(key)
        else:
            value = getattr(record, key, None)
        if value is not None:
            return str(value)
    return None


def _resolve_framework_logical_sample_id(
    dataset: Any,
    index: int,
) -> SampleIdResolution | None:
    """Resolve the current narrow built-in logical/framework compatibility case."""
    class_name = dataset.__class__.__name__
    if class_name in FRAMEWORK_LOGICAL_CLASS_NAMES:
        split = "train" if getattr(dataset, "train", False) else "test"
        return SampleIdResolution(
            sample_id=f"logical:{split}#{index}",
            sample_id_scheme="logical",
            sample_id_resolver="framework_logical",
        )
    return None


def _get_indexed_value(
    values: Any,
    index: int,
    *,
    allow_missing_index: bool = False,
) -> Any | None:
    try:
        if index >= len(values):
            if allow_missing_index and len(values) > 0:
                index = 0
            else:
                return None
        return values[index]
    except Exception:
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
