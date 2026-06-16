from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypyrus.core.sample_id import (
    FILE_COLLECTION_SAMPLE_ATTRS,
    STRUCTURED_RECORD_CONTAINER_ATTRS,
    STRUCTURED_RECORD_ID_ATTRS,
)
from pypyrus.provenance.fingerprints import hash_json, stable_json_dumps

MAX_LOCAL_SAMPLED_FILES = 256
LOCAL_SAMPLE_BYTES = 64 * 1024


@dataclass(slots=True, kw_only=True)
class DatasetDescriptor:
	dataset_id: str
	name: str
	uri: str | None = None
	version_hint: str | None = None


@dataclass(slots=True, kw_only=True)
class DatasetFingerprint:
	fingerprint: str
	fingerprint_method: str


def _identity_target(dataset: Any) -> Any:
	"""Unwrap PyPyrus dataset wrappers before inferring dataset identity."""
	current = dataset
	seen: set[int] = set()
	while hasattr(current, "dataset") and id(current) not in seen:
		seen.add(id(current))
		try:
			next_dataset = getattr(current, "dataset")
		except Exception:
			break
		if next_dataset is None:
			break
		current = next_dataset
	return current


def _best_effort_dataset_name(dataset: Any) -> str:
	return dataset.__class__.__name__


def _best_effort_dataset_uri(dataset: Any) -> str | None:
	for attr in ("root", "path", "data_dir", "directory"):
		if hasattr(dataset, attr):
			value = getattr(dataset, attr)
			if value is not None:
				return str(value)
	return None


def _best_effort_version_hint(dataset: Any) -> str | None:
	for attr in ("revision", "version"):
		if hasattr(dataset, attr):
			value = getattr(dataset, attr)
			if value is not None:
				return str(value)
	return None


def infer_dataset_descriptor(dataset: Any) -> DatasetDescriptor:
	dataset = _identity_target(dataset)
	descriptor_payload = {
		"class_name": dataset.__class__.__name__,
		"name": _best_effort_dataset_name(dataset),
		"uri": _best_effort_dataset_uri(dataset),
		"version_hint": _best_effort_version_hint(dataset),
		"length": len(dataset) if hasattr(dataset, "__len__") else None,
	}
	return DatasetDescriptor(
		dataset_id=hash_json(descriptor_payload),
		name=descriptor_payload["name"],
		uri=descriptor_payload["uri"],
		version_hint=descriptor_payload["version_hint"],
	)


def infer_dataset_descriptor_with_overrides(
	dataset: Any,
	*,
	name_override: str | None = None,
	uri_override: str | None = None,
	version_hint_override: str | None = None,
) -> DatasetDescriptor:
	dataset = _identity_target(dataset)
	descriptor_payload = {
		"class_name": dataset.__class__.__name__,
		"name": name_override if name_override is not None else _best_effort_dataset_name(dataset),
		"uri": uri_override if uri_override is not None else _best_effort_dataset_uri(dataset),
		"version_hint": (
			version_hint_override
			if version_hint_override is not None
			else _best_effort_version_hint(dataset)
		),
		"length": len(dataset) if hasattr(dataset, "__len__") else None,
	}
	return DatasetDescriptor(
		dataset_id=hash_json(descriptor_payload),
		name=descriptor_payload["name"],
		uri=descriptor_payload["uri"],
		version_hint=descriptor_payload["version_hint"],
	)


def dataset_id_from_fingerprint(fingerprint: DatasetFingerprint) -> str:
	return f"{fingerprint.fingerprint_method}:{fingerprint.fingerprint}"


def _hf_builtin_fingerprint(dataset: Any) -> DatasetFingerprint | None:
	for attr in ("_fingerprint", "fingerprint"):
		if hasattr(dataset, attr):
			value = getattr(dataset, attr)
			if isinstance(value, str) and value.strip():
				return DatasetFingerprint(
					fingerprint=value.strip(),
					fingerprint_method="hf_builtin",
				)
	return None


def _iter_local_files(path: Path) -> list[Path]:
	if path.is_file():
		return [path]

	files = [p for p in path.rglob("*") if p.is_file()]
	files.sort(key=lambda p: p.as_posix())
	return files


def _local_sampled_manifest_fingerprint(path: Path) -> DatasetFingerprint:
	files = _iter_local_files(path)
	hasher = hashlib.sha256()

	for f in files:
		rel = f.name if path.is_file() else f.relative_to(path).as_posix()
		stat = f.stat()
		hasher.update(rel.encode("utf-8"))
		hasher.update(b"\x00")
		hasher.update(str(stat.st_size).encode("ascii"))
		hasher.update(b"\x00")

	if files:
		stride = max(1, math.ceil(len(files) / MAX_LOCAL_SAMPLED_FILES))
		sampled = files[::stride][:MAX_LOCAL_SAMPLED_FILES]
		for f in sampled:
			rel = f.name if path.is_file() else f.relative_to(path).as_posix()
			with f.open("rb") as handle:
				chunk = handle.read(LOCAL_SAMPLE_BYTES)
			hasher.update(rel.encode("utf-8"))
			hasher.update(b"\x01")
			hasher.update(chunk)
			hasher.update(b"\x00")

	return DatasetFingerprint(
		fingerprint=hasher.hexdigest(),
		fingerprint_method="local_sampled_manifest_v1",
	)


def fingerprint_local_path(path: str | Path) -> DatasetFingerprint:
	path_obj = Path(path).expanduser()
	return _local_sampled_manifest_fingerprint(path_obj)


def _is_supported_in_memory(value: Any) -> bool:
	if value is None or isinstance(value, (bool, int, float, str, bytes)):
		return True
	if isinstance(value, (list, tuple)):
		return all(_is_supported_in_memory(item) for item in value)
	if isinstance(value, dict):
		return all(isinstance(k, (str, int, float, bool)) and _is_supported_in_memory(v) for k, v in value.items())
	return False


def _normalize_in_memory(value: Any) -> Any:
	if value is None or isinstance(value, (bool, int, str)):
		return value
	if isinstance(value, float):
		if math.isfinite(value):
			return value
		return {"__float__": repr(value)}
	if isinstance(value, bytes):
		return {"__bytes_hex__": value.hex()}
	if isinstance(value, (list, tuple)):
		return [_normalize_in_memory(v) for v in value]
	if isinstance(value, dict):
		items = sorted(value.items(), key=lambda kv: stable_json_dumps(str(kv[0])))
		return {str(k): _normalize_in_memory(v) for k, v in items}
	raise TypeError(f"Unsupported in-memory value type: {type(value)}")


def _in_memory_deterministic_fingerprint(dataset: Any) -> DatasetFingerprint | None:
    dataset = _identity_target(dataset)

    if _is_supported_in_memory(dataset):
        candidate = dataset
    else:
        candidate = _best_effort_in_memory_payload(dataset)
        if candidate is None:
            return None
        if not _is_supported_in_memory(candidate):
            return None

    return DatasetFingerprint(
        fingerprint=hash_json(_normalize_in_memory(candidate)),
        fingerprint_method="in_memory_deterministic_v1",
    )


def _best_effort_in_memory_payload(dataset: Any) -> Any:
	for attr in (
		"data",
		*STRUCTURED_RECORD_CONTAINER_ATTRS,
		*STRUCTURED_RECORD_ID_ATTRS,
		*FILE_COLLECTION_SAMPLE_ATTRS,
	):
		if not hasattr(dataset, attr):
			continue
		try:
			value = getattr(dataset, attr)
		except Exception:
			continue
		if _is_supported_in_memory(value):
			return value
	return None


def _fallback_descriptor_fingerprint(
	descriptor: DatasetDescriptor,
	*,
	reason: str,
) -> DatasetFingerprint:
	payload = {
		"dataset_id": descriptor.dataset_id,
		"name": descriptor.name,
		"uri": descriptor.uri,
		"version_hint": descriptor.version_hint,
		"reason": reason,
	}
	return DatasetFingerprint(
		fingerprint=hash_json(payload),
		fingerprint_method="descriptor_hash_fallback",
	)


def resolve_dataset_identity(
	dataset: Any,
	*,
	name_override: str | None = None,
	uri_override: str | None = None,
	version_hint_override: str | None = None,
) -> tuple[DatasetDescriptor, DatasetFingerprint, str | None]:
	"""
	Resolve descriptor and fingerprint for a dataset.

	Strategy order:
	1) Hugging Face built-in fingerprint
	2) Local file/directory sampled-content manifest hash
	3) Supported in-memory deterministic hash
	4) Descriptor-hash fallback

	Returns (descriptor, fingerprint, warning_message).
	warning_message is set only when fallback is used after a strategy failure.
	"""
	dataset = _identity_target(dataset)
	descriptor = infer_dataset_descriptor_with_overrides(
		dataset,
		name_override=name_override,
		uri_override=uri_override,
		version_hint_override=version_hint_override,
	)

	hf = _hf_builtin_fingerprint(dataset)
	if hf is not None:
		descriptor.dataset_id = dataset_id_from_fingerprint(hf)
		return descriptor, hf, None

	uri = descriptor.uri
	if uri:
		path = Path(uri).expanduser()
		if path.exists() and (path.is_file() or path.is_dir()):
			try:
				fingerprint = _local_sampled_manifest_fingerprint(path)
				descriptor.dataset_id = dataset_id_from_fingerprint(fingerprint)
				return descriptor, fingerprint, None
			except Exception as exc:  # pragma: no cover - defensive fallback
				warning = f"local fingerprinting failed: {exc.__class__.__name__}"
				fingerprint = _fallback_descriptor_fingerprint(descriptor, reason=warning)
				descriptor.dataset_id = dataset_id_from_fingerprint(fingerprint)
				return descriptor, fingerprint, warning

	in_memory = _in_memory_deterministic_fingerprint(dataset)
	if in_memory is not None:
		descriptor.dataset_id = dataset_id_from_fingerprint(in_memory)
		return descriptor, in_memory, None

	fingerprint = _fallback_descriptor_fingerprint(descriptor, reason="unsupported_source")
	descriptor.dataset_id = dataset_id_from_fingerprint(fingerprint)
	return descriptor, fingerprint, None
