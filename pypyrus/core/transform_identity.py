"""Transform declaration extraction for provenance logging.

This module turns a dataset transform (or transform pipeline) into a
deterministic, JSON-safe declaration used by ``TransformDeclaredEvent``.

What it records:
- ordered ``transform_list`` with transform name/module/params
- ``params_hash`` derived from the canonicalized transform list
- ``introspection_level``: ``full`` when all values are structured,
  ``partial`` when any fallback serialization is used

Fallback chain (best-effort, never raises for unknown objects):
1. If the object looks like Compose (has ``.transforms`` list/tuple), serialize
    each child transform in order.
2. Otherwise serialize the object as a single transform.
3. For each parameter value, normalize primitives/containers/tensor-like values
    first; for unsupported objects, fall back to class name plus ``repr``.

The goal is robust capture with stable hashes, while still logging partial
information when full introspection is not possible.
"""

from __future__ import annotations

from typing import Any

from pypyrus.provenance.fingerprints import hash_json, stable_json_dumps

def extract_transform_declaration(transform: Any) -> dict[str, Any] | None:
    """
    Convert a transform pipeline into a deterministic structured declaration.

    Returns None if no transform is present.
    """
    if transform is None:
        return None

    chain, used_fallback = _extract_transform_chain(transform)

    declaration = {
        "transform_list": chain,
        "params_hash": hash_json(chain),
        "introspection_level": "partial" if used_fallback else "full",
    }
    return declaration


def _extract_transform_chain(transform: Any) -> tuple[list[dict[str, Any]], bool]:
    """
    Extract an ordered list of transform specs.

    Supports common torchvision-style Compose pipelines and falls back
    to a single transform spec for unknown objects.
    """
    # torchvision Compose-style objects usually expose `.transforms`
    if hasattr(transform, "transforms") and isinstance(transform.transforms, (list, tuple)):
        chain: list[dict[str, Any]] = []
        any_fallback = False
        for t in transform.transforms:
            transform_spec, used_fallback = _serialize_transform(t)
            chain.append(transform_spec)
            any_fallback = any_fallback or used_fallback
        return chain, any_fallback

    transform_spec, used_fallback = _serialize_transform(transform)
    return [transform_spec], used_fallback


def _serialize_transform(transform: Any) -> tuple[dict[str, Any], bool]:
    """
    Serialize a single transform object into a deterministic dictionary.
    """
    params, used_fallback = _extract_transform_params(transform)

    return {
        "name": transform.__class__.__name__,
        "module": getattr(transform.__class__, "__module__", None),
        "params": params,
    }, used_fallback


def _extract_transform_params(transform: Any) -> tuple[dict[str, Any], bool]:
    """
    Extract JSON-safe public parameters from a transform.

    Best-effort strategy:
    - use vars(transform) when available
    - ignore private attributes
    - convert values recursively into JSON-safe forms
    """
    try:
        raw = vars(transform)
    except TypeError:
        raw = {}

    params: dict[str, Any] = {}
    any_fallback = False
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        normalized, used_fallback = _to_json_safe(value)
        params[key] = normalized
        any_fallback = any_fallback or used_fallback

    return params, any_fallback


def _to_json_safe(value: Any) -> tuple[Any, bool]:
    """
    Convert arbitrary Python values into stable JSON-safe structures.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, False

    if isinstance(value, dict):
        normalized_items: dict[str, Any] = {}
        any_fallback = False
        for k, v in sorted(value.items(), key=lambda item: str(item[0])):
            normalized, used_fallback = _to_json_safe(v)
            normalized_items[str(k)] = normalized
            any_fallback = any_fallback or used_fallback
        return normalized_items, any_fallback

    if isinstance(value, (list, tuple)):
        normalized_list: list[Any] = []
        any_fallback = False
        for v in value:
            normalized, used_fallback = _to_json_safe(v)
            normalized_list.append(normalized)
            any_fallback = any_fallback or used_fallback
        return normalized_list, any_fallback

    if isinstance(value, set):
        normalized_values: list[Any] = []
        any_fallback = False
        for v in value:
            normalized, used_fallback = _to_json_safe(v)
            normalized_values.append(normalized)
            any_fallback = any_fallback or used_fallback
        return sorted(normalized_values, key=stable_json_dumps), any_fallback

    # torch / numpy-like objects often have useful metadata
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": value.__class__.__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }, False

    # nested transform / callable fallback
    if hasattr(value, "__class__"):
        return {
            "type": value.__class__.__name__,
            "repr": repr(value),
        }, True

    return repr(value), True


def transform_chain_id(transform_decl: dict[str, Any]) -> str:
    """
    Compute a stable chain identifier from a transform declaration.
    """
    return hash_json(transform_decl["transform_list"])