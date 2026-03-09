from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable
"""
This module provides detinistic hashing utilities, so that batch identity can be compared across runs,
metadata (configs, transform declarations, etc.) can be hashed for reproducibility checks, sample IDs can
be serialized for storage.
"""

DEFAULT_HASH_ALGO = "sha256"


def _get_hasher(algorithm: str = DEFAULT_HASH_ALGO) -> "hashlib._Hash":
    """
    Return a hashlib hasher instance for the requested algorithm.
    """
    try:
        return hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc


def hash_bytes(data: bytes, algorithm: str = DEFAULT_HASH_ALGO) -> str:
    """
    Hash raw bytes and return a hexadecimal digest.
    """
    hasher = _get_hasher(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def stable_json_dumps(value: Any) -> str:
    """
    Serialize a Python object to canonical JSON.

    This is useful for hashing configs, transform declarations, and other
    structured metadata in a deterministic way.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_json(value: Any, algorithm: str = DEFAULT_HASH_ALGO) -> str:
    """
    Hash a Python object by first converting it to canonical JSON.
    """
    payload = stable_json_dumps(value).encode("utf-8")
    return hash_bytes(payload, algorithm=algorithm)


def encode_sample_ids(sample_ids: Iterable[int | str]) -> bytes:
    """
    Deterministically encode an ordered sequence of sample IDs into bytes.

    This encoding is simple and stable:
    - IDs are converted to strings
    - Serialized as canonical JSON
    - Encoded as UTF-8 bytes

    This is suitable for:
    - hashing ordered batch identity
    - optional compression before storage
    """
    normalized = [str(sample_id) for sample_id in sample_ids]
    return stable_json_dumps(normalized).encode("utf-8")


def hash_ordered_ids(
    sample_ids: Iterable[int | str],
    algorithm: str = DEFAULT_HASH_ALGO,
) -> str:
    """
    Hash an ordered sequence of sample IDs.

    This is the primary batch fingerprint used for reproducibility comparison.
    Changing either membership or order will change the resulting hash.
    """
    encoded = encode_sample_ids(sample_ids)
    return hash_bytes(encoded, algorithm=algorithm)


def hash_unordered_ids(
    sample_ids: Iterable[int | str],
    algorithm: str = DEFAULT_HASH_ALGO,
) -> str:
    """
    Hash a batch as an unordered multiset of sample IDs.

    This can be useful for distinguishing:
    - same members, different order
    - different members entirely
    """
    normalized = sorted(str(sample_id) for sample_id in sample_ids)
    encoded = stable_json_dumps(normalized).encode("utf-8")
    return hash_bytes(encoded, algorithm=algorithm)