from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


EventType = Literal[
    "run_start",
    "run_end",
    "dataset_registered",
    "loader_registered",
    "transform_declared",
    "batch_delivered",
    "environment_snapshot",
]

RunStatus = Literal["success", "failure", "interrupted"]


@dataclass(slots=True, kw_only=True)
class ProvenanceEvent:
    """Base class for all provenance events."""

    run_id: str
    event_type: EventType
    timestamp: str = field(default_factory=utc_now_iso)
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, kw_only=True)
class RunStartEvent(ProvenanceEvent):
    code_ref: str | None = None
    config_ref: str | None = None
    config_json: dict[str, Any] | None = None
    environment_hash: str | None = None
    seed_summary: dict[str, Any] | None = None

    event_type: EventType = field(default="run_start", init=False)


@dataclass(slots=True, kw_only=True)
class RunEndEvent(ProvenanceEvent):
    status: RunStatus = "success"
    event_count: int | None = None

    event_type: EventType = field(default="run_end", init=False)


@dataclass(slots=True, kw_only=True)
class DatasetRegisteredEvent(ProvenanceEvent):
    dataset_id: str
    name: str
    role: str
    uri: str | None = None
    version_hint: str | None = None
    fingerprint: str | None = None
    fingerprint_method: str | None = None
    sample_id_scheme: str | None = None
    sample_id_resolver: str | None = None

    event_type: EventType = field(default="dataset_registered", init=False)


@dataclass(slots=True, kw_only=True)
class LoaderRegisteredEvent(ProvenanceEvent):
    loader_id: str
    dataset_registration_event_id: str
    role: str

    event_type: EventType = field(default="loader_registered", init=False)


@dataclass(slots=True, kw_only=True)
class TransformDeclaredEvent(ProvenanceEvent):
    dataset_registration_event_id: str
    transform_list: list[dict[str, Any]]
    params_hash: str
    introspection_level: Literal["full", "partial"]

    event_type: EventType = field(default="transform_declared", init=False)


@dataclass(slots=True, kw_only=True)
class BatchDeliveredEvent(ProvenanceEvent):
    loader_id: str
    global_step: int
    global_sequence: int
    batch_size: int
    batch_fingerprint: str
    sample_ids_blob: bytes | None = None

    event_type: EventType = field(default="batch_delivered", init=False)


@dataclass(slots=True, kw_only=True)
class EnvironmentSnapshotEvent(ProvenanceEvent):
    python_version: str
    library_versions_hash: str | None = None
    hardware_summary: str | None = None
    cuda_version: str | None = None

    event_type: EventType = field(default="environment_snapshot", init=False)
