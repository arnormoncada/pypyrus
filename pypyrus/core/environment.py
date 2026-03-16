"""Environment snapshot helpers for run-level provenance.

This module collects a compact, best-effort snapshot of the execution
environment at run start. It is designed to enrich reproducibility evidence
without breaking training when certain details are unavailable.

Captured fields:
- python_version
- library_versions_hash (deterministic hash of installed package versions)
- hardware_summary (canonical JSON of basic platform/hardware info)
- cuda_version (optional; detected through torch when available)

All collectors are resilient: unavailable sources return None rather than
raising, so snapshot emission remains non-fatal.
"""

from __future__ import annotations

import platform
from importlib import import_module
from importlib import metadata
from typing import Any

from pypyrus.provenance.fingerprints import hash_json, stable_json_dumps


def _collect_library_versions_hash() -> str | None:
    """Hash installed package versions in a deterministic way."""
    try:
        packages: list[dict[str, str]] = []
        for dist in metadata.distributions():
            name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.name
            version = dist.version
            if name and version:
                packages.append({"name": str(name), "version": str(version)})
        packages.sort(key=lambda p: (p["name"].lower(), p["version"]))
        return hash_json(packages)
    except Exception:
        return None


def _collect_hardware_summary() -> str:
    """Capture a compact hardware and platform summary as canonical JSON."""
    summary: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }
    return stable_json_dumps(summary)


def _collect_cuda_version() -> str | None:
    """Best-effort CUDA version detection via torch when available."""
    try:
        torch = import_module("torch")

        cuda_version = getattr(torch.version, "cuda", None)
        return str(cuda_version) if cuda_version else None
    except Exception:
        return None


def collect_environment_snapshot() -> dict[str, str | None]:
    """Collect fields used by EnvironmentSnapshotEvent with safe fallbacks."""
    return {
        "python_version": platform.python_version(),
        "library_versions_hash": _collect_library_versions_hash(),
        "hardware_summary": _collect_hardware_summary(),
        "cuda_version": _collect_cuda_version(),
    }
