"""
PyPyrus: A data provenance layer for transparent and reproducible ML systems.
"""

from __future__ import annotations

__all__ = [
    "Run",
    "attach",
]


def __getattr__(name: str):
    """Lazily expose the public API to avoid importing torch on CLI/query paths."""
    if name == "Run":
        from pypyrus.core.run import Run

        return Run
    if name == "attach":
        from pypyrus.core.attach import attach

        return attach
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
