"""
Core orchestration: Run lifecycle, dataset identity, and top-level attach logic.
"""

from __future__ import annotations

__all__ = [
    "Run",
    "attach",
]


def __getattr__(name: str):
    if name == "Run":
        from pypyrus.core.run import Run

        return Run
    if name == "attach":
        from pypyrus.core.attach import attach

        return attach
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
