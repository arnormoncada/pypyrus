"""
Persistence backends for provenance events (e.g. SQLite).

Storage knows how to store/query events, but is framework-agnostic.
"""

from __future__ import annotations

from .store import Store
from .sqlite_store import SQLiteStore

__all__ = [
    "Store",
    "SQLiteStore",
]