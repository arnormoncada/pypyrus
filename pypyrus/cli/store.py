from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pypyrus.storage.sqlite_store import SQLiteStore


DEFAULT_DB_PATH = "pypyrus.db"


def resolve_db_path(explicit_path: str | None = None) -> Path:
    """Resolve the database path from CLI arg, env var, or default."""
    raw = explicit_path or os.environ.get("PYPYRUS_DB") or DEFAULT_DB_PATH
    return Path(raw).expanduser()


@contextmanager
def open_query_store(explicit_path: str | None = None) -> Iterator[SQLiteStore]:
    """
    Open an existing SQLite store for read/query operations.

    Query commands fail fast when the target DB path does not exist to avoid
    silently creating an empty database on typos.
    """
    db_path = resolve_db_path(explicit_path)
    if not db_path.exists():
        raise FileNotFoundError(f"PyPyrus database not found: {db_path}")

    store = SQLiteStore(db_path)
    try:
        yield store
    finally:
        store.close()
