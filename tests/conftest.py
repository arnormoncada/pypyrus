from __future__ import annotations

from pathlib import Path

import pytest

from pypyrus.storage.sqlite_store import SQLiteStore


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "pypyrus_test.db"


@pytest.fixture
def store(db_path: Path) -> SQLiteStore:
    sqlite_store = SQLiteStore(db_path)
    try:
        yield sqlite_store
    finally:
        sqlite_store.close()
