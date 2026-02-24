"""
Simple schema loader for PyPyrus.

Reads .sql files from the schema/ directory and executes them in order.
Uses CREATE TABLE IF NOT EXISTS so it's safe to run multiple times.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def load_schema(conn: sqlite3.Connection, schema_dir: Path | None = None) -> None:
    """
    Load and execute all .sql files from the schema directory.

    Args:
        conn: SQLite connection.
        schema_dir: Directory containing .sql files. Defaults to pypyrus/storage/schema/.
    """
    if schema_dir is None:
        schema_dir = Path(__file__).parent / "schema"
        print(f"Using default schema directory: {schema_dir}")

    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")

    for filepath in sorted(schema_dir.glob("*.sql")):
        sql = filepath.read_text()
        conn.executescript(sql)
        # print(f"Executed sql: {sql}... (from {filepath.name})")

    conn.commit()


def init_db(db_path: str, schema_dir: Path | None = None) -> sqlite3.Connection:
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file.
        schema_dir: Directory containing .sql files (optional).

    Returns:
        SQLite connection with schema applied.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    load_schema(conn, schema_dir)
    return conn
