"""Smoke-check the installed PyPyrus package runtime.

This script is intended for CI packaging validation. It checks that the
installed package can be imported, the CLI parser can be constructed, the
packaged SQLite schema is available at runtime, and a minimal Run can be
created and persisted through SQLiteStore.
"""

from __future__ import annotations

import sqlite3
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    import_module("pypyrus")
    cli_main = import_module("pypyrus.cli.main")
    build_parser = getattr(cli_main, "build_parser")
    parser = build_parser()
    parser.parse_args(["runs", "list"])

    from pypyrus.core.run import Run
    from pypyrus.storage.sqlite_store import SQLiteStore

    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "sanity.db"

        store = SQLiteStore(db_path)
        try:
            with Run(store=store) as run:
                pass
        finally:
            store.close()

        conn = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

            run_row = conn.execute(
                "SELECT run_id, status, event_count FROM runs WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()

            environment_count = conn.execute(
                "SELECT COUNT(*) FROM environment_snapshot WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()[0]
        finally:
            conn.close()

    required_tables = {
        "runs",
        "datasets",
        "run_datasets",
        "batch_delivered",
        "transform_declared",
        "environment_snapshot",
        "loaders",
    }
    missing = sorted(required_tables - tables)
    if missing:
        raise RuntimeError(
            f"Missing expected tables after runtime initialization: {', '.join(missing)}"
        )

    if run_row is None:
        raise RuntimeError("Expected one persisted run row, but none was found.")

    if run_row[0] != run.run_id:
        raise RuntimeError("Persisted run row does not match the created run ID.")

    if run_row[1] != "success":
        raise RuntimeError(f"Expected run status 'success', got: {run_row[1]!r}")

    expected_event_count = 3 if environment_count else 2
    if run_row[2] != expected_event_count:
        raise RuntimeError(
            "Unexpected run event_count. "
            f"Expected {expected_event_count}, got {run_row[2]!r}."
        )

    print("Installed package runtime checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
