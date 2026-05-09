"""
SQLite-based provenance store implementation.

Implements the Store interface for PyPyrus.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pypyrus.provenance.events import (
    ProvenanceEvent,
    RunStartEvent,
    RunEndEvent,
    DatasetRegisteredEvent,
    LoaderRegisteredEvent,
    TransformDeclaredEvent,
    BatchDeliveredEvent,
    EnvironmentSnapshotEvent,
)

from pypyrus.storage.store import Store
from pypyrus.storage.migrate import load_schema


class SQLiteStore(Store):
    """
    Append-only provenance store backed by SQLite.

    Uses a thread-local connection model for thread safety.
    """

    def __init__(self, db_path: str | Path = "pypyrus.db"):
        self.db_path = str(db_path)
        self._local = threading.local()
        self.initialize()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize schema."""
        conn = self._get_conn()
        load_schema(conn)
        self._ensure_runs_metadata_columns(conn)
        self._ensure_dataset_metadata_columns(conn)
        # self._validate_schema_compatibility(conn)

    def close(self) -> None:
        """Close connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def flush(self) -> None:
        """Commit current transaction."""
        self._get_conn().commit()

    def _ensure_dataset_metadata_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        if "sample_id_scheme" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN sample_id_scheme TEXT")
        if "sample_id_resolver" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN sample_id_resolver TEXT")
        conn.commit()

    def _ensure_runs_metadata_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "config_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN config_json TEXT")
        conn.commit()

    # ------------------------------------------------------------------
    # Event Writing
    # ------------------------------------------------------------------

    def append_event(self, event: ProvenanceEvent) -> None:
        """
        Persist a provenance event.
        """

        if isinstance(event, RunStartEvent):
            self._insert_run_start(event)

        elif isinstance(event, RunEndEvent):
            self._insert_run_end(event)

        elif isinstance(event, DatasetRegisteredEvent):
            self._insert_dataset_registered(event)

        elif isinstance(event, LoaderRegisteredEvent):
            self._insert_loader_registered(event)

        elif isinstance(event, TransformDeclaredEvent):
            self._insert_transform_declared(event)

        elif isinstance(event, BatchDeliveredEvent):
            self._insert_batch_delivered(event)

        elif isinstance(event, EnvironmentSnapshotEvent):
            self._insert_environment_snapshot(event)

        else:
            raise ValueError(f"Unsupported event type: {type(event)}")

    # ------------------------------------------------------------------
    # Event inserts
    # ------------------------------------------------------------------

    def _insert_run_start(self, event: RunStartEvent) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO runs (
                run_id,
                start_time,
                code_ref,
                config_ref,
                config_json,
                environment_hash,
                seed_summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.timestamp,
                event.code_ref,
                event.config_ref,
                json.dumps(event.config_json) if event.config_json is not None else None,
                event.environment_hash,
                json.dumps(event.seed_summary) if event.seed_summary is not None else None,
            ),
        )

    def _insert_run_end(self, event: RunEndEvent) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE runs
            SET end_time = ?, status = ?, event_count = ?
            WHERE run_id = ?
            """,
            (event.timestamp, event.status, event.event_count, event.run_id),
        )

    def _insert_dataset_registered(self, event: DatasetRegisteredEvent) -> None:
        conn = self._get_conn()

        conn.execute(
            """
            INSERT OR IGNORE INTO datasets (
                event_id,
                dataset_id,
                name,
                uri,
                version_hint,
                fingerprint,
                fingerprint_method,
                sample_id_scheme,
                sample_id_resolver,
                registered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.dataset_id,
                event.name,
                event.uri,
                event.version_hint,
                event.fingerprint,
                event.fingerprint_method,
                event.sample_id_scheme,
                event.sample_id_resolver,
                event.timestamp,
            ),
        )

        conn.execute(
            """
            UPDATE datasets
            SET
                sample_id_scheme = COALESCE(sample_id_scheme, ?),
                sample_id_resolver = COALESCE(sample_id_resolver, ?)
            WHERE dataset_id = ?
            """,
            (
                event.sample_id_scheme,
                event.sample_id_resolver,
                event.dataset_id,
            ),
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO run_datasets (
                run_id,
                dataset_id,
                registered_at,
                role
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.dataset_id,
                event.timestamp,
                event.role,
            ),
        )

    def _insert_transform_declared(self, event: TransformDeclaredEvent) -> None:
        conn = self._get_conn()

        conn.execute(
            """
            INSERT INTO transform_declared (
                event_id,
                run_id,
                dataset_id,
                timestamp,
                transform_list_json,
                params_hash,
                introspection_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.run_id,
                event.dataset_id,
                event.timestamp,
                json.dumps(event.transform_list),
                event.params_hash,
                event.introspection_level,
            ),
        )

    def _insert_loader_registered(self, event: LoaderRegisteredEvent) -> None:
        conn = self._get_conn()

        conn.execute(
            """
            INSERT INTO loaders (
                event_id,
                loader_id,
                run_id,
                dataset_id,
                role,
                registered_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.loader_id,
                event.run_id,
                event.dataset_id,
                event.role,
                event.timestamp,
            ),
        )

    def _insert_batch_delivered(self, event: BatchDeliveredEvent) -> None:
        conn = self._get_conn()

        conn.execute(
            """
            INSERT INTO batch_delivered (
                event_id,
                run_id,
                loader_id,
                dataset_id,
                global_step,
                global_sequence,
                timestamp,
                batch_size,
                batch_fingerprint,
                sample_ids_blob
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.run_id,
                event.loader_id,
                event.dataset_id,
                event.global_step,
                event.global_sequence,
                event.timestamp,
                event.batch_size,
                event.batch_fingerprint,
                event.sample_ids_blob,
            ),
        )

    def _insert_environment_snapshot(self, event: EnvironmentSnapshotEvent) -> None:
        conn = self._get_conn()

        conn.execute(
            """
            INSERT INTO environment_snapshot (
                event_id,
                run_id,
                timestamp,
                python_version,
                library_versions_hash,
                hardware_summary,
                cuda_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.run_id,
                event.timestamp,
                event.python_version,
                event.library_versions_hash,
                event.hardware_summary,
                event.cuda_version,
            ),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Basic event query interface.
        """

        conn = self._get_conn()

        if event_type == "batch_delivered":
            rows = conn.execute(
                """
                SELECT
                    b.*,
                    l.role
                FROM batch_delivered b
                JOIN loaders l ON l.loader_id = b.loader_id
                WHERE b.run_id = ?
                ORDER BY b.global_sequence
                """,
                (run_id,),
            ).fetchall()

        elif event_type == "dataset_registered":
            rows = conn.execute(
                """
                SELECT
                    d.event_id, d.dataset_id, d.name, d.uri,
                    d.version_hint, d.fingerprint, d.fingerprint_method,
                    d.sample_id_scheme, d.sample_id_resolver,
                    d.registered_at AS timestamp,
                    rd.run_id, rd.role
                FROM datasets d
                JOIN run_datasets rd ON rd.dataset_id = d.dataset_id
                WHERE rd.run_id = ?
                ORDER BY rd.registered_at
                """,
                (run_id,),
            ).fetchall()

        elif event_type == "loader_registered":
            rows = conn.execute(
                """
                SELECT
                    event_id,
                    loader_id,
                    run_id,
                    dataset_id,
                    role,
                    registered_at AS timestamp
                FROM loaders
                WHERE run_id = ?
                ORDER BY registered_at
                """,
                (run_id,),
            ).fetchall()

        elif event_type == "transform_declared":
            rows = conn.execute(
                """
                SELECT * FROM transform_declared
                WHERE run_id = ?
                ORDER BY timestamp
                """,
                (run_id,),
            ).fetchall()

        elif event_type == "environment_snapshot":
            rows = conn.execute(
                """
                SELECT * FROM environment_snapshot
                WHERE run_id = ?
                ORDER BY timestamp
                """,
                (run_id,),
            ).fetchall()

        else:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def list_runs(self) -> list[str]:
        """Return known run IDs."""

        conn = self._get_conn()

        rows = conn.execute(
            "SELECT run_id FROM runs ORDER BY start_time DESC"
        ).fetchall()

        return [r["run_id"] for r in rows]
