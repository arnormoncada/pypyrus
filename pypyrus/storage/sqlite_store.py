"""
SQLite-based provenance store implementation.

Simple, portable, good for thesis MVP. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from pypyrus.storage.migrate import load_schema


class SQLiteStore:
    """
    Append-only provenance store backed by SQLite.

    Thread-safe via connection-per-thread pattern.
    """

    def __init__(self, db_path: str | Path = "pypyrus.db"):
        """
        Initialize the store.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory.
        """
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for cursor with auto-commit."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Load schema from SQL files."""
        load_schema(self._get_conn())

    # =========================================================================
    # RUNS
    # =========================================================================

    def create_run(
        self,
        run_id: UUID,
        start_time: datetime | None = None,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
        config_hash: str | None = None,
        env_hash: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Record a new run."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, start_time, git_commit, git_dirty,
                    config_hash, env_hash, tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    (start_time or datetime.now()).isoformat(),
                    git_commit,
                    1 if git_dirty else 0 if git_dirty is not None else None,
                    config_hash,
                    env_hash,
                    json.dumps(tags) if tags else None,
                ),
            )

    def end_run(self, run_id: UUID, end_time: datetime | None = None) -> None:
        """Mark a run as finished."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE runs SET end_time = ? WHERE run_id = ?",
                ((end_time or datetime.now()).isoformat(), str(run_id)),
            )

    def get_run(self, run_id: UUID) -> dict[str, Any] | None:
        """Retrieve a run by ID."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE run_id = ?", (str(run_id),))
            row = cur.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # DATASETS
    # =========================================================================

    def register_dataset(
        self,
        dataset_id: UUID,
        name: str,
        uri: str,
        fingerprint: str,
        fingerprint_strategy: str,
        version_hint: str | None = None,
    ) -> None:
        """Register a dataset (upsert - updates fingerprint if already exists)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, uri, version_hint,
                    fingerprint, fingerprint_strategy, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    fingerprint_strategy = excluded.fingerprint_strategy
                """,
                (
                    str(dataset_id),
                    name,
                    uri,
                    version_hint,
                    fingerprint,
                    fingerprint_strategy,
                    datetime.now().isoformat(),
                ),
            )

    def link_dataset_to_run(
        self,
        run_id: UUID,
        dataset_id: UUID,
        role: str | None = None,
    ) -> None:
        """Associate a dataset with a run. Role: train/val/test/unspecified."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO run_datasets (run_id, dataset_id, role, registered_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(run_id), str(dataset_id), role, datetime.now().isoformat()),
            )

    def get_datasets_for_run(self, run_id: UUID) -> list[dict[str, Any]]:
        """Get all datasets used in a run."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT d.* FROM datasets d
                JOIN run_datasets rd ON d.dataset_id = rd.dataset_id
                WHERE rd.run_id = ?
                """,
                (str(run_id),),
            )
            return [dict(row) for row in cur.fetchall()]

    # =========================================================================
    # EVENTS
    # =========================================================================

    def log_batch_consumed(
        self,
        run_id: UUID,
        dataset_id: UUID,
        global_step: int,
        batch_size: int,
        batch_fingerprint: str,
        sample_ids: list[int] | None = None,
        rng_state_hash: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Log a batch consumed event."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO batch_consumed (
                    run_id, dataset_id, global_step, timestamp,
                    batch_size, batch_fingerprint, sample_ids_json, rng_state_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(dataset_id),
                    global_step,
                    (timestamp or datetime.now()).isoformat(),
                    batch_size,
                    batch_fingerprint,
                    json.dumps(sample_ids) if sample_ids else None,
                    rng_state_hash,
                ),
            )

    def log_transform_declared(
        self,
        run_id: UUID,
        dataset_id: UUID,
        transform_chain_id: str,
        transform_list: list[dict[str, Any]],
        params_hash: str,
        seed_policy: str | None = None,
        deterministic: bool | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Log the declared transform pipeline for a dataset (once per run)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO transform_declared (
                    run_id, dataset_id, timestamp,
                    transform_chain_id, transform_list_json, params_hash,
                    seed_policy, deterministic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(dataset_id),
                    (timestamp or datetime.now()).isoformat(),
                    transform_chain_id,
                    json.dumps(transform_list),
                    params_hash,
                    seed_policy,
                    1 if deterministic else 0 if deterministic is not None else None,
                ),
            )

    def log_access_agg(
        self,
        run_id: UUID,
        dataset_id: UUID,
        operation: str,
        count: int,
        worker_id: int | None = None,
        process_id: int | None = None,
        sample_ref: Any | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Log aggregated dataset access stats."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_access_agg (
                    run_id, dataset_id, timestamp, operation,
                    worker_id, process_id, count, sample_ref_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(dataset_id),
                    (timestamp or datetime.now()).isoformat(),
                    operation,
                    worker_id,
                    process_id,
                    count,
                    json.dumps(sample_ref) if sample_ref else None,
                ),
            )

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_batches_for_run(
        self,
        run_id: UUID,
        dataset_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Query batch_consumed events for a run."""
        query = "SELECT * FROM batch_consumed WHERE run_id = ?"
        params: list[Any] = [str(run_id)]

        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(str(dataset_id))

        query += " ORDER BY global_step"

        with self._cursor() as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def get_transforms_for_run(self, run_id: UUID) -> list[dict[str, Any]]:
        """Get declared transform pipelines for a run."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM transform_declared WHERE run_id = ? ORDER BY timestamp",
                (str(run_id),),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_access_agg_for_run(self, run_id: UUID) -> list[dict[str, Any]]:
        """Get aggregated access stats for a run."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.name AS dataset_name,
                    agg.operation,
                    SUM(agg.count) AS total_accesses,
                    MIN(agg.timestamp) AS first_access,
                    MAX(agg.timestamp) AS last_access
                FROM dataset_access_agg agg
                JOIN datasets d ON agg.dataset_id = d.dataset_id
                WHERE agg.run_id = ?
                GROUP BY agg.dataset_id, agg.operation
                ORDER BY d.name, agg.operation
                """,
                (str(run_id),),
            )
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


def _hash_json(data: dict[str, Any] | None) -> str | None:
    """Create a stable hash of JSON-serializable data."""
    if data is None:
        return None
    import hashlib

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
