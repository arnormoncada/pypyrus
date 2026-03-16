from __future__ import annotations

import warnings
from typing import Iterable
from uuid import uuid4

from pypyrus.core.environment import collect_environment_snapshot
from pypyrus.provenance.events import (
    EnvironmentSnapshotEvent,
    ProvenanceEvent,
    RunEndEvent,
    RunStartEvent,
)

from pypyrus.storage.store import Store
from pypyrus.storage.sqlite_store import SQLiteStore


class Run:
    """
    Represents a single training execution.

    The Run coordinates event emission and persistence.
    """

    def __init__(
        self,
        store: Store | None = None,
        run_id: str | None = None,
        # run_name: str | None = None,
    ):
        self.run_id = run_id or str(uuid4())
        # self.run_name = run_name
        self.store = store or SQLiteStore()

        self._started = False
        self._ended = False
        self._batch_sequence: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        code_ref: str | None = None,
        config_ref: str | None = None,
        environment_hash: str | None = None,
        seed_summary: dict | None = None,
    ) -> None:
        """
        Start the run and emit RunStartEvent.
        """

        if self._started:
            raise RuntimeError("Run already started")

        event = RunStartEvent(
            run_id=self.run_id,
            code_ref=code_ref,
            config_ref=config_ref,
            environment_hash=environment_hash,
            seed_summary=seed_summary,
            # run_name=self.run_name,
        )

        self.emit(event)

        # Emit a best-effort environment snapshot once at run start.
        try:
            env = collect_environment_snapshot()
            self.emit(
                EnvironmentSnapshotEvent(
                    run_id=self.run_id,
                    python_version=env["python_version"] or "unknown",
                    library_versions_hash=env["library_versions_hash"],
                    hardware_summary=env["hardware_summary"],
                    cuda_version=env["cuda_version"],
                )
            )
        except Exception as exc:  # pragma: no cover - defensive safety path
            warnings.warn(
                f"PyPyrus environment snapshot capture failed: {exc.__class__.__name__}",
                stacklevel=2,
            )

        self._started = True

    def end(self, status: str = "success") -> None:
        """
        End the run and emit RunEndEvent.
        """

        if self._ended:
            return

        event = RunEndEvent(
            run_id=self.run_id,
            status=status,
        )

        self.emit(event)

        self.store.flush()

        self._ended = True

    # ------------------------------------------------------------------
    # Batch sequence
    # ------------------------------------------------------------------

    def next_batch_sequence(self) -> int:
        """
        Return the next run-level batch sequence number and advance the counter.

        This gives every BatchDeliveredEvent a unique, monotonically increasing
        position in the run timeline, regardless of which loader emitted it.
        """
        seq = self._batch_sequence
        self._batch_sequence += 1
        return seq

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def emit(self, event: ProvenanceEvent) -> None:
        """
        Emit a provenance event.

        This forwards the event to the configured store.
        """

        if event.run_id != self.run_id:
            raise ValueError(
                f"Event run_id ({event.run_id}) does not match Run ({self.run_id})"
            )

        self.store.append_event(event)

    def emit_many(self, events: Iterable[ProvenanceEvent]) -> None:
        """
        Emit multiple events.
        """

        for event in events:
            self.emit(event)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Run":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "failure" if exc else "success"
        self.end(status=status)
        self.store.close()