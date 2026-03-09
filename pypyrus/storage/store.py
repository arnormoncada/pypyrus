from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Any

from pypyrus.provenance.events import ProvenanceEvent


class Store(ABC):
    """
    Abstract base class for provenance storage backends.

    A Store is responsible for persisting provenance events and
    retrieving them for analysis.

    Implementations must guarantee append-only semantics.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the storage backend.

        This typically creates database schema if it does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close any underlying resources (database connections, files, etc.).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Event Writing
    # ------------------------------------------------------------------

    @abstractmethod
    def append_event(self, event: ProvenanceEvent) -> None:
        """
        Persist a single provenance event.
        """
        raise NotImplementedError

    def append_events(self, events: Iterable[ProvenanceEvent]) -> None:
        """
        Persist multiple events.

        Default implementation calls append_event sequentially,
        but backends may override this for batch efficiency.
        """
        for event in events:
            self.append_event(event)

    @abstractmethod
    def flush(self) -> None:
        """
        Flush buffered writes to durable storage.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve events for a given run.

        Parameters
        ----------
        run_id:
            The run identifier.

        event_type:
            Optional filter for a specific event type.

        Returns
        -------
        List of event dictionaries.
        """
        raise NotImplementedError

    @abstractmethod
    def list_runs(self) -> list[str]:
        """
        Return all known run IDs.
        """
        raise NotImplementedError