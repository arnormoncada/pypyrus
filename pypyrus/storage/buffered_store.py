from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Literal

from pypyrus.provenance.events import ProvenanceEvent
from pypyrus.storage.store import Store


_MessageKind = Literal["event", "flush", "stop"]

 
@dataclass(slots=True)
class _EventMessage:
    event: ProvenanceEvent


@dataclass(slots=True)
class _FlushMessage:
    target_seq: int
    done: threading.Event
    error: Exception | None = None


@dataclass(slots=True)
class _StopMessage:
    done: threading.Event
    error: Exception | None = None


@dataclass(slots=True)
class _QueueMessage:
    kind: _MessageKind
    payload: _EventMessage | _FlushMessage | _StopMessage


class BufferedStore(Store):
    """
    Strict buffered writer wrapper around another Store.

    Producer threads enqueue events; a single writer thread owns durable writes
    (`append_event`, `flush`, and `close`) against the wrapped store.
    """

    def __init__(
        self,
        inner_store: Store,
        *,
        queue_size: int = 1024,
    ):
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")

        self._inner = inner_store
        self._queue: queue.Queue[_QueueMessage] = queue.Queue(maxsize=queue_size)
        self._enqueue_lock = threading.Lock()
        self._next_seq = 0
        self._written_seq = -1
        self._closed = False

        self._failure_lock = threading.Lock()
        self._writer_failure: Exception | None = None

        self._writer = threading.Thread(
            target=self._writer_loop,
            name="pypyrus-buffered-store-writer",
            daemon=True,
        )
        self._writer.start()

    def initialize(self) -> None:
        # The wrapped store is already initialized by construction.
        return None

    def append_event(self, event: ProvenanceEvent) -> None:
        self._raise_if_failed_or_closed()

        with self._enqueue_lock:
            self._next_seq += 1
            message = _QueueMessage(kind="event", payload=_EventMessage(event=event))
            # Strict mode: block until there is room.
            self._queue.put(message)

        self._raise_if_failed_or_closed()

    def flush(self) -> None:
        self._raise_if_failed_or_closed()

        with self._enqueue_lock:
            target_seq = self._next_seq - 1
            done = threading.Event()
            request = _FlushMessage(target_seq=target_seq, done=done)
            self._queue.put(_QueueMessage(kind="flush", payload=request))

        done.wait()
        if request.error is not None:
            raise request.error
        self._raise_if_failed_or_closed()

    def close(self) -> None:
        if self._closed:
            return

        # Ensure all already-enqueued events are durably persisted first.
        self.flush()

        done = threading.Event()
        request = _StopMessage(done=done)
        self._queue.put(_QueueMessage(kind="stop", payload=request))
        done.wait()

        if request.error is not None:
            raise request.error

        self._writer.join(timeout=5.0)
        self._raise_if_failed_or_closed()
        self._closed = True

    def get_events(self, run_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        # Provide read-your-writes semantics from caller perspective.
        self.flush()
        return self._inner.get_events(run_id=run_id, event_type=event_type)

    def list_runs(self) -> list[str]:
        self.flush()
        return self._inner.list_runs()

    def _writer_loop(self) -> None:
        try:
            while True:
                message = self._queue.get()
                if message.kind == "event":
                    payload = message.payload
                    assert isinstance(payload, _EventMessage)
                    self._inner.append_event(payload.event)
                    self._written_seq += 1
                    self._queue.task_done()
                    continue

                if message.kind == "flush":
                    payload = message.payload
                    assert isinstance(payload, _FlushMessage)
                    try:
                        # FIFO queue order guarantees all pre-flush events have
                        # already been written before this point.
                        self._inner.flush()
                    except Exception as exc:  # pragma: no cover - safety path
                        payload.error = exc
                        self._record_failure(exc)
                    finally:
                        payload.done.set()
                        self._queue.task_done()
                    continue

                payload = message.payload
                assert isinstance(payload, _StopMessage)
                try:
                    self._inner.flush()
                    self._inner.close()
                except Exception as exc:  # pragma: no cover - safety path
                    payload.error = exc
                    self._record_failure(exc)
                finally:
                    payload.done.set()
                    self._queue.task_done()
                return
        except Exception as exc:  # pragma: no cover - defensive safety path
            self._record_failure(exc)

    def _record_failure(self, exc: Exception) -> None:
        with self._failure_lock:
            if self._writer_failure is None:
                self._writer_failure = exc

    def _raise_if_failed_or_closed(self) -> None:
        if self._closed:
            raise RuntimeError("BufferedStore is already closed")
        with self._failure_lock:
            if self._writer_failure is not None:
                raise RuntimeError("BufferedStore writer failed") from self._writer_failure
