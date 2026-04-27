from __future__ import annotations

import json
import threading
import time

import pytest

from pypyrus.core.run import Run
from pypyrus.provenance.events import RunStartEvent
from pypyrus.storage.buffered_store import BufferedStore
from pypyrus.storage.sqlite_store import SQLiteStore
from pypyrus.storage.store import Store

from tests.helpers import fetch_one


class RecordingStore(Store):
    """Test store that records writes and can block append calls."""

    def __init__(self):
        self.events = []
        self.flush_calls = 0
        self.closed = False
        self.append_thread_ids: list[int] = []
        self.flush_thread_ids: list[int] = []
        self.close_thread_ids: list[int] = []
        self.block_first_append = False
        self._first_append_seen = False
        self._append_gate = threading.Event()
        self._lock = threading.Lock()

    def initialize(self) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self.close_thread_ids.append(threading.get_ident())

    def append_event(self, event) -> None:
        with self._lock:
            self.append_thread_ids.append(threading.get_ident())
            should_block = self.block_first_append and not self._first_append_seen
            if should_block:
                self._first_append_seen = True

        if should_block:
            self._append_gate.wait(timeout=2.0)

        with self._lock:
            self.events.append(event)

    def flush(self) -> None:
        with self._lock:
            self.flush_calls += 1
            self.flush_thread_ids.append(threading.get_ident())

    def get_events(self, run_id: str, event_type: str | None = None):
        return []

    def list_runs(self) -> list[str]:
        return []

    def release_first_append(self) -> None:
        self._append_gate.set()



def _event(run_id: str, idx: int) -> RunStartEvent:
    return RunStartEvent(run_id=run_id, code_ref=f"code:{idx}")


def test_buffered_store_strict_mode_blocks_when_queue_is_full() -> None:
    inner = RecordingStore()
    inner.block_first_append = True
    buffered = BufferedStore(inner, queue_size=1)

    buffered.append_event(_event("r1", 1))
    buffered.append_event(_event("r1", 2))

    third_append_done = threading.Event()

    def append_third() -> None:
        buffered.append_event(_event("r1", 3))
        third_append_done.set()

    thread = threading.Thread(target=append_third)
    thread.start()

    # Queue is full until writer can proceed past the blocked first append.
    time.sleep(0.1)
    assert not third_append_done.is_set()

    inner.release_first_append()
    thread.join(timeout=2.0)
    assert third_append_done.is_set()

    buffered.close()


def test_buffered_store_flush_waits_for_preflush_events() -> None:
    inner = RecordingStore()
    inner.block_first_append = True
    buffered = BufferedStore(inner, queue_size=4)

    buffered.append_event(_event("r2", 1))

    flush_done = threading.Event()

    def do_flush() -> None:
        buffered.flush()
        flush_done.set()

    thread = threading.Thread(target=do_flush)
    thread.start()

    # Flush should wait while the first append is still blocked.
    time.sleep(0.1)
    assert not flush_done.is_set()

    inner.release_first_append()
    thread.join(timeout=2.0)
    assert flush_done.is_set()

    assert len(inner.events) == 1
    assert inner.flush_calls == 1

    buffered.close()


# def test_buffered_store_warns_when_queue_backpressure_blocks() -> None:
#     inner = RecordingStore()
#     inner.block_first_append = True
#     buffered = BufferedStore(inner, queue_size=1)

#     buffered.append_event(_event("rb", 1))

#     with pytest.warns(UserWarning, match="queue is full"):
#         buffered.append_event(_event("rb", 2))

#     releaser = threading.Thread(target=lambda: (time.sleep(0.05), inner.release_first_append()))
#     releaser.start()

#     buffered.append_event(_event("rb", 3))

#     releaser.join(timeout=2.0)
#     buffered.close()


def test_buffered_store_writer_thread_owns_inner_store_calls() -> None:
    inner = RecordingStore()
    buffered = BufferedStore(inner, queue_size=4)

    main_thread_id = threading.get_ident()

    buffered.append_event(_event("r3", 1))
    buffered.flush()
    buffered.close()

    assert inner.append_thread_ids
    assert inner.flush_thread_ids
    assert inner.close_thread_ids

    assert all(thread_id != main_thread_id for thread_id in inner.append_thread_ids)
    assert all(thread_id != main_thread_id for thread_id in inner.flush_thread_ids)
    assert all(thread_id != main_thread_id for thread_id in inner.close_thread_ids)


def test_run_can_switch_to_buffered_strict_mode(db_path) -> None:
    sqlite_store = SQLiteStore(db_path)

    with Run(
        store=sqlite_store,
        store_mode="buffered_strict",
        buffered_queue_size=16,
    ) as run:
        pass

    row = fetch_one(
        db_path,
        "SELECT run_id, status, config_ref, config_json FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert row["run_id"] == run.run_id
    assert row["status"] == "success"
    assert row["config_ref"] is not None

    raw_config_json = row["config_json"]
    assert raw_config_json is not None
    parsed = json.loads(raw_config_json)
    assert parsed["pypyrus"]["store_mode"] == "buffered_strict"
    assert parsed["pypyrus"]["buffered_queue_size"] == 16


def test_run_rejects_unknown_store_mode(store) -> None:
    with pytest.raises(ValueError, match="store_mode"):
        Run(store=store, store_mode="invalid")


def test_buffered_store_flush_call_cut_holds_under_concurrent_producers() -> None:
    inner = RecordingStore()
    buffered = BufferedStore(inner, queue_size=256)

    counter_lock = threading.Lock()
    next_event_id = 0
    completed_ids: set[int] = set()
    stop = threading.Event()

    def producer() -> None:
        nonlocal next_event_id
        while not stop.is_set():
            with counter_lock:
                next_event_id += 1
                idx = next_event_id
            buffered.append_event(_event("r4", idx))
            with counter_lock:
                completed_ids.add(idx)

    producers = [threading.Thread(target=producer) for _ in range(3)]
    for thread in producers:
        thread.start()

    # Let producers enqueue concurrently before establishing the flush cut.
    time.sleep(0.15)
    with counter_lock:
        completed_before_flush = set(completed_ids)

    buffered.flush()

    # All events enqueued before flush-call must be persisted when flush returns.
    persisted_ids = {
        int(event.code_ref.split(":", 1)[1])
        for event in inner.events
        if event.code_ref is not None and event.code_ref.startswith("code:")
    }
    assert completed_before_flush.issubset(persisted_ids)

    stop.set()
    for thread in producers:
        thread.join(timeout=2.0)

    buffered.close()
