from __future__ import annotations

import sqlite3

import pytest

from pypyrus.core import run as run_module
from pypyrus.core.run import Run
from pypyrus.provenance.events import DatasetRegisteredEvent
from pypyrus.storage.sqlite_store import SQLiteStore

from tests.helpers import fetch_one


def test_run_marks_failure_when_context_exits_with_exception(db_path, store) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with Run(store=store) as run:
            raise RuntimeError("boom")

    run_row = fetch_one(
        db_path,
        "SELECT status, end_time FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["status"] == "failure"
    assert run_row["end_time"] is not None


def test_run_marks_success_when_using_buffered_store_mode(db_path, store) -> None:
    with Run(store=store, store_mode="buffered_strict") as run:
        pass

    run_row = fetch_one(
        db_path,
        "SELECT status, end_time FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["status"] == "success"
    assert run_row["end_time"] is not None


def test_run_auto_captures_code_ref_when_not_provided(db_path, monkeypatch, store) -> None:
    monkeypatch.setattr(
        run_module,
        "collect_code_ref",
        lambda: "git:deadbeefdeadbeefdeadbeefdeadbeefdeadbeef:dirty",
    )

    with Run(store=store) as run:
        pass

    run_row = fetch_one(
        db_path,
        "SELECT code_ref FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["code_ref"] == "git:deadbeefdeadbeefdeadbeefdeadbeefdeadbeef:dirty"


def test_run_stores_null_code_ref_when_git_capture_is_unavailable(
    db_path,
    monkeypatch,
    store,
) -> None:
    monkeypatch.setattr(run_module, "collect_code_ref", lambda: None)

    with Run(store=store) as run:
        pass

    run_row = fetch_one(
        db_path,
        "SELECT code_ref FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["code_ref"] is None


def test_explicit_code_ref_overrides_auto_capture(db_path, monkeypatch, store) -> None:
    monkeypatch.setattr(run_module, "collect_code_ref", lambda: "git:auto:dirty")

    run = Run(store=store)
    run.start(code_ref="git:manual:clean")
    run.end()

    run_row = fetch_one(
        db_path,
        "SELECT code_ref FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["code_ref"] == "git:manual:clean"


def test_run_persists_optional_run_name(db_path, store) -> None:
    with Run(store=store, run_name="baseline-a") as run:
        pass

    run_row = fetch_one(
        db_path,
        "SELECT run_name FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["run_name"] == "baseline-a"


def test_store_adds_run_name_column_for_legacy_runs_table(db_path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            code_ref TEXT,
            config_ref TEXT,
            environment_hash TEXT,
            seed_summary_json TEXT,
            end_time TEXT,
            status TEXT,
            event_count INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    conn.close()

    assert "config_json" in columns
    assert "run_name" in columns


def test_run_end_persists_total_event_count(db_path, monkeypatch, store) -> None:
    monkeypatch.setattr(run_module, "collect_code_ref", lambda: "git:test:clean")
    monkeypatch.setattr(
        run_module,
        "collect_environment_snapshot",
        lambda: {
            "python_version": "3.11.0",
            "library_versions_hash": "libhash",
            "hardware_summary": "cpu",
            "cuda_version": None,
        },
    )

    with Run(store=store) as run:
        run.emit(
            DatasetRegisteredEvent(
                run_id=run.run_id,
                dataset_id="dataset-001",
                name="dummy",
                role="train",
                fingerprint="abc123",
                fingerprint_method="path",
            )
        )

    run_row = fetch_one(
        db_path,
        "SELECT event_count FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["event_count"] == 4
