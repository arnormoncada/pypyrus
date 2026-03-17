from __future__ import annotations

import pytest

from pypyrus.core import run as run_module
from pypyrus.core.run import Run

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
