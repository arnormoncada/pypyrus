from __future__ import annotations

import pytest

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
