from __future__ import annotations

import gzip
import json

from pypyrus.cli.main import main
from pypyrus.provenance.events import (
    BatchDeliveredEvent,
    DatasetRegisteredEvent,
    EnvironmentSnapshotEvent,
    LoaderRegisteredEvent,
    RunEndEvent,
    RunStartEvent,
    TransformDeclaredEvent,
)
from pypyrus.provenance.fingerprints import hash_json
from pypyrus.storage.sqlite_store import SQLiteStore


def test_runs_list_and_show_render_expected_output(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_runs.db"
    store = SQLiteStore(db_path)

    _seed_run(
        store,
        run_id="run-002",
        status="failure",
        role="train",
        sample_ids=[0, 1],
        fingerprint="batch-b",
    )
    _seed_run(
        store,
        run_id="run-001",
        status="success",
        role="train",
        sample_ids=[0, 1, 2],
        fingerprint="batch-a",
    )
    store.close()

    exit_code = main(["--db", str(db_path), "runs", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "RUN ID" in captured.out
    assert "DURATION" in captured.out
    assert "DATASETS" in captured.out
    assert "LOADERS" in captured.out
    assert "ROLES" in captured.out
    assert "BATCHES" in captured.out
    assert "run-001" in captured.out
    assert "run-002" in captured.out
    assert "10m00s" in captured.out
    assert "train" in captured.out

    exit_code = main(["--db", str(db_path), "runs", "show", "run-001"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run overview" in captured.out
    assert "Run ID: run-001" in captured.out
    assert "Duration: 10m00s" in captured.out
    assert "Code ref: git:run-001:clean" in captured.out
    assert "Config ref: config-run-001" in captured.out
    assert "Environment hash: env-run-001" in captured.out
    assert 'Seed summary: {"global_seed": 7}' in captured.out
    assert "Summary" in captured.out
    assert "Batch counts by role: train=1" in captured.out
    assert "Datasets" in captured.out
    assert "fingerprint: fingerprint-run-001" in captured.out
    assert "Loaders" in captured.out
    assert "loader-run-001-train" in captured.out
    assert "Transforms" in captured.out
    assert "ScaleTransform" in captured.out
    assert "Environment" in captured.out
    assert "Library versions hash: lib-hash" in captured.out
    assert 'Hardware summary: {"system":"Darwin"}' in captured.out


def test_compare_and_batch_show_support_json_output(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_compare.db"
    store = SQLiteStore(db_path)

    _seed_run(
        store,
        run_id="run-a",
        status="success",
        role="train",
        sample_ids=[0, 1],
        fingerprint="shared-batch",
    )
    _seed_run(
        store,
        run_id="run-b",
        status="success",
        role="train",
        sample_ids=[0, 1],
        fingerprint="different-batch",
    )
    store.close()

    exit_code = main(["--db", str(db_path), "--json", "compare", "run-a", "run-b"])
    captured = capsys.readouterr()

    comparison = json.loads(captured.out)
    assert exit_code == 0
    assert comparison["run_id_a"] == "run-a"
    assert comparison["run_id_b"] == "run-b"
    assert comparison["roles_compared"] == ["train"]
    assert comparison["dataset_identities_match"] is False
    assert comparison["batch_streams_match"] is False
    assert comparison["roles"]["train"]["fully_matches"] is False
    assert comparison["roles"]["train"]["reason"] == "dataset_fingerprint_mismatch"
    assert comparison["roles"]["train"]["dataset_identity_matches"] is False
    assert comparison["roles"]["train"]["batch_stream_matches"] is False

    exit_code = main(
        ["--db", str(db_path), "--json", "batches", "show", "run-a", "--step", "0"]
    )
    captured = capsys.readouterr()

    batch = json.loads(captured.out)
    assert exit_code == 0
    assert batch["run_id"] == "run-a"
    assert batch["global_step"] == 0
    assert batch["sample_ids"] == [0, 1]


def test_missing_db_returns_nonzero(tmp_path, capsys) -> None:
    missing_db = tmp_path / "missing.db"

    exit_code = main(["--db", str(missing_db), "runs", "list"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "database not found" in captured.err.lower()


def _seed_run(
    store: SQLiteStore,
    *,
    run_id: str,
    status: str,
    role: str,
    sample_ids: list[int],
    fingerprint: str,
) -> None:
    dataset_id = f"dataset-{run_id}"
    loader_id = f"loader-{run_id}-{role}"
    transform_list = [
        {
            "name": "ScaleTransform",
            "module": "tests.cli",
            "params": {"scale": 2.0},
        }
    ]

    store.append_event(
        RunStartEvent(
            run_id=run_id,
            timestamp=f"2026-03-16T00:00:0{1 if run_id.endswith('1') else 2}+00:00",
            code_ref=f"git:{run_id}:clean",
            config_ref=f"config-{run_id}",
            environment_hash=f"env-{run_id}",
            seed_summary={"global_seed": 7},
        )
    )
    store.append_event(
        EnvironmentSnapshotEvent(
            run_id=run_id,
            python_version="3.12.0",
            library_versions_hash="lib-hash",
            hardware_summary='{"system":"Darwin"}',
            cuda_version=None,
        )
    )
    store.append_event(
        DatasetRegisteredEvent(
            run_id=run_id,
            dataset_id=dataset_id,
            name="TinyDataset",
            role=role,
            fingerprint=f"fingerprint-{run_id}",
            fingerprint_method="in_memory_deterministic_v1",
        )
    )
    store.append_event(
        TransformDeclaredEvent(
            run_id=run_id,
            dataset_id=dataset_id,
            transform_chain_id=hash_json(transform_list),
            transform_list=transform_list,
            params_hash=hash_json(transform_list),
            introspection_level="full",
        )
    )
    store.append_event(
        LoaderRegisteredEvent(
            run_id=run_id,
            loader_id=loader_id,
            dataset_id=dataset_id,
            role=role,
        )
    )
    store.append_event(
        BatchDeliveredEvent(
            run_id=run_id,
            loader_id=loader_id,
            dataset_id=dataset_id,
            global_step=0,
            global_sequence=0,
            batch_size=len(sample_ids),
            batch_fingerprint=fingerprint,
            sample_ids_blob=gzip.compress(json.dumps(sample_ids).encode("utf-8")),
        )
    )
    store.append_event(
        RunEndEvent(
            run_id=run_id,
            status=status,
            timestamp=f"2026-03-16T00:10:0{1 if run_id.endswith('1') else 2}+00:00",
        )
    )
    store.flush()
