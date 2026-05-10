from __future__ import annotations

import gzip
import json
from pathlib import Path

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
from tests.helpers import TinyFileCollectionDataset, TinyRecordsDataset, TinyRowsDataset


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
    assert "sample_id_scheme: index" in captured.out
    assert "sample_id_resolver: fallback_index" in captured.out
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
    assert batch["global_sequence"] == 0
    assert batch["sample_ids"] == ["index:0", "index:1"]

    exit_code = main(["--db", str(db_path), "batches", "show", "run-a", "--step", "0"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Batch" in captured.out
    assert "Role: train" in captured.out
    assert "Loader ID: loader-run-a-train" in captured.out
    assert "Batch fingerprint: shared-batch" in captured.out
    assert "Sample IDs: [index:0, index:1]" in captured.out


def test_batches_show_uses_run_global_step(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_batches_global.db"
    store = SQLiteStore(db_path)

    _seed_run(
        store,
        run_id="run-global",
        status="success",
        role="train",
        sample_ids=[0, 1],
        fingerprint="batch-train",
    )
    _append_loader_batch(
        store,
        run_id="run-global",
        role="val",
        sample_ids=[2, 3],
        fingerprint="batch-val",
    )
    store.close()

    exit_code = main(
        ["--db", str(db_path), "--json", "batches", "show", "run-global", "--step", "1"]
    )
    captured = capsys.readouterr()
    batch = json.loads(captured.out)

    assert exit_code == 0
    assert batch["run_id"] == "run-global"
    assert batch["global_sequence"] == 1
    assert batch["loader_id"] == "loader-run-global-val"
    assert batch["role"] == "val"


def test_samples_find_supports_direct_and_filepath_lookup(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_samples.db"
    store = SQLiteStore(db_path)
    dataset_root = _build_tiny_file_dataset(tmp_path / "dataset")
    dataset = TinyFileCollectionDataset(dataset_root)

    from torch.utils.data import DataLoader
    from pypyrus.core.attach import attach
    from pypyrus.core.run import Run

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)
    store.close()

    exit_code = main(
        ["--db", str(db_path), "--json", "samples", "find", run.run_id, "--sample-id", "filepath:class_a/item_0.txt"]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 0
    assert result["found"] is True
    assert result["sample_id"] == "filepath:class_a/item_0.txt"
    assert result["sample_id_scheme"] == "filepath"
    assert result["query_scope"] == "run"
    assert result["occurrence_count"] == 1

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--file",
            "class_a/item_0.txt",
            "--dataset-path",
            str(dataset_root),
        ]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 0
    assert result["found"] is True
    assert result["query_scope"] == "dataset"
    assert result["query_resolution"]["sample_id"] == "filepath:class_a/item_0.txt"


def test_runs_show_and_samples_find_support_structured_record_sample_ids(
    tmp_path, capsys
) -> None:
    db_path = tmp_path / "cli_records.db"
    store = SQLiteStore(db_path)

    from torch.utils.data import DataLoader
    from pypyrus.core.attach import attach
    from pypyrus.core.run import Run

    records_loader = DataLoader(
        TinyRecordsDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    rows_loader = DataLoader(
        TinyRowsDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    with Run(store=store) as run:
        list(attach(records_loader, run, role="train"))
        list(attach(rows_loader, run, role="val"))
    store.close()

    exit_code = main(["--db", str(db_path), "runs", "show", run.run_id])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "sample_id_scheme: record_id" in captured.out
    assert "sample_id_scheme: row" in captured.out
    assert "sample_id_resolver: structured_record" in captured.out

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--sample-id",
            "record_id:cust_001",
        ]
    )
    captured = capsys.readouterr()
    record_result = json.loads(captured.out)

    assert exit_code == 0
    assert record_result["found"] is True
    assert record_result["sample_id_scheme"] == "record_id"
    assert record_result["matching_roles"] == ["train"]

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--sample-id",
            "row:0",
        ]
    )
    captured = capsys.readouterr()
    row_result = json.loads(captured.out)

    assert exit_code == 0
    assert row_result["found"] is True
    assert row_result["sample_id_scheme"] == "row"
    assert row_result["matching_roles"] == ["val"]


def test_samples_find_filepath_lookup_requires_fingerprint_match(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_samples_mismatch.db"
    store = SQLiteStore(db_path)
    dataset_root = _build_tiny_file_dataset(tmp_path / "dataset")
    wrong_root = _build_tiny_file_dataset(tmp_path / "dataset_other", file_name="different.txt")
    dataset = TinyFileCollectionDataset(dataset_root)

    from torch.utils.data import DataLoader
    from pypyrus.core.attach import attach
    from pypyrus.core.run import Run

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)
    store.close()

    exit_code = main(
        [
            "--db",
            str(db_path),
            "samples",
            "find",
            run.run_id,
            "--file",
            f"class_a/{wrong_root.joinpath('class_a', 'different.txt').name}",
            "--dataset-path",
            str(wrong_root),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "matches the provided dataset path fingerprint" in captured.err


def test_samples_find_filepath_lookup_supports_symlinked_split_paths(tmp_path, capsys) -> None:
    db_path = tmp_path / "cli_samples_symlink.db"
    store = SQLiteStore(db_path)

    raw_root = _build_tiny_file_dataset(tmp_path / "raw_dataset")
    split_root = tmp_path / "split_dataset"
    (split_root / "class_a").mkdir(parents=True, exist_ok=True)
    (split_root / "class_b").mkdir(parents=True, exist_ok=True)
    (split_root / "class_a" / "item_0.txt").symlink_to(raw_root / "class_a" / "item_0.txt")
    (split_root / "class_b" / "item_1.txt").symlink_to(raw_root / "class_b" / "item_1.txt")

    dataset = TinyFileCollectionDataset(split_root)

    from torch.utils.data import DataLoader
    from pypyrus.core.attach import attach
    from pypyrus.core.run import Run

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with Run(store=store) as run:
        attached = attach(loader, run, role="train")
        list(attached)
    store.close()

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--file",
            "class_a/item_0.txt",
            "--dataset-path",
            str(split_root),
        ]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 0
    assert result["found"] is True
    assert result["sample_id"] == "filepath:class_a/item_0.txt"
    assert result["query_scope"] == "dataset"


def test_samples_find_distinguishes_duplicate_relative_paths_by_dataset_scope(
    tmp_path, capsys
) -> None:
    db_path = tmp_path / "cli_samples_scoped.db"
    store = SQLiteStore(db_path)

    train_root = _build_tiny_file_dataset(tmp_path / "train_dataset", file_contents="train-a")
    test_root = _build_tiny_file_dataset(tmp_path / "test_dataset", file_contents="test-a")

    from torch.utils.data import DataLoader
    from pypyrus.core.attach import attach
    from pypyrus.core.run import Run

    train_loader = DataLoader(
        TinyFileCollectionDataset(train_root),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TinyFileCollectionDataset(test_root),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    with Run(store=store) as run:
        list(attach(train_loader, run, role="train"))
        list(attach(test_loader, run, role="test"))
    store.close()

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--sample-id",
            "filepath:class_a/item_0.txt",
        ]
    )
    captured = capsys.readouterr()
    broad_result = json.loads(captured.out)

    assert exit_code == 0
    assert broad_result["found"] is True
    assert broad_result["query_scope"] == "run"
    assert broad_result["occurrence_count"] == 2
    assert broad_result["matching_roles"] == ["test", "train"]
    assert len(broad_result["matching_dataset_ids"]) == 2

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--file",
            "class_a/item_0.txt",
            "--dataset-path",
            str(train_root),
        ]
    )
    captured = capsys.readouterr()
    train_result = json.loads(captured.out)

    assert exit_code == 0
    assert train_result["found"] is True
    assert train_result["query_scope"] == "dataset"
    assert train_result["matching_roles"] == ["train"]
    assert len(train_result["matching_dataset_ids"]) == 1
    assert train_result["matching_dataset_ids"] == train_result["scoped_dataset_ids"]

    train_dataset_id = train_result["matching_dataset_ids"][0]
    assert train_result["query_resolution"]["matching_dataset_ids"] == [train_dataset_id]

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--json",
            "samples",
            "find",
            run.run_id,
            "--sample-id",
            "filepath:class_a/item_0.txt",
            "--dataset-id",
            train_dataset_id,
        ]
    )
    captured = capsys.readouterr()
    scoped_result = json.loads(captured.out)

    assert exit_code == 0
    assert scoped_result["found"] is True
    assert scoped_result["query_scope"] == "dataset"
    assert scoped_result["matching_roles"] == ["train"]
    assert scoped_result["matching_dataset_ids"] == [train_dataset_id]
    assert scoped_result["scoped_dataset_ids"] == [train_dataset_id]


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
    dataset_id = f"in_memory_deterministic_v1:fingerprint-{run_id}"
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
            sample_id_scheme="index",
            sample_id_resolver="fallback_index",
        )
    )
    dataset_registration_event_id = store.get_events(run_id, event_type="dataset_registered")[0]["event_id"]
    store.append_event(
        TransformDeclaredEvent(
            run_id=run_id,
            dataset_registration_event_id=dataset_registration_event_id,
            transform_list=transform_list,
            params_hash=hash_json(transform_list),
            introspection_level="full",
        )
    )
    store.append_event(
        LoaderRegisteredEvent(
            run_id=run_id,
            loader_id=loader_id,
            dataset_registration_event_id=dataset_registration_event_id,
            role=role,
        )
    )
    store.append_event(
        BatchDeliveredEvent(
            run_id=run_id,
            loader_id=loader_id,
            global_step=0,
            global_sequence=0,
            batch_size=len(sample_ids),
            batch_fingerprint=fingerprint,
            sample_ids_blob=gzip.compress(
                json.dumps([f"index:{sample_id}" for sample_id in sample_ids]).encode("utf-8")
            ),
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


def _append_loader_batch(
    store: SQLiteStore,
    *,
    run_id: str,
    role: str,
    sample_ids: list[int],
    fingerprint: str,
) -> None:
    dataset_id = f"in_memory_deterministic_v1:fingerprint-{run_id}-{role}"
    loader_id = f"loader-{run_id}-{role}"

    store.append_event(
        DatasetRegisteredEvent(
            run_id=run_id,
            dataset_id=dataset_id,
            name=f"TinyDataset-{role}",
            role=role,
            fingerprint=f"fingerprint-{run_id}-{role}",
            fingerprint_method="in_memory_deterministic_v1",
            sample_id_scheme="index",
            sample_id_resolver="fallback_index",
        )
    )
    dataset_registration_event_id = store.get_events(run_id, event_type="dataset_registered")[-1]["event_id"]
    store.append_event(
        LoaderRegisteredEvent(
            run_id=run_id,
            loader_id=loader_id,
            dataset_registration_event_id=dataset_registration_event_id,
            role=role,
        )
    )
    store.append_event(
        BatchDeliveredEvent(
            run_id=run_id,
            loader_id=loader_id,
            global_step=0,
            global_sequence=1,
            batch_size=len(sample_ids),
            batch_fingerprint=fingerprint,
            sample_ids_blob=gzip.compress(
                json.dumps([f"index:{sample_id}" for sample_id in sample_ids]).encode("utf-8")
            ),
        )
    )
    store.flush()


def _build_tiny_file_dataset(
    root: Path,
    *,
    file_name: str = "item_0.txt",
    file_contents: str = "a",
) -> Path:
    (root / "class_a").mkdir(parents=True, exist_ok=True)
    (root / "class_b").mkdir(parents=True, exist_ok=True)
    (root / "class_a" / file_name).write_text(file_contents)
    (root / "class_b" / "item_1.txt").write_text("b")
    return root
