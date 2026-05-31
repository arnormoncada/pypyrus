from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run instrumented workload(s) and summarize PyPyrus SQLite storage "
            "footprint for each resulting database."
        )
    )
    parser.add_argument(
        "--workload",
        choices=["plant", "covtype", "all"],
        default="all",
        help="Which storage-footprint workload to run. Default: all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/storage_footprint"),
        help="Directory for generated databases and summary JSON. Default: experiments/results/storage_footprint",
    )
    parser.add_argument(
        "--plant-data-root",
        type=Path,
        default=Path("experiments/plant_seedlings/data/split"),
        help="Plant-seedlings split root. Default: experiments/plant_seedlings/data/split",
    )
    parser.add_argument(
        "--covtype-data-path",
        type=Path,
        default=Path("experiments/forest_covertype/data/covtype_with_sample_id.csv"),
        help="Forest Covertype CSV path. Default: experiments/forest_covertype/data/covtype_with_sample_id.csv",
    )
    parser.add_argument(
        "--plant-epochs",
        type=int,
        default=3,
        help="Epoch count for the plant-seedlings run. Default: 3",
    )
    parser.add_argument(
        "--covtype-epochs",
        type=int,
        default=3,
        help="Epoch count for the forest-covertype run. Default: 3",
    )
    parser.add_argument(
        "--plant-batch-size",
        type=int,
        default=32,
        help="Batch size for the plant-seedlings run. Default: 32",
    )
    parser.add_argument(
        "--covtype-batch-size",
        type=int,
        default=256,
        help="Batch size for the forest-covertype run. Default: 256",
    )
    parser.add_argument(
        "--plant-num-workers",
        type=int,
        default=2,
        help="DataLoader worker count for plant seedlings. Default: 2",
    )
    parser.add_argument(
        "--covtype-num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for forest covertype. Default: 0",
    )
    parser.add_argument(
        "--covtype-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the covtype MLP. Default: 128",
    )
    parser.add_argument(
        "--covtype-test-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for the covtype split. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shared base seed for the storage-footprint runs. Default: 42",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove the target output directory before running.",
    )
    return parser


def parse_timing_line(timing_path: Path) -> dict[str, Any]:
    if not timing_path.exists():
        return {}

    lines = [line.strip() for line in timing_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}

    fields: dict[str, Any] = {}
    for token in lines[-1].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "elapsed_seconds":
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
        else:
            fields[key] = value
    return fields


def sqlite_count(conn: sqlite3.Connection, table: str, run_id: str) -> int:
    return int(
        conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE run_id = ?",
            (run_id,),
        ).fetchone()["count"]
    )


def summarize_db(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        run_row = conn.execute(
            """
            SELECT run_id, run_name, status, event_count, start_time, end_time
            FROM runs
            ORDER BY start_time DESC
            LIMIT 1
            """
        ).fetchone()
        if run_row is None:
            raise RuntimeError(f"No runs found in SQLite database: {db_path}")

        run_id = str(run_row["run_id"])
        event_count = int(run_row["event_count"] or 0)
        batch_count = sqlite_count(conn, "batch_delivered", run_id)
        dataset_registration_count = sqlite_count(conn, "dataset_registrations", run_id)
        loader_count = sqlite_count(conn, "loaders", run_id)
        transform_count = sqlite_count(conn, "transform_declared", run_id)
        environment_snapshot_count = sqlite_count(conn, "environment_snapshot", run_id)
    finally:
        conn.close()

    db_bytes = db_path.stat().st_size
    summary: dict[str, Any] = {
        "db_path": str(db_path),
        "db_bytes": db_bytes,
        "db_mib": db_bytes / (1024 * 1024),
        "run_id": run_id,
        "run_name": run_row["run_name"],
        "status": run_row["status"],
        "event_count": event_count,
        "batch_count": batch_count,
        "dataset_registration_count": dataset_registration_count,
        "loader_count": loader_count,
        "transform_count": transform_count,
        "environment_snapshot_count": environment_snapshot_count,
        "approx_bytes_per_batch": (db_bytes / batch_count) if batch_count else None,
        "approx_bytes_per_event": (db_bytes / event_count) if event_count else None,
    }
    return summary


def run_command(cmd: list[str]) -> None:
    print("Running command:", flush=True)
    print(" " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_plant_case(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    case_root = output_root / "plant"
    case_root.mkdir(parents=True, exist_ok=True)
    db_path = case_root / "plant_storage.db"
    timing_path = case_root / "plant_timing.txt"
    if db_path.exists():
        db_path.unlink()
    if timing_path.exists():
        timing_path.unlink()

    cmd = [
        sys.executable,
        "experiments/plant_seedlings/train_mobilenetv3_small.py",
        "--data-root",
        str(args.plant_data_root.expanduser().resolve()),
        "--epochs",
        str(args.plant_epochs),
        "--batch-size",
        str(args.plant_batch_size),
        "--num-workers",
        str(args.plant_num_workers),
        "--seed",
        str(args.seed),
        "--db-path",
        str(db_path),
        "--run-name",
        "storage-plant",
        "--timing-file",
        str(timing_path),
    ]
    run_command(cmd)

    summary = summarize_db(db_path)
    summary["workload"] = "plant"
    summary["timing"] = parse_timing_line(timing_path)
    return summary


def run_covtype_case(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    case_root = output_root / "covtype"
    case_root.mkdir(parents=True, exist_ok=True)
    db_path = case_root / "covtype_storage.db"
    timing_path = case_root / "covtype_timing.txt"
    if db_path.exists():
        db_path.unlink()
    if timing_path.exists():
        timing_path.unlink()

    cmd = [
        sys.executable,
        "experiments/forest_covertype/train_covtype_mlp.py",
        "--data-path",
        str(args.covtype_data_path.expanduser().resolve()),
        "--epochs",
        str(args.covtype_epochs),
        "--batch-size",
        str(args.covtype_batch_size),
        "--hidden-dim",
        str(args.covtype_hidden_dim),
        "--test-ratio",
        str(args.covtype_test_ratio),
        "--num-workers",
        str(args.covtype_num_workers),
        "--seed",
        str(args.seed),
        "--db-path",
        str(db_path),
        "--run-name",
        "storage-covtype",
        "--timing-file",
        str(timing_path),
    ]
    run_command(cmd)

    summary = summarize_db(db_path)
    summary["workload"] = "covtype"
    summary["timing"] = parse_timing_line(timing_path)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()

    if args.reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = ["plant", "covtype"] if args.workload == "all" else [args.workload]
    summaries: list[dict[str, Any]] = []

    for workload in selected:
        if workload == "plant":
            summary = run_plant_case(args, output_root)
        elif workload == "covtype":
            summary = run_covtype_case(args, output_root)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"Unsupported workload: {workload}")

        summary_path = output_root / workload / f"{workload}_storage_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(summary)

        print()
        print(
            f"[{workload}] db_mib={summary['db_mib']:.3f} "
            f"events={summary['event_count']} "
            f"batches={summary['batch_count']} "
            f"bytes_per_batch={summary['approx_bytes_per_batch']:.2f}"
        )

    suite_summary = {
        "workload_selection": args.workload,
        "output_root": str(output_root),
        "cases": summaries,
    }
    suite_summary_path = output_root / "storage_summary.json"
    suite_summary_path.write_text(
        json.dumps(suite_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(f"Suite summary written to: {suite_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
