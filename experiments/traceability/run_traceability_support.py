from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypyrus.cli.render import render_run_overview, render_sample_find
from pypyrus.reporting import (
    build_run_overview,
    find_sample_occurrences,
    get_batch_for_run_step,
    get_batches_for_run,
)
from pypyrus.storage.sqlite_store import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more instrumented workloads and collect compact "
            "traceability-support evidence from the resulting PyPyrus databases."
        )
    )
    parser.add_argument(
        "--workload",
        choices=["plant", "covtype", "all"],
        default="all",
        help="Which traceability workload to run. Default: all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/traceability_support"),
        help="Directory for generated databases and evidence bundles. Default: experiments/results/traceability_support",
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
        default=1,
        help="Epoch count for the plant-seedlings run. Default: 1",
    )
    parser.add_argument(
        "--covtype-epochs",
        type=int,
        default=1,
        help="Epoch count for the forest-covertype run. Default: 1",
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
        help="Shared seed for the traceability runs. Default: 42",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove the target output directory before running.",
    )
    return parser


def run_command(cmd: list[str]) -> None:
    print("Running command:", flush=True)
    print(" " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_compact_batch(batch: dict[str, Any], *, preview_count: int = 12) -> str:
    sample_ids = [str(item) for item in (batch.get("sample_ids") or [])]
    preview = sample_ids[:preview_count]
    more_count = max(len(sample_ids) - len(preview), 0)
    preview_text = "[" + ", ".join(preview) + "]"
    if more_count > 0:
        preview_text += f" ... (+{more_count} more)"

    lines = [
        "Batch",
        "-" * 60,
        f"Run ID: {batch.get('run_id')}",
        f"Role: {batch.get('role')}",
        f"Loader ID: {batch.get('loader_id')}",
        f"Dataset ID: {batch.get('dataset_id')}",
        f"Timestamp: {batch.get('timestamp') or '<unknown>'}",
        f"Global step: {batch.get('global_step')}",
        f"Global sequence: {batch.get('global_sequence')}",
        f"Batch size: {batch.get('batch_size')}",
        f"Batch fingerprint: {batch.get('batch_fingerprint')}",
        f"Sample IDs preview: {preview_text}",
    ]
    return "\n".join(lines)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


def select_traceability_evidence(db_path: Path) -> dict[str, Any]:
    store = SQLiteStore(db_path)

    try:
        run_rows = store.list_runs()
        if not run_rows:
            raise RuntimeError(f"No run IDs found in SQLite database: {db_path}")
        run_id = run_rows[0]

        overview = build_run_overview(store, run_id)
        if overview is None:
            raise RuntimeError(f"Could not build run overview for run_id={run_id}")

        train_batches = get_batches_for_run(
            store,
            run_id,
            include_sample_ids=True,
            role="train",
        )
        if not train_batches:
            raise RuntimeError(f"No train batches found for run_id={run_id}")

        selected_train_batch = train_batches[0]
        selected_step = int(selected_train_batch["global_sequence"])
        selected_batch = get_batch_for_run_step(
            store,
            run_id,
            selected_step,
            include_sample_ids=True,
        )
        if selected_batch is None:
            raise RuntimeError(
                f"Could not reload batch for run_id={run_id} at step={selected_step}"
            )

        sample_ids = selected_batch.get("sample_ids") or []
        if not sample_ids:
            raise RuntimeError(
                f"Selected batch at step={selected_step} has no recorded sample IDs."
            )
        selected_sample_id = str(sample_ids[0])
        sample_lookup = find_sample_occurrences(store, run_id, selected_sample_id)

        return {
            "run_id": run_id,
            "overview": overview,
            "selected_batch_step": selected_step,
            "selected_batch_role": selected_batch.get("role"),
            "selected_sample_id": selected_sample_id,
            "selected_batch": selected_batch,
            "sample_lookup": sample_lookup,
        }
    finally:
        store.close()


def write_traceability_artifacts(case_root: Path, evidence: dict[str, Any]) -> None:
    summary_path = case_root / "traceability_summary.json"
    write_text(
        summary_path,
        json.dumps(make_json_safe(evidence), indent=2, sort_keys=True),
    )

    write_text(
        case_root / "run_overview.txt",
        render_run_overview(evidence["overview"]),
    )
    write_text(
        case_root / "selected_batch.txt",
        render_compact_batch(evidence["selected_batch"]),
    )
    write_text(
        case_root / "sample_lookup.txt",
        render_sample_find(evidence["sample_lookup"]),
    )


def run_plant_case(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    case_root = output_root / "plant"
    case_root.mkdir(parents=True, exist_ok=True)
    db_path = case_root / "plant_traceability.db"
    if db_path.exists():
        db_path.unlink()

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
        "traceability-plant",
    ]
    run_command(cmd)

    evidence = select_traceability_evidence(db_path)
    evidence["workload"] = "plant"
    evidence["db_path"] = str(db_path)
    write_traceability_artifacts(case_root, evidence)
    return evidence


def run_covtype_case(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    case_root = output_root / "covtype"
    case_root.mkdir(parents=True, exist_ok=True)
    db_path = case_root / "covtype_traceability.db"
    if db_path.exists():
        db_path.unlink()

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
        "traceability-covtype",
    ]
    run_command(cmd)

    evidence = select_traceability_evidence(db_path)
    evidence["workload"] = "covtype"
    evidence["db_path"] = str(db_path)
    write_traceability_artifacts(case_root, evidence)
    return evidence


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()

    if args.reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = ["plant", "covtype"] if args.workload == "all" else [args.workload]
    results: list[dict[str, Any]] = []

    for workload in selected:
        if workload == "plant":
            evidence = run_plant_case(args, output_root)
        elif workload == "covtype":
            evidence = run_covtype_case(args, output_root)
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"Unsupported workload: {workload}")

        print()
        print(
            f"[{workload}] run_id={evidence['run_id']} "
            f"selected_step={evidence['selected_batch_step']} "
            f"sample_id={evidence['selected_sample_id']} "
            f"occurrences={evidence['sample_lookup']['occurrence_count']}"
        )
        results.append(
            {
                "workload": evidence["workload"],
                "db_path": evidence["db_path"],
                "run_id": evidence["run_id"],
                "run_name": evidence["overview"]["run"].get("run_name"),
                "selected_batch_step": evidence["selected_batch_step"],
                "selected_sample_id": evidence["selected_sample_id"],
                "occurrence_count": evidence["sample_lookup"]["occurrence_count"],
                "matching_steps": evidence["sample_lookup"]["matching_steps"],
            }
        )

    suite_summary = {
        "workload_selection": args.workload,
        "output_root": str(output_root),
        "cases": results,
    }
    suite_summary_path = output_root / "traceability_suite_summary.json"
    write_text(suite_summary_path, json.dumps(suite_summary, indent=2, sort_keys=True))

    print()
    print(f"Suite summary written to: {suite_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
