from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.plant_seedlings.repro_utils import (
    build_training_config,
    collect_pair_evidence,
    create_a3_modified_dataset_copy,
    execute_one_run,
    make_json_safe,
)
from pypyrus.reporting import format_run_comparison


CASE_METADATA = {
    "a1": {
        "label": "A1 baseline match",
        "db_name": "a1_match.db",
        "report_name": "a1_match.json",
        "run_name_a": "a1-match-a",
        "run_name_b": "a1-match-b",
    },
    "a2": {
        "label": "A2 shuffle divergence",
        "db_name": "a2_shuffle.db",
        "report_name": "a2_shuffle.json",
    },
    "a3": {
        "label": "A3 dataset-content divergence",
        "db_name": "a3_dataset_content.db",
        "report_name": "a3_dataset_content.json",
        "run_name_a": "a3-data-base",
        "run_name_b": "a3-data-mod",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the A-series PyPyrus reproducibility and divergence suite "
            "on the plant-seedlings workload."
        )
    )
    parser.add_argument(
        "--case",
        choices=["a1", "a2", "a3", "all"],
        default="all",
        help="Which reproducibility case to run. Default: all",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/repro_suite"),
        help="Directory for per-case databases and evidence. Default: experiments/results/repro_suite",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs per run. Default: 3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for train/test loaders. Default: 32",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count. Default: 2",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier head. Default: 1e-3",
    )
    parser.add_argument(
        "--seed-a",
        type=int,
        default=42,
        help="Seed for run A. Default: 42",
    )
    parser.add_argument(
        "--seed-b",
        type=int,
        default=99,
        help="Seed for run B in divergence cases. Default: 99",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove the target output directory before running the suite.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = ["a1", "a2", "a3"] if args.case == "all" else [args.case]

    suite_results: list[dict[str, Any]] = []
    for case in cases:
        result = run_case(
            case=case,
            data_root=data_root,
            output_root=output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            seed_a=args.seed_a,
            seed_b=args.seed_b,
        )
        suite_results.append(result)

    if args.case == "all":
        summary_path = output_root / "suite_summary.json"
        summary = {
            "experiment": "repro_suite",
            "cases": suite_results,
        }
        summary_path.write_text(
            json.dumps(make_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        print("")
        print(f"Suite summary written to: {summary_path}")

    return 0


def run_case(
    *,
    case: str,
    data_root: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    seed_a: int,
    seed_b: int,
) -> dict[str, Any]:
    metadata = CASE_METADATA[case]
    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / metadata["db_name"]
    report_path = case_dir / metadata["report_name"]

    if db_path.exists():
        db_path.unlink()
    if report_path.exists():
        report_path.unlink()

    print(metadata["label"])
    print("-" * 60)
    print(f"DB path: {db_path}")
    print(f"Report path: {report_path}")

    if case == "a1":
        result = _run_a1(
            data_root=data_root,
            db_path=db_path,
            report_path=report_path,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            seed=seed_a,
        )
    elif case == "a2":
        result = _run_a2(
            data_root=data_root,
            db_path=db_path,
            report_path=report_path,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            seed_a=seed_a,
            seed_b=seed_b,
        )
    elif case == "a3":
        result = _run_a3(
            data_root=data_root,
            case_dir=case_dir,
            db_path=db_path,
            report_path=report_path,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            seed=seed_a,
        )
    else:
        raise ValueError(f"Unsupported case: {case}")

    print("")
    return result


def _run_a1(
    *,
    data_root: Path,
    db_path: Path,
    report_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    metadata = CASE_METADATA["a1"]
    config_a = build_training_config(
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed,
    )
    config_b = dict(config_a)

    run_id_a = execute_one_run(
        db_path,
        config_a,
        run_name=metadata["run_name_a"],
    )
    run_id_b = execute_one_run(
        db_path,
        config_b,
        run_name=metadata["run_name_b"],
    )

    evidence = collect_pair_evidence(
        db_path=db_path,
        report_path=report_path,
        case="a1",
        case_label=metadata["label"],
        config_a=config_a,
        config_b=config_b,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        run_name_a=metadata["run_name_a"],
        run_name_b=metadata["run_name_b"],
        sample_file=str(config_a["sample_file"]),
    )
    report_path.write_text(
        json.dumps(make_json_safe(evidence), indent=2),
        encoding="utf-8",
    )
    print(_render_case_summary(report_path, evidence))
    return {
        "case": "a1",
        "report_path": str(report_path),
        "db_path": str(db_path),
        "runs": evidence["runs"],
        "comparison": evidence["comparison"],
    }


def _run_a2(
    *,
    data_root: Path,
    db_path: Path,
    report_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    seed_a: int,
    seed_b: int,
) -> dict[str, Any]:
    run_name_a = f"a2-shuffle-s{seed_a}"
    run_name_b = f"a2-shuffle-s{seed_b}"
    config_a = build_training_config(
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed_a,
    )
    config_b = build_training_config(
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed_b,
        sample_file=str(config_a["sample_file"]),
    )

    run_id_a = execute_one_run(db_path, config_a, run_name=run_name_a)
    run_id_b = execute_one_run(db_path, config_b, run_name=run_name_b)

    evidence = collect_pair_evidence(
        db_path=db_path,
        report_path=report_path,
        case="a2",
        case_label=CASE_METADATA["a2"]["label"],
        config_a=config_a,
        config_b=config_b,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        run_name_a=run_name_a,
        run_name_b=run_name_b,
        sample_file=str(config_a["sample_file"]),
    )
    report_path.write_text(
        json.dumps(make_json_safe(evidence), indent=2),
        encoding="utf-8",
    )
    print(_render_case_summary(report_path, evidence))
    return {
        "case": "a2",
        "report_path": str(report_path),
        "db_path": str(db_path),
        "runs": evidence["runs"],
        "comparison": evidence["comparison"],
    }


def _run_a3(
    *,
    data_root: Path,
    case_dir: Path,
    db_path: Path,
    report_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    metadata = CASE_METADATA["a3"]
    derived_data_root = case_dir / "derived_dataset"
    mutation = create_a3_modified_dataset_copy(
        source_data_root=data_root,
        derived_data_root=derived_data_root,
    )
    baseline_config = build_training_config(
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed,
    )
    modified_config = build_training_config(
        data_root=derived_data_root,
        train_root=derived_data_root / "train",
        test_root=data_root / "test",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed,
        sample_file=str(baseline_config["sample_file"]),
    )

    run_id_a = execute_one_run(
        db_path,
        baseline_config,
        run_name=metadata["run_name_a"],
    )
    run_id_b = execute_one_run(
        db_path,
        modified_config,
        run_name=metadata["run_name_b"],
    )

    evidence = collect_pair_evidence(
        db_path=db_path,
        report_path=report_path,
        case="a3",
        case_label=metadata["label"],
        config_a=baseline_config,
        config_b=modified_config,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        run_name_a=metadata["run_name_a"],
        run_name_b=metadata["run_name_b"],
        sample_file=str(baseline_config["sample_file"]),
        mutation=mutation,
    )
    report_path.write_text(
        json.dumps(make_json_safe(evidence), indent=2),
        encoding="utf-8",
    )
    print(_render_case_summary(report_path, evidence))
    return {
        "case": "a3",
        "report_path": str(report_path),
        "db_path": str(db_path),
        "runs": evidence["runs"],
        "comparison": evidence["comparison"],
        "mutation": mutation,
    }


def _render_case_summary(report_path: Path, evidence: dict[str, Any]) -> str:
    comparison = evidence["comparison"]
    return "\n".join(
        [
            f"Run A: {evidence['runs']['run_a']['run_id']} ({evidence['runs']['run_a']['run_name']})",
            f"Run B: {evidence['runs']['run_b']['run_id']} ({evidence['runs']['run_b']['run_name']})",
            "",
            format_run_comparison(comparison),
            "",
            f"JSON evidence written to: {report_path}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
