"""
Run the baseline positive-control reproducibility experiment for PyPyrus.

This script executes the same plant-seedlings training configuration twice,
stores both runs in one explicit PyPyrus SQLite database, compares the runs,
and optionally writes a JSON evidence bundle.

Example:

    python experiments/run_baseline_reproducibility_match.py \
      --data-root examples/plant_seedlings/data/split \
      --epochs 3 \
      --seed 42 \
      --reset-db
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from pypyrus import Run, attach
from pypyrus.reporting import (
    build_run_overview,
    compare_runs,
    find_sample_occurrences,
    format_run_comparison,
    get_batch_for_run_step,
    resolve_file_query_for_run,
)
from pypyrus.storage.sqlite_store import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the baseline reproducibility-match experiment by running "
            "the same plant-seedlings training configuration twice."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("experiments/results/baseline_reproducibility_match.db"),
        help=(
            "Path to the PyPyrus SQLite database used for this experiment. "
            "Default: experiments/results/baseline_reproducibility_match.db"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("experiments/results/baseline_reproducibility_match.json"),
        help=(
            "Path for the JSON evidence bundle. "
            "Default: experiments/results/baseline_reproducibility_match.json"
        ),
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
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier head. Default: 1e-3",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count. Default: 2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model init and train-loader shuffle. Default: 42",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help=(
            "Optional relative train-split file path to use for sample lookup "
            "evidence. Defaults to the first train sample discovered."
        ),
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Remove the target database before running the experiment.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    data_root = args.data_root.expanduser().resolve()
    train_root = data_root / "train"
    test_root = data_root / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Expected train/ and test/ under data root: {data_root}"
        )

    db_path = args.db.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset_db and db_path.exists():
        db_path.unlink()

    sample_file = args.sample_file or infer_first_train_sample_relative_path(train_root)

    config = {
        "data_root": str(data_root),
        "train_root": str(train_root),
        "test_root": str(test_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "sample_file": sample_file,
    }

    print("Baseline reproducibility match experiment")
    print("-" * 60)
    print(f"DB path: {db_path}")
    print(f"Report path: {report_path}")
    print(f"Sample evidence target: {sample_file}")
    print("")

    run_id_a = execute_one_run(db_path, config)
    run_id_b = execute_one_run(db_path, config)

    store = SQLiteStore(db_path)
    try:
        comparison = compare_runs(store, run_id_a, run_id_b)
        overview_a = build_run_overview(store, run_id_a)
        overview_b = build_run_overview(store, run_id_b)
        run_a_train_batch_0 = get_batch_for_run_step(
            store, run_id_a, 0, include_sample_ids=True
        )
        run_b_train_batch_0 = get_batch_for_run_step(
            store, run_id_b, 0, include_sample_ids=True
        )
        sample_lookup_a = build_sample_lookup_evidence(
            store,
            run_id=run_id_a,
            train_root=train_root,
            relative_file_path=sample_file,
        )
        sample_lookup_b = build_sample_lookup_evidence(
            store,
            run_id=run_id_b,
            train_root=train_root,
            relative_file_path=sample_file,
        )
    finally:
        store.close()

    evidence = {
        "experiment": "baseline_reproducibility_match",
        "config": config,
        "db_path": str(db_path),
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "comparison": comparison,
        "run_overview_a": overview_a,
        "run_overview_b": overview_b,
        "supporting_checks": {
            "batch_step_0": {
                "run_a": run_a_train_batch_0,
                "run_b": run_b_train_batch_0,
            },
            "sample_lookup": {
                "relative_file_path": sample_file,
                "run_a": sample_lookup_a,
                "run_b": sample_lookup_b,
            },
        },
    }

    report_path.write_text(
        json.dumps(make_json_safe(evidence), indent=2), encoding="utf-8"
    )

    print(f"Run A: {run_id_a}")
    print(f"Run B: {run_id_b}")
    print("")
    print(format_run_comparison(comparison))
    print("")
    print(f"JSON evidence written to: {report_path}")

    return 0


def execute_one_run(db_path: Path, config: dict[str, Any]) -> str:
    store = SQLiteStore(db_path)
    run = Run(store=store)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(config["seed"]))
    train_generator = torch.Generator().manual_seed(int(config["seed"]))

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    train_data = datasets.ImageFolder(config["train_root"], transform=transform)
    test_data = datasets.ImageFolder(config["test_root"], transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_in_features, len(train_data.classes))
    model.to(device)

    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=float(config["lr"]))
    loss_fn = nn.CrossEntropyLoss()

    with run:
        train_loader = attach(train_loader, run, role="train")
        test_loader = attach(test_loader, run, role="test")

        for _ in range(int(config["epochs"])):
            train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            evaluate(model, test_loader, loss_fn, device)

    return run.run_id


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += float(loss.item())
        total_batches += 1
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(labels.size(0))

    average_loss = total_loss / max(total_batches, 1)
    accuracy = correct / max(total_examples, 1)
    return average_loss, accuracy


def infer_first_train_sample_relative_path(train_root: Path) -> str:
    dataset = datasets.ImageFolder(train_root)
    if not dataset.samples:
        raise ValueError(f"No samples found under train root: {train_root}")
    sample_path = Path(dataset.samples[0][0])
    return sample_path.relative_to(train_root).as_posix()


def build_sample_lookup_evidence(
    store: SQLiteStore,
    *,
    run_id: str,
    train_root: Path,
    relative_file_path: str,
) -> dict[str, Any]:
    resolved = resolve_file_query_for_run(
        store,
        run_id,
        dataset_path=train_root,
        file_path=relative_file_path,
    )
    return {
        "resolved_query": resolved,
        "occurrences": find_sample_occurrences(
            store,
            run_id,
            resolved["sample_id"],
            dataset_ids=resolved["matching_dataset_ids"],
        ),
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
            if key != "sample_ids_blob"
        }
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    return value


if __name__ == "__main__":
    raise SystemExit(main())
