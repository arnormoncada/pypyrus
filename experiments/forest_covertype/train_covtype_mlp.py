"""
Train a small MLP on a covtype CSV with an explicit custom sample-ID resolver.

Training goal:

- predict the forest cover type class from tabular feature columns
- keep exact row-level provenance through a custom sample ID resolver
- record that both train/test splits came from one source CSV file

Expected CSV shape:

- one stable identifier column, default: ``sample_id``
- one target column, default: ``Cover_Type``
- remaining columns are treated as numeric features

Example:

    python experiments/forest_covertype/train_covtype_mlp.py \
      --data-path experiments/forest_covertype/data/covtype_with_sample_id.csv \
      --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pypyrus import Run, attach
from pypyrus.core.sample_id import SampleIdResolution
from pypyrus.storage.sqlite_store import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small MLP to predict forest cover type from a covtype CSV file."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to covtype_with_sample_id.csv",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="sample_id",
        help="Stable sample identifier column. Default: sample_id",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Cover_Type",
        help="Target column name. Default: Cover_Type",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for the test split. Default: 0.2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size. Default: 256",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the MLP. Default: 128",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for split, shuffle, and init. Default: 7",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Default: 0",
    )
    parser.add_argument(
        "--no-instrumentation",
        action="store_true",
        help="Skip pypyrus DataLoader instrumentation attach().",
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        default=None,
        help="Optional file where runtime timing is appended.",
    )
    parser.add_argument(
        "--buffered-queue",
        action="store_true",
        help="Enable BufferedStore strict mode for the Run.",
    )
    parser.add_argument(
        "--buffered-queue-size",
        type=int,
        default=2048,
        help="Queue size for BufferedStore strict mode. Default: 2048",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Explicit SQLite database path for instrumented runs. Default: use Run() default store path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable run name stored in the run metadata.",
    )
    return parser


class CovtypeDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        features = torch.tensor(row["features"], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


class CovtypeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_rows(
    data_path: Path,
    *,
    id_column: str,
    target_column: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {id_column, target_column}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(missing)}"
            )

        feature_columns = [
            name for name in fieldnames
            if name not in {id_column, target_column}
        ]
        if not feature_columns:
            raise ValueError("No feature columns found after excluding id/target columns.")

        rows: list[dict[str, Any]] = []
        label_values: set[str] = set()
        for csv_row in reader:
            sample_id = str(csv_row[id_column]).strip()
            target_value = str(csv_row[target_column]).strip()
            if not sample_id:
                raise ValueError("Encountered an empty sample_id value.")
            if not target_value:
                raise ValueError("Encountered an empty target value.")

            try:
                features = [float(csv_row[column]) for column in feature_columns]
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse numeric features for sample_id={sample_id}"
                ) from exc

            rows.append(
                {
                    "sample_id": sample_id,
                    "features": features,
                    "target_value": target_value,
                }
            )
            label_values.add(target_value)

    labels = sorted(label_values)
    label_to_id = {label: index for index, label in enumerate(labels)}
    for row in rows:
        row["label"] = label_to_id[row["target_value"]]

    return rows, labels


def split_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio))
    test_rows = shuffled[:test_count]
    train_rows = shuffled[test_count:]
    return train_rows, test_rows


def covtype_sample_id_resolver(
    dataset: Any,
    index: int,
    sample: tuple[torch.Tensor, torch.Tensor],
) -> SampleIdResolution:
    row = dataset.rows[index]
    return SampleIdResolution(
        sample_id=f"record_id:{row['sample_id']}",
        sample_id_scheme="record_id",
        sample_id_resolver="user_override",
    )


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

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
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

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = loss_fn(logits, labels)

        total_loss += float(loss.item())
        total_batches += 1
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(labels.size(0))

    avg_loss = total_loss / max(total_batches, 1)
    accuracy = correct / max(total_examples, 1)
    return avg_loss, accuracy


def main() -> int:
    args = build_parser().parse_args()
    timer_start = time.perf_counter()

    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"CSV data path not found: {data_path}")
    if args.buffered_queue_size <= 0:
        raise ValueError("--buffered-queue-size must be > 0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, label_names = load_rows(
        data_path,
        id_column=args.id_column,
        target_column=args.target_column,
    )
    train_rows, test_rows = split_rows(
        rows,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    train_dataset = CovtypeDataset(train_rows)
    test_dataset = CovtypeDataset(test_rows)

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    input_dim = len(train_rows[0]["features"])
    num_classes = len(label_names)
    model = CovtypeMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"seed={args.seed}")
    print(f"source_csv={data_path}")
    print(f"num_features={input_dim}")
    print(f"num_classes={num_classes}")
    print(f"train_rows={len(train_rows)}")
    print(f"test_rows={len(test_rows)}")

    use_instrumentation = not args.no_instrumentation
    if use_instrumentation:
        store = None
        if args.db_path is not None:
            db_path = args.db_path.expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            store = SQLiteStore(db_path)
        store_mode = "buffered_strict" if args.buffered_queue else "sync"
        with Run(
            store=store,
            store_mode=store_mode,
            buffered_queue_size=args.buffered_queue_size,
            run_name=args.run_name,
        ) as run:
            train_loader = attach(
                train_loader,
                run,
                role="train",
                sample_id_resolver=covtype_sample_id_resolver,
                dataset_name="CovtypeCSVDataset",
                dataset_uri=str(data_path),
                dataset_version_hint="split=train",
            )
            test_loader = attach(
                test_loader,
                run,
                role="test",
                sample_id_resolver=covtype_sample_id_resolver,
                dataset_name="CovtypeCSVDataset",
                dataset_uri=str(data_path),
                dataset_version_hint="split=test",
            )

            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
                test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"test_loss={test_loss:.4f} "
                    f"test_acc={test_accuracy:.4f}"
                )
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_accuracy:.4f}"
            )

    if args.timing_file is not None:
        elapsed_seconds = time.perf_counter() - timer_start
        timing_line = (
            f"instrumentation={use_instrumentation} "
            f"epochs={args.epochs} "
            f"batch_size={args.batch_size} "
            f"num_workers={args.num_workers} "
            f"elapsed_seconds={elapsed_seconds:.6f}\n"
        )
        timing_path = args.timing_file.expanduser().resolve()
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("a", encoding="utf-8") as timing_file:
            timing_file.write(timing_line)
        print(f"timing: {timing_line.strip()}")
        print(f"timing_written_to={timing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
