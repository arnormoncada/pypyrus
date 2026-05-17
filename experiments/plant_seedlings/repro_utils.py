from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
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
)
from pypyrus.storage.sqlite_store import SQLiteStore


def build_training_config(
    *,
    data_root: Path,
    train_root: Path | None = None,
    test_root: Path | None = None,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
    sample_file: str | None = None,
    train_dataset_uri: str | None = None,
    test_dataset_uri: str | None = None,
) -> dict[str, Any]:
    resolved_train_root = train_root or (data_root / "train")
    resolved_test_root = test_root or (data_root / "test")
    if not resolved_train_root.exists() or not resolved_test_root.exists():
        raise FileNotFoundError(
            f"Expected train/ and test/ under data root: {data_root}"
        )

    resolved_sample_file = sample_file or infer_first_train_sample_relative_path(
        resolved_train_root
    )
    return {
        "data_root": str(data_root),
        "train_root": str(resolved_train_root),
        "test_root": str(resolved_test_root),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "num_workers": int(num_workers),
        "seed": int(seed),
        "sample_file": resolved_sample_file,
        "train_dataset_uri": train_dataset_uri,
        "test_dataset_uri": test_dataset_uri,
    }


def execute_one_run(
    db_path: Path,
    config: dict[str, Any],
    *,
    run_name: str,
) -> str:
    store = SQLiteStore(db_path)
    run = Run(store=store, run_name=run_name)
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

    optimizer = torch.optim.Adam(
        model.classifier[-1].parameters(),
        lr=float(config["lr"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    with run:
        train_loader = attach(
            train_loader,
            run,
            role="train",
            dataset_uri=config.get("train_dataset_uri"),
        )
        test_loader = attach(
            test_loader,
            run,
            role="test",
            dataset_uri=config.get("test_dataset_uri"),
        )

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
    relative_file_path: str,
) -> dict[str, Any]:
    sample_id = f"filepath:{relative_file_path}"
    return {
        "sample_id": sample_id,
        "occurrences": find_sample_occurrences(
            store,
            run_id,
            sample_id,
        ),
    }


def collect_pair_evidence(
    *,
    db_path: Path,
    report_path: Path,
    case: str,
    case_label: str,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    run_id_a: str,
    run_id_b: str,
    run_name_a: str,
    run_name_b: str,
    sample_file: str,
    mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
            relative_file_path=sample_file,
        )
        sample_lookup_b = build_sample_lookup_evidence(
            store,
            run_id=run_id_b,
            relative_file_path=sample_file,
        )
    finally:
        store.close()

    evidence = {
        "experiment": "repro_suite",
        "case": case,
        "case_label": case_label,
        "db_path": str(db_path),
        "report_path": str(report_path),
        "runs": {
            "run_a": {
                "run_id": run_id_a,
                "run_name": run_name_a,
                "config": config_a,
            },
            "run_b": {
                "run_id": run_id_b,
                "run_name": run_name_b,
                "config": config_b,
            },
        },
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
    if mutation is not None:
        evidence["mutation"] = mutation
    return evidence


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


def create_a3_modified_dataset_copy(
    *,
    source_data_root: Path,
    derived_data_root: Path,
) -> dict[str, Any]:
    if derived_data_root.exists():
        shutil.rmtree(derived_data_root)
    shutil.copytree(source_data_root, derived_data_root)

    train_root = derived_data_root / "train"
    relative_path = infer_first_train_sample_relative_path(train_root)
    target_path = train_root / relative_path

    with Image.open(target_path) as image:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        save_kwargs = image.info.copy()
        flipped.save(target_path, **_filtered_save_kwargs(save_kwargs))

    return {
        "source_data_root": str(source_data_root),
        "derived_data_root": str(derived_data_root),
        "mutated_relative_path": relative_path,
        "mutation_type": "horizontal_flip_overwrite",
    }


def _filtered_save_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "dpi",
        "icc_profile",
        "quality",
        "optimize",
        "progressive",
        "subsampling",
        "compress_level",
    }
    return {key: value for key, value in raw_kwargs.items() if key in allowed}
