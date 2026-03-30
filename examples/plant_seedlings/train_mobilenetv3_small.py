"""
Train a small MobileNetV3 classifier on an ImageFolder-style plant seedlings split.

Expected dataset layout:

    <data_root>/
      train/
        <class_name>/
          image_1.png
          ...
      test/
        <class_name>/
          image_2.png
          ...

Example:

    python examples/plant_seedlings/train_mobilenetv3_small.py \
      --data-root examples/plant_seedlings/data/split \
      --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from pypyrus import Run, attach


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small MobileNetV3 model on an ImageFolder-style dataset split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs. Default: 3",
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
        default=7,
        help="Random seed for model init and train-loader shuffle. Default: 7",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.seed)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    train_data = datasets.ImageFolder(train_root, transform=transform)
    test_data = datasets.ImageFolder(test_root, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_in_features, len(train_data.classes))
    model.to(device)

    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    with Run() as run:
        print(f"seed={args.seed}")
        train_loader = attach(train_loader, run, role="train")
        test_loader = attach(test_loader, run, role="test")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_accuracy:.4f}"
            )

    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
