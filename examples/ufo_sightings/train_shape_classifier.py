"""
Train a UFO shape classifier on sighting comments.

The task is to predict the reported UFO shape from the `comments` field in the
structured UFO sightings CSV. The label space is the top 10 most common shapes
plus `other`. The script supports two model paths:

* `fast`: a tiny torch-native text classifier that trains quickly
* `transformer`: a small pretrained BERT model

Example:

    python examples/ufo_sightings/train_shape_classifier.py \
      --data-path examples/ufo_sightings/data/scrubbed.csv \
      --epochs 2 \
      --model fast
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from pypyrus import Run, attach


MODEL_NAME = "prajjwal1/bert-tiny"
FAST_MODEL_NAME = "fasttext_like"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a UFO shape classifier from comments."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the UFO sightings CSV file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs. Default: 2",
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
        default=None,
        help="Learning rate. Defaults to 1e-3 for --model fast and 5e-5 for --model transformer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for split, model init, and train-loader shuffle. Default: 7",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenized comment length. Default: 128",
    )
    parser.add_argument(
        "--model",
        choices=("fast", "transformer"),
        default="fast",
        help="Model type to train. Default: fast",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"CSV data path not found: {data_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = args.lr if args.lr is not None else default_learning_rate(args.model)

    records, label_names = load_ufo_records(data_path)
    train_records, test_records = split_records(records, seed=args.seed)

    label_to_id = {label: index for index, label in enumerate(label_names)}
    tokenizer, model, model_name = build_model_stack(
        model_type=args.model,
        train_records=train_records,
        num_labels=len(label_names),
        max_length=args.max_length,
    )

    train_dataset = UFOCommentsDataset(
        train_records,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
    )
    test_dataset = UFOCommentsDataset(
        test_records,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with Run() as run:
        print(f"seed={args.seed}")
        print(f"model={model_name}")
        print(f"labels={label_names}")
        train_loader = attach(train_loader, run, role="train")
        test_loader = attach(test_loader, run, role="test")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            test_loss, test_accuracy = evaluate(model, test_loader, device)
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_accuracy:.4f}"
            )

    return 0


def load_ufo_records(
    data_path: Path,
    *,
    top_k_shapes: int = 10,
) -> tuple[list[dict[str, str]], list[str]]:
    raw_rows: list[dict[str, str]] = []

    with data_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            comments = normalize_text(row.get("comments"))
            shape = normalize_text(row.get("shape"))
            if not comments or not shape:
                continue
            raw_rows.append(
                {
                    "record_id": f"ufo_{row_index}",
                    "comments": comments,
                    "shape": shape,
                }
            )

    shape_counts = Counter(row["shape"] for row in raw_rows)
    top_shapes = [shape for shape, _ in shape_counts.most_common(top_k_shapes)]
    label_names = top_shapes + ["other"]

    records: list[dict[str, str]] = []
    top_shape_set = set(top_shapes)
    for row in raw_rows:
        label = row["shape"] if row["shape"] in top_shape_set else "other"
        records.append(
            {
                "record_id": row["record_id"],
                "comments": row["comments"],
                "label": label,
            }
        )

    return records, label_names


def split_records(
    records: list[dict[str, str]],
    *,
    seed: int,
    test_ratio: float = 0.2,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio))
    test_records = shuffled[:test_count]
    train_records = shuffled[test_count:]
    return train_records, test_records


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def default_learning_rate(model_type: str) -> float:
    if model_type == "transformer":
        return 5e-5
    return 1e-3


def build_model_stack(
    *,
    model_type: str,
    train_records: list[dict[str, str]],
    num_labels: int,
    max_length: int,
) -> tuple[Any, torch.nn.Module, str]:
    if model_type == "transformer":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
        )
        return tokenizer, model, MODEL_NAME

    vocab = build_vocabulary(train_records)
    tokenizer = SimpleWhitespaceTokenizer(vocab=vocab)
    model = FastTextLikeClassifier(
        vocab_size=len(vocab),
        num_labels=num_labels,
    )
    return tokenizer, model, FAST_MODEL_NAME


def build_vocabulary(
    records: list[dict[str, str]],
    *,
    max_vocab_size: int = 20_000,
) -> dict[str, int]:
    counts = Counter()
    for record in records:
        counts.update(record["comments"].split())

    vocab = {"<pad>": PAD_TOKEN_ID, "<unk>": UNK_TOKEN_ID}
    for token, _ in counts.most_common(max_vocab_size - len(vocab)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class UFOCommentsDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        *,
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record["comments"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_to_id[record["label"]], dtype=torch.long),
        }


class SimpleWhitespaceTokenizer:
    def __init__(self, *, vocab: dict[str, int]):
        self.vocab = vocab

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        if not truncation:
            raise ValueError("SimpleWhitespaceTokenizer expects truncation=True")
        if padding != "max_length":
            raise ValueError("SimpleWhitespaceTokenizer expects padding='max_length'")
        if return_tensors != "pt":
            raise ValueError("SimpleWhitespaceTokenizer expects return_tensors='pt'")

        token_ids = [
            self.vocab.get(token, UNK_TOKEN_ID)
            for token in text.split()
        ][:max_length]
        attention_mask = [1] * len(token_ids)

        if len(token_ids) < max_length:
            pad_count = max_length - len(token_ids)
            token_ids.extend([PAD_TOKEN_ID] * pad_count)
            attention_mask.extend([0] * pad_count)

        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


@dataclass
class ClassifierOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class FastTextLikeClassifier(torch.nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        self.classifier = torch.nn.Linear(embed_dim, num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ClassifierOutput:
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        masked_embeddings = embeddings * mask
        pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled)
        loss = (
            self.loss_fn(logits, labels)
            if labels is not None
            else torch.tensor(0.0, device=logits.device)
        )
        return ClassifierOutput(loss=loss, logits=logits)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total_examples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]

        total_loss += float(outputs.loss.item())
        total_batches += 1
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(labels.size(0))

    average_loss = total_loss / max(total_batches, 1)
    accuracy = correct / max(total_examples, 1)
    return average_loss, accuracy


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


if __name__ == "__main__":
    raise SystemExit(main())
