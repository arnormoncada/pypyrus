from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from torch.utils.data import DataLoader


def _load_example_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "ufo_sightings"
        / "train_shape_classifier.py"
    )
    spec = importlib.util.spec_from_file_location("ufo_shape_example", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load example module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_ufo_records_maps_rare_shapes_to_other(tmp_path) -> None:
    module = _load_example_module()
    data_path = tmp_path / "ufo.csv"

    fieldnames = [
        "datetime",
        "city",
        "state",
        "country",
        "shape",
        "duration (seconds)",
        "duration (hours/min)",
        "comments",
        "date posted",
        "latitude",
        "longitude ",
    ]
    rows = []
    for index, shape in enumerate(
        [
            "light",
            "triangle",
            "circle",
            "fireball",
            "other",
            "unknown",
            "sphere",
            "disk",
            "oval",
            "formation",
            "rare_shape",
            "rare_shape_2",
        ]
    ):
        rows.append(
            {
                "datetime": f"1/1/2000 0{index}:00",
                "city": "city",
                "state": "st",
                "country": "us",
                "shape": shape,
                "duration (seconds)": "60",
                "duration (hours/min)": "1 minute",
                "comments": f"comment {index}",
                "date posted": "1/2/2000",
                "latitude": "1.0",
                "longitude ": "2.0",
            }
        )

    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records, label_names = module.load_ufo_records(data_path, top_k_shapes=10)

    assert label_names[-1] == "other"
    assert len(label_names) == 11
    assert records[0]["record_id"] == "ufo_0"
    assert records[-1]["record_id"] == "ufo_11"
    assert records[-2]["label"] == "other"
    assert records[-1]["label"] == "other"


def test_split_records_is_deterministic() -> None:
    module = _load_example_module()
    records = [
        {"record_id": f"ufo_{index}", "comments": f"comment {index}", "label": "light"}
        for index in range(20)
    ]

    train_a, test_a = module.split_records(records, seed=11)
    train_b, test_b = module.split_records(records, seed=11)

    assert train_a == train_b
    assert test_a == test_b
    assert len(train_a) == 16
    assert len(test_a) == 4


def test_ufo_comments_dataset_exposes_records_and_collates() -> None:
    module = _load_example_module()

    class FakeTokenizer:
        def __call__(self, text, *, truncation, padding, max_length, return_tensors):
            assert truncation is True
            assert padding == "max_length"
            assert return_tensors == "pt"
            import torch

            tokens = [len(text) % 7, max_length % 13, 1]
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    dataset = module.UFOCommentsDataset(
        [
            {"record_id": "ufo_10", "comments": "orange light in sky", "label": "light"},
            {"record_id": "ufo_11", "comments": "triangle above trees", "label": "triangle"},
        ],
        tokenizer=FakeTokenizer(),
        label_to_id={"light": 0, "triangle": 1},
        max_length=16,
    )

    assert dataset.records[0]["record_id"] == "ufo_10"
    first_sample = next(iter(dataset))
    assert first_sample["record_id"] == "ufo_10"
    assert module.ufo_sample_id_resolver(dataset, 0, first_sample) == "record_id:ufo_10"

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=module.ufo_comments_collate,
    )
    batch = next(iter(loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].tolist() == [0, 1]
