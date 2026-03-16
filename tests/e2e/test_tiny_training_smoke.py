from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run

from tests.helpers import TinyMapDataset, fetch_all, fetch_one


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def test_tiny_training_loop_persists_provenance_smoke(db_path, store) -> None:
    torch.manual_seed(7)

    loader = DataLoader(TinyMapDataset(n=12), batch_size=4, shuffle=False)
    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    with Run(store=store) as run:
        train_loader = attach(loader, run, role="train")
        for features, labels in train_loader:
            logits = model(features)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    run_row = fetch_one(
        db_path,
        "SELECT status FROM runs WHERE run_id = ?",
        (run.run_id,),
    )
    assert run_row["status"] == "success"

    batch_rows = fetch_all(
        db_path,
        "SELECT batch_size FROM batch_delivered WHERE run_id = ? ORDER BY global_sequence",
        (run.run_id,),
    )
    assert [row["batch_size"] for row in batch_rows] == [4, 4, 4]
