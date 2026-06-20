from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pypyrus.core.attach import attach
from pypyrus.core.run import Run

from tests.helpers import TinyMapDataset


def _build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def _train_for_epochs(
    loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    epochs: int,
) -> list[float]:
    losses: list[float] = []
    for _ in range(epochs):
        for features, labels in loader:
            logits = model(features)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses

def test_instrumented_training_preserves_final_weights(store) -> None:

    torch.manual_seed(1234)
    initial_state = copy.deepcopy(_build_model().state_dict())

    baseline_model = _build_model()
    baseline_model.load_state_dict(copy.deepcopy(initial_state))
    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=0.05)

    instrumented_model = _build_model()
    instrumented_model.load_state_dict(copy.deepcopy(initial_state))
    instrumented_optimizer = torch.optim.SGD(
        instrumented_model.parameters(),
        lr=0.05,
    )

    loss_fn = nn.CrossEntropyLoss()
    baseline_loader = DataLoader(
        TinyMapDataset(n=18),
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )
    comparison_loader = DataLoader(
        TinyMapDataset(n=18),
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    baseline_losses = _train_for_epochs(
        baseline_loader,
        baseline_model,
        baseline_optimizer,
        loss_fn,
        epochs=2,
    )

    with Run(store=store) as run:
        attached_loader = attach(comparison_loader, run, role="train")
        instrumented_losses = _train_for_epochs(
            attached_loader,
            instrumented_model,
            instrumented_optimizer,
            loss_fn,
            epochs=2,
        )

    assert baseline_losses == instrumented_losses

    baseline_state = copy.deepcopy(baseline_model.state_dict())
    instrumented_state = copy.deepcopy(instrumented_model.state_dict())
    assert baseline_state.keys() == instrumented_state.keys()
    for name, baseline_tensor in baseline_state.items():
        assert torch.equal(baseline_tensor, instrumented_state[name]), name
