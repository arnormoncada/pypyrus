"""
Tiny smoke test for PyPyrus instrumentation.

This verifies that:
- dataset wrapper injects sample IDs
- collate preserves IDs
- dataloader emits BatchDeliveredEvent

Should emit:
RunStartedEvent
DatasetRegisteredEvent
TransformDeclaredEvent (if dataset has transform)
BatchDeliveredEvent x 4
RunEndedEvent
"""

import torch
from torch.utils.data import Dataset, DataLoader

from pypyrus.core.run import Run
from pypyrus.core.attach import attach


# ---------------------------------------------------------
# Dummy dataset
# ---------------------------------------------------------

class TinyDataset(Dataset):
    def __init__(self, n=10):
        self.data = torch.arange(n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return something simple
        return self.data[idx], idx


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

dataset = TinyDataset(10)

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
)

# ---------------------------------------------------------
# Run with PyPyrus
# ---------------------------------------------------------

with Run() as run:

    loader = attach(loader, run)

    print("\nIterating batches:\n")

    for batch in loader:
        print("Batch:", batch)

print("\nDone.")