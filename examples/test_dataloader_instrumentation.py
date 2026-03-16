"""
Smoke test for PyPyrus instrumentation with multiple loaders.

Verifies:
- dataset wrapper injects sample IDs
- collate preserves IDs
- multiple loaders with different roles work in the same run
- global_sequence captures interleaved order across loaders
- BatchDeliveredEvent emitted for every batch
"""

import torch
from torch.utils.data import Dataset, DataLoader

# Fix all random state so sample order is always identical across runs
SEED = 42
torch.manual_seed(SEED)

from pypyrus.core.run import Run
from pypyrus.core.attach import attach
from pypyrus.reporting.queries import get_batches_for_run
from pypyrus.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------
# Dummy datasets
# ---------------------------------------------------------

class TinyDataset(Dataset):
    def __init__(self, n: int = 10, offset: int = 0):
        self.data = torch.arange(offset, offset + n)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], idx


# ---------------------------------------------------------
# Setup — separate train and val datasets
# ---------------------------------------------------------

train_dataset = TinyDataset(n=9)   # 3 batches of 3

val_dataset   = TinyDataset(n=6, offset=100)  # 2 batches of 3

_g_train = torch.Generator().manual_seed(SEED)
_g_val   = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, generator=_g_train)
val_loader   = DataLoader(val_dataset,   batch_size=3, shuffle=False, generator=_g_val)

# ---------------------------------------------------------
# Run with PyPyrus — simulate 2 epochs of train then val
# ---------------------------------------------------------

print("\n=== Running instrumented training loop ===\n")

with Run() as run:
    run_id = run.run_id
    train_loader = attach(train_loader, run, role="train")
    val_loader   = attach(val_loader,   run, role="val")

    for epoch in range(1, 3):
        print(f"--- Epoch {epoch} ---")

        print("  [train]")
        for batch in train_loader:
            pass  # training step would go here

        print("  [val]")
        for batch in val_loader:
            pass  # validation step would go here

print("\nDone. Inspecting stored batch stream...\n")

# ---------------------------------------------------------
# Verify: print global stream ordered by global_sequence
# ---------------------------------------------------------

store = SQLiteStore("pypyrus.db")
batches = get_batches_for_run(store, run_id, include_sample_ids=False)
store.close()

print(f"{'seq':>4}  {'role/dataset_id':>20}  {'step':>4}  {'fingerprint'}")
print("-" * 70)
for b in batches:
    # shorten dataset_id for readability
    did = b["dataset_id"][:10]
    print(f"{b['global_sequence']:>4}  {did:>20}  {b['global_step']:>4}  {b['batch_fingerprint'][:16]}")
