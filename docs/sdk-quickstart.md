# SDK Quickstart

This page shows the smallest useful PyPyrus workflow:

1. create a `Run`
2. attach one or more DataLoaders
3. train as usual
4. inspect the recorded run with the CLI

## Install

```bash
pip install -e .
```

PyPyrus currently targets map-style PyTorch DataLoaders.

If you want to run optional example paths that use extra dependencies, install:

```bash
pip install -e ".[examples]"
```

## Minimal Example

```python
from torch.utils.data import DataLoader

from pypyrus import Run, attach


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with Run() as run:
    train_loader = attach(train_loader, run, role="train")
    test_loader = attach(test_loader, run, role="test")

    for batch in train_loader:
        ...
```

## What `Run` Does

`Run` is the audit boundary for one training execution.

When a run starts, PyPyrus records:

- a new `run_id`
- a best-effort code reference
- a best-effort environment snapshot

When the run ends, PyPyrus records the final status and flushes the store.

By default, `Run()` uses the local SQLite store backing `./pypyrus.db`, unless
you override it with `PYPYRUS_DB` or pass a custom store instance.

## What `attach(...)` Does

`attach(...)` wraps a PyTorch DataLoader so PyPyrus can observe:

- dataset registration
- loader registration
- transform declarations when available
- delivered batches and their sample IDs

The required argument is:

- `role`

Use roles like:

- `train`
- `val`
- `test`

Roles let PyPyrus distinguish multiple loaders in one run.

## Optional `sample_id_resolver`

If PyPyrus cannot infer the right sample identity from your dataset shape, pass
`sample_id_resolver=` to `attach(...)`.

Example:

```python
def sample_id_resolver(dataset, index, sample):
    row = dataset.records[index]
    return f"record_id:{row['customer_id']}"


with Run() as run:
    train_loader = attach(
        train_loader,
        run,
        role="train",
        sample_id_resolver=sample_id_resolver,
    )
```

See [Custom Dataset Integration](custom-dataset-integration.md) for the full
contract.

## Inspect the Result

After a run finishes:

```bash
pypyrus runs list
pypyrus runs show <run_id>
```

Useful next steps:

- [CLI Usage](cli-usage.md)
- [Sample Identity Contract](sample-identity-contract.md)
