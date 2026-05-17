<div style="display: flex; align-items: center; gap: 24px;">
    <img src="docs/PyPyrus-logo.png" alt="PyPyrus logo" width="320">
    <div>
        <h1>PyPyrus</h1>
        <p><strong>Data provenance for PyTorch training runs</strong></p>
        <p>Track datasets, batches, and sample usage across runs.</p>
    </div>
</div>

PyPyrus is a data provenance layer for PyTorch training runs. It attaches to
DataLoaders, records which datasets and batches were delivered, and gives you a
CLI to inspect, compare, and query those runs afterward.

The current focus is single-machine PyTorch workflows. The public SDK surface is
intentionally small:

- `Run`
- `attach`

## Install

```bash
pip install -e .
```

Optional example dependencies, including the UFO transformer path:

```bash
pip install -e ".[examples]"
```

## Minimal SDK Example

```python
from torch.utils.data import DataLoader

from pypyrus import Run, attach


loader = DataLoader(dataset, batch_size=32, shuffle=True)

with Run() as run:
    tracked_loader = attach(loader, run, role="train")
    for batch in tracked_loader:
        ...
```

PyPyrus records run metadata, dataset identity, loader registrations, and the
batch stream delivered to your training loop.

Dataset contract notes:

- map-style datasets should inherit `torch.utils.data.Dataset`
- iterable datasets should inherit `torch.utils.data.IterableDataset`
- iterable datasets must provide `sample_id_resolver=...` at `attach(...)`

If your dataset comes from a source file that PyPyrus cannot infer from the
dataset object, pass explicit provenance metadata at attach time:

```python
with Run() as run:
    tracked_loader = attach(
        loader,
        run,
        role="train",
        dataset_uri="/data/scrubbed.csv",
        dataset_name="PokemonCSVDataset",
        dataset_version_hint="preprocessed-v1",
    )
```

## Run Store Modes

`Run` supports two store modes:

- `sync` (default): events are written synchronously on the caller path.
- `buffered_strict` (experimental): events are enqueued and written by a
    dedicated writer thread with strict backpressure (no event dropping).

Example:

```python
with Run(store_mode="buffered_strict", buffered_queue_size=1024) as run:
        tracked_loader = attach(loader, run, role="train")
        for batch in tracked_loader:
                ...
```

Tradeoffs:

- `sync` is simpler and often better for low-throughput or short runs.
- `buffered_strict` is experimental. It can reduce write-path blocking, but it
    still keeps event preparation on the training path and may add queue/thread
    coordination cost.
- `buffered_strict` should not be treated as a guaranteed performance win.
- In strict mode, if the queue is full, producer threads block until space is
    available; PyPyrus does not drop events.

## Minimal CLI Workflow

```bash
pypyrus runs list
pypyrus runs show <run_id>
pypyrus compare <run_a> <run_b>
pypyrus batches show <run_id> --step 12
pypyrus samples find <run_id> --sample-id index:3
```

## User Docs

- [SDK Quickstart](docs/sdk-quickstart.md)
- [Sample Identity Contract](docs/sample-identity-contract.md)
- [CLI Usage](docs/cli-usage.md)
- [Custom Dataset Integration](docs/custom-dataset-integration.md)

## Experiments

- [Plant seedlings image classification](experiments/plant_seedlings/train_mobilenetv3_small.py)
- [UFO sightings shape classification](experiments/ufo_sightings/train_shape_classifier.py)

## Internal Design Notes

These are working documents for implementation planning, not end-user docs:

- [Sample ID resolver architecture](docs/sample-id-resolver-architecture.md)
- [Internal working notes](docs/internal-working-notes.md)
