# PyPyrus

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

## Examples

- [Plant seedlings image classification](examples/plant_seedlings/train_mobilenetv3_small.py)
- [UFO sightings shape classification](examples/ufo_sightings/train_shape_classifier.py)

## Internal Design Notes

These are working documents for implementation planning, not end-user docs:

- [Sample ID resolver architecture](docs/sample-id-resolver-architecture.md)
- [Internal working notes](docs/internal-working-notes.md)
