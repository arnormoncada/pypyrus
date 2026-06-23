<p align="center">
  <img src="docs/PyPyrus large.png" alt="PyPyrus logo" width="640">
</p>

<p align="center"><strong>Data provenance for PyTorch training runs</strong></p>
<p align="center">Track datasets, batches, and sample usage across runs.</p>

PyPyrus is a data provenance layer for PyTorch training runs. It attaches to
DataLoaders, records which datasets and batches were delivered, and gives you a
CLI to inspect, compare, and query those runs afterward.

The current focus is single-machine PyTorch workflows. The public SDK surface is
intentionally small:

- `Run`
- `attach`

## New to PyPyrus? 
#### *(or you are my thesis supervisor/examiner and want to try pypyrus out)* 
Start with the [SDK Quickstart](docs/sdk-quickstart.md) for a short end-to-end walkthrough using the plant seedlings example. More references and docs are at the bottom of this page 📜

## Install

```bash
git clone https://github.com/arnormoncada/pypyrus.git
cd pypyrus

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

Experiment datasets under `experiments/*/data/` use Git LFS. After cloning,
run:

```bash
git lfs install
git lfs pull
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
        dataset_uri="path/to/data/scrubbed.csv",
        dataset_name="PokemonCSVDataset",
        dataset_version_hint="preprocessed-v1",
    )
```

## Run Store Modes

`Run` supports two store modes:

- `sync` (default): events are written synchronously on the caller path.
- `buffered_strict` (WIP): Currently rather simple, but events are enqueued and written by a
    dedicated writer thread using the SQLiteStore (sync). The defualt store is recommended and battle-tested, this one is still a concept in development use at your own risk.

Example:

```python
with Run(store_mode="buffered_strict", buffered_queue_size=1024) as run:
        tracked_loader = attach(loader, run, role="train")
        for batch in tracked_loader:
                ...
```

## Minimal CLI Workflow
See the PyPyrus CLI help:
```bash
pypyrus --help
```

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
- [Forest Covertype tabular classification](experiments/forest_covertype/train_covtype_mlp.py)

## Minimal Example: MNIST

- [Simple MLP training on MNIST](examples/mnist/train_simple_mlp.py)