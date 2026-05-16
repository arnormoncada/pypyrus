# Custom Dataset Integration

PyPyrus can infer sample identity automatically for a few dataset shapes.

If it cannot, use `sample_id_resolver=` in `attach(...)`.

## Dataset Base Class Requirement

PyPyrus expects your dataset to inherit the corresponding PyTorch base class:

- map-style datasets: `torch.utils.data.Dataset`
- iterable datasets: `torch.utils.data.IterableDataset`

Custom objects that only implement `__getitem__`, `__len__`, or `__iter__`
without inheriting these bases are rejected at `attach(...)`.

## Option 1: Fit a Built-In Contract

### File Collection Datasets

If one file corresponds to one sample, the recommended contract is:

- `.samples`
- `.root`

Example:

```python
class MyImageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = [
            (str(root / "cats" / "cat_1.jpg"), 0),
            (str(root / "dogs" / "dog_1.jpg"), 1),
        ]
```

PyPyrus can then emit:

- `filepath:cats/cat_1.jpg`

Compatibility alias:

- `.imgs`

### Structured Record Datasets

If your dataset is really a collection of rows or records, the recommended
contract is:

- `.records`

and ideally each record exposes:

- `record_id`

Example:

```python
class MyTabularDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __getitem__(self, index):
        row = self.records[index]
        return features_from_row(row), label_from_row(row)
```

With records like:

```python
{"record_id": "customer_84291", "age": 42, "label": 1}
```

PyPyrus can then emit:

- `record_id:customer_84291`

Compatibility aliases:

- `.rows`
- `.record_ids`
- `.ids`

If your records exist but do not expose a stable key, PyPyrus falls back to:

- `row:<index>`

## Option 2: Provide `sample_id_resolver=...`

Use this when:

- your dataset shape does not match a built-in contract
- your real logical sample identity is better than the default one
- your dataset inherits `IterableDataset`

Example:

```python
from pypyrus import Run, attach


def sample_id_resolver(dataset, index, sample):
    row = dataset.records[index]
    return f"record_id:{row['customer_id']}"


with Run() as run:
    tracked_loader = attach(
        loader,
        run,
        role="train",
        sample_id_resolver=sample_id_resolver,
    )
```

The resolver receives:

- `dataset`
- `index`
- `sample`

For map-style datasets, `index` is the dataset index.

For iterable datasets, `index` is the stream position. Iterable datasets must
provide `sample_id_resolver=...`.

It can return:

- a normalized string like `record_id:customer_84291`
- an integer, which PyPyrus normalizes to `index:<value>`
- or a `SampleIdResolution` for full control

## Full-Control Resolver Result

If you want to control both the stored ID and the metadata shown in `runs show`,
return `SampleIdResolution` directly:

```python
from pypyrus.core.sample_id import SampleIdResolution


def sample_id_resolver(dataset, index, sample):
    row = dataset.records[index]
    return SampleIdResolution(
        sample_id=f"record_id:{row['customer_id']}",
        sample_id_scheme="record_id",
        sample_id_resolver="user_override",
    )
```

## Dataset Source Provenance

Dataset provenance and sample identity are separate:

- dataset provenance: where the dataset came from
- sample identity: how one sample is named inside that dataset

Example:

- dataset URI: `/data/scrubbed.csv`
- sample ID: `record_id:customer_84291`

Example:

```python
tracked_loader = attach(
    loader,
    run,
    role="train",
    dataset_uri="/data/scrubbed.csv",
    dataset_name="CustomerCSVDataset",
)
```

That records the source file. Sample IDs still come from `.records`,
`.record_ids`, or `sample_id_resolver=...`.

### Best Practice

For image-folder or per-file datasets:

- expose `.samples` + `.root`

For CSV / Parquet / single-file tabular datasets:

- expose a path-like attribute such as `.path` when that fits your dataset class
- or pass `dataset_uri=...` explicitly to `attach(...)`
- use `.records`, `.record_ids`, or `sample_id_resolver=...` for row identity

### Best Practice For Row-Based Datasets

For row-based datasets, prefer IDs that survive splitting and reordering.

Best to worst:

1. stable record key from the source data
   Example: `id`, `uuid`, `record_id`, `key`
2. explicit `sample_id_resolver=...`
   when your stable key exists but is not exposed through a built-in contract
3. fallback `row:<index>`
   only when no better row identity exists

Example with stable row IDs:

```python
class CustomerDataset(Dataset):
    def __init__(self, records, *, path):
        self.records = records
        self.path = path

    def __getitem__(self, index):
        row = self.records[index]
        return features_from_row(row), label_from_row(row)
```

With records like:

```python
{"record_id": "cust_84291", "age": 42, "label": 1}
```

PyPyrus can identify samples as:

- `record_id:cust_84291`

This is the recommended shape for train/test splits from one source file.

### Split Caveat For `row:<index>`

Be careful with `row:<index>` for train/test splits.

Example:

- source file: `/data/scrubbed.csv`
- train split uses some subset of rows
- test split uses a different subset of rows

If PyPyrus uses `row:<index>`:

- `row:0` in the train dataset means "first row in the train dataset view"
- `row:0` in the test dataset means "first row in the test dataset view"

Those are not necessarily the same source row.

If you need exact row identity across splits, do not rely on `row:<index>`.
Provide:

- stable row keys in `.records`
- or `.record_ids`
- or `sample_id_resolver=...`

Use the explicit attach-time override when:

- your dataset wraps a source file but does not expose its path
- the dataset class name alone is not descriptive enough in run reports
- you want to include a preprocessing/version label with
  `dataset_version_hint=...`

## Custom Collate That Reorders or Drops Samples

If your collate function reorders or filters samples, PyPyrus cannot infer the
new ID order unless you pass IDs through collate explicitly. Use
`id_aware_collate=True` and make your collate function accept
`(samples, sample_ids)` and return `(batch, remapped_ids)`.

If your collate only shuffles within the batch (no drop/dup), this is fine: the
batch still contains the same samples, and batch membership stays correct. The
only mismatch is ordering: IDs remain in the original order, so they no longer
align with the shuffled payload positions.

Example (reorder and filter in collate):

```python
from torch.utils.data import DataLoader
from pypyrus import Run, attach


def id_aware_collate(samples, sample_ids):
    # Example: keep even-indexed samples and reverse their order.
    kept = [(s, sid) for s, sid in zip(samples, sample_ids) if sid.endswith(":0") or sid.endswith(":2")]
    kept = list(reversed(kept))

    remapped_samples = [s for s, _ in kept]
    remapped_ids = [sid for _, sid in kept]

    # Build your batch from the remapped samples.
    batch = remapped_samples
    return batch, remapped_ids


with Run() as run:
    tracked_loader = attach(
        loader,
        run,
        role="train",
        id_aware_collate=True,
    )
```

Why this matters:

- PyPyrus stores `remapped_ids` for each delivered batch.
- If you reorder or drop samples, returning `(batch, remapped_ids)` is the only
  way to guarantee correct sample-to-ID mapping in provenance events.

If you need exact per-sample alignment (e.g., to map a specific payload slot to
its ID), enable `id_aware_collate=True` even for shuffle-only collates.

If your collate needs additional parameters, pre-bind them before constructing
the DataLoader collate function (for example with `functools.partial`).

```python
from functools import partial


def custom_collate(samples, sample_ids, threshold, mode):
    # build batch using samples (+ sample_ids when id_aware_collate=True)
    ...


collate_fn = partial(custom_collate, threshold=0.3, mode="fast")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

tracked_loader = attach(loader, run, role="train", id_aware_collate=True)
```

When `id_aware_collate=False`, PyPyrus may still call a collate that accepts
`sample_ids`, but it always stores the original ordered IDs unless
`id_aware_collate=True`.

## What If Collate Changes Batch Size

If your collate drops or duplicates items, PyPyrus has no safe, generic way to
reconstruct the correct ID list after the fact. Length mismatches alone do not
tell us how items were filtered or duplicated. In those cases, the best and
explicit approach is `id_aware_collate=True`, where your collate function
returns the remapped IDs alongside the batch.

## Recommended Guidance

Use the built-in contracts when they fit naturally:

- `.samples` + `.root` for file collections
- `.records` + `record_id` for structured records

Use `sample_id_resolver=` when:

- your internal dataset shape differs
- your logical sample key is domain-specific
- you want to guarantee a stronger stable identity than path or index
- you want row/sample identity to stay separate from dataset source provenance

## Examples in This Repo

- [Plant seedlings example](../examples/plant_seedlings/train_mobilenetv3_small.py)
  shows file-collection datasets
- [UFO sightings example](../examples/ufo_sightings/train_shape_classifier.py)
  shows a structured-record dataset shape and defaults to the fast torch-native
  classifier path; the transformer path is optional

Related pages:

- [Sample Identity Contract](sample-identity-contract.md)
- [SDK Quickstart](sdk-quickstart.md)
