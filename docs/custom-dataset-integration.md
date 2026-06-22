# Custom Dataset Integration

Best practice: provide `sample_id_resolver=` in `attach(...)`.

That is the strongest and most explicit way to control sample identity.
PyPyrus can also infer sample identity automatically for a few built-in
map-style dataset shapes.

## Dataset Base Class Requirement

PyPyrus expects your dataset to inherit the corresponding PyTorch base class:

- map-style datasets: `torch.utils.data.Dataset`
- iterable datasets: `torch.utils.data.IterableDataset`

Custom objects that only implement `__getitem__`, `__len__`, or `__iter__`
without inheriting these bases are rejected at `attach(...)`.

## Option 1: Fit a Built-In Contract

This is the convenience path for map-style datasets when you do not provide
`sample_id_resolver=...`.

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

Accepted `.samples` / `.imgs` entry forms:

- a path-like value directly
- a tuple/list whose first element is the path, as in the example above

Absolute paths are generally **encouraged** for both obtaining the root and for the sample paths. PyPyrus will normalize them to relative paths for storage.

### Structured Record Datasets

If your dataset is really a collection of rows or records, PyPyrus supports two
built-in shapes.

Shape A: a record container whose items carry their own stable key.

- `.records`
- compatibility alias: `.rows`
- preferred key fields inside each record:
  - `record_id`
  - `id`
  - `uuid`
  - `key`

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

Shape B: a separate indexed list of stable record IDs aligned with the dataset
order.

- `.record_ids`
- compatibility alias: `.ids`

Example:

```python
class MyTabularDataset(Dataset):
    def __init__(self, rows, record_ids):
        self.rows = rows
        self.record_ids = record_ids

    def __getitem__(self, index):
        row = self.rows[index]
        return features_from_row(row), label_from_row(row)
```

Here `record_ids[index]` is treated as the sample ID for the sample produced by
`__getitem__(index)`.

PyPyrus can then emit:

- `record_id:<record_ids[index]>`

If your records exist but do not expose a stable key, PyPyrus falls back to:

- `row:<index>`

## Option 2: Provide `sample_id_resolver=...`

This is the recommended path.

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

Best practice is to provide dataset provenance explicitly with `dataset_uri=...`,
independent of dataset type, when you want a stable and clear source reference
in run reports.

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

That records the source file explicitly. Sample IDs still come from `.records`,
`.record_ids`, or `sample_id_resolver=...`.

PyPyrus may also infer source provenance from natural dataset attributes when
they exist:

- file-collection datasets often expose `.root`
- record-based or single-file datasets may expose a path-like attribute such as
  `.path`

When that inference is weak, unavailable, or less clear than the logical source
you want to report, prefer setting `dataset_uri=...` explicitly.

### Best Practice

- Pass `dataset_uri=...` explicitly to `attach(...)`
- use natural built-in source attributes like `.root` or `.path` when they fit
  your dataset class naturally

## Split Caveat For `row:<index>`

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

- stable row keys in `.records` / `.rows`
- or aligned `.record_ids` / `.ids`
- or `sample_id_resolver=...`

## Custom Collate That Reorders or Drops Samples

If your collate reorders, drops, or duplicates samples, use
`id_aware_collate=True` and make it accept `(samples, sample_ids)` and return
`(batch, remapped_ids)`.

If your collate only changes the order within the batch, batch membership stays
correct, but the stored IDs remain in the original order unless you opt into
`id_aware_collate=True`.

Example (reorder and filter in collate):

```python
from torch.utils.data import DataLoader
from pypyrus import Run, attach


def id_aware_collate(samples, sample_ids):
    kept = [(s, sid) for s, sid in zip(samples, sample_ids) if sid.endswith(":0") or sid.endswith(":2")]
    kept = list(reversed(kept))

    remapped_samples = [s for s, _ in kept]
    remapped_ids = [sid for _, sid in kept]
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

PyPyrus stores `remapped_ids` for the delivered batch. If you reorder, drop, or
duplicate samples, returning `(batch, remapped_ids)` keeps sample IDs aligned
with the delivered payload.

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

When `id_aware_collate=False`, PyPyrus stores the original ordered IDs.

## What If Collate Changes Batch Size

If your collate drops or duplicates items, PyPyrus has no safe, generic way to
reconstruct the correct ID list after the fact. Length mismatches alone do not
tell us how items were filtered or duplicated. In those cases, the best and
explicit approach is `id_aware_collate=True`, where your collate function
returns the remapped IDs alongside the batch.

## Recommended Guidance

Best practice is still `sample_id_resolver=...`, especially when you care about
stable sample identity across splits, wrappers, or custom dataset shapes.

Use the built-in contracts as the convenience path when they fit naturally:

- `.samples` + `.root` for file collections
- `.records` / `.rows` with stable key fields for structured records
- `.record_ids` / `.ids` aligned with dataset order for structured records

Use `sample_id_resolver=` when:

- your internal dataset shape differs
- your logical sample key is domain-specific
- you want to guarantee a stronger stable identity

## Examples in This Repo

- [Plant seedlings example](../experiments/plant_seedlings/train_mobilenetv3_small.py)
  shows file-collection datasets
- [Forest Covertype tabular classification](experiments/forest_covertype/train_covtype_mlp.py)


Related pages:

- [Sample Identity Contract](sample-identity-contract.md)
- [SDK Quickstart](sdk-quickstart.md)
