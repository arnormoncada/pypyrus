# Custom Dataset Integration

PyPyrus can infer sample identity automatically for a few dataset shapes.

If your dataset does not fit one of those shapes cleanly, use
`sample_id_resolver=` when you call `attach(...)`.

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

## Examples in This Repo

- [Plant seedlings example](../examples/plant_seedlings/train_mobilenetv3_small.py)
  shows file-collection datasets
- [UFO sightings example](../examples/ufo_sightings/train_shape_classifier.py)
  shows a structured-record dataset shape and defaults to the fast torch-native
  classifier path; the transformer path is optional

Related pages:

- [Sample Identity Contract](sample-identity-contract.md)
- [SDK Quickstart](sdk-quickstart.md)
