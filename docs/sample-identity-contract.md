# Sample Identity Contract

PyPyrus stores a normalized `sample_id` for each delivered sample when it can.

The goal is simple:

- use the strongest stable sample identity available
- fall back safely when a stronger identity is not available

These contracts are **PyPyrus conventions**, not PyTorch-wide standards.

## Dataset Type Requirements

PyPyrus expects the DataLoader dataset to follow the PyTorch dataset base
classes explicitly:

- map-style datasets should inherit `torch.utils.data.Dataset`
- iterable datasets should inherit `torch.utils.data.IterableDataset`

Datasets that only duck-type these interfaces without inheriting the PyTorch
bases are rejected at `attach(...)`.

## Normalized Sample ID Schemes

PyPyrus currently emits these normalized schemes:

- `filepath:<relative-path>`
- `record_id:<value>`
- `row:<value>`
- `logical:<split>#<index-or-key>`
- `index:<value>`

## Resolution Order

PyPyrus resolves sample IDs in this order:

1. user-provided `sample_id_resolver`
2. file collection contract
3. structured record contract
4. narrow logical/framework compatibility cases
5. fallback positional identity

For `IterableDataset`, PyPyrus does not use the built-in fallback path.
Iterable datasets must provide `sample_id_resolver=...`.

## File Collection Contract

Use this family when one file corresponds to one sample.

Recommended contract:

- `.samples`
- `.root`

Recognized compatibility alias:

- `.imgs`

If PyPyrus can read a file path from `.samples` and relate it to `.root`, it
emits:

- `filepath:<relative-path>`

Example:

- `filepath:train/cats/cat_12.jpg`

PyTorch examples:

- `torchvision.datasets.DatasetFolder`
- `torchvision.datasets.ImageFolder`

## Structured Record Contract

Use this family when a dataset is really a collection of rows or records.

Recommended contract:

- `.records`

Recognized compatibility aliases:

- `.rows`
- `.record_ids`
- `.ids`

Preferred record key fields, in priority order:

- `record_id`
- `id`
- `uuid`
- `key`

Behavior:

- if `record_ids` or `ids` exist, PyPyrus emits `record_id:<value>`
- if `records` or `rows` exist and the record exposes one of the key fields,
  PyPyrus emits `record_id:<value>`
- if `records` or `rows` exist but no key field is available, PyPyrus emits
  `row:<index>`

Examples:

- `record_id:customer_84291`
- `row:183`

## Logical / Framework Compatibility

This is intentionally narrow in the current MVP.

PyPyrus has a small built-in logical compatibility case for some torchvision
datasets where path- or record-style identity is not the natural fit.

This should be treated as compatibility behavior, not as a general user-facing
contract.

## Fallback Positional Identity

If no stronger built-in contract matches, PyPyrus falls back to:

- `index:<i>`

This is the default map-style positional identity.

Example:

- `index:42`

## Dataset Scope Matters

For file-based lookup, sample identity is interpreted together with dataset
context.

That means:

- `filepath:class_a/item_0.txt` is readable and stable
- but exact lookup is effectively scoped by `dataset_id + sample_id`

This is why:

- `samples find --file ... --dataset-path ...` is dataset-scoped
- bare `--sample-id` can match multiple datasets in one run

## When To Use `sample_id_resolver=...`

Use a custom resolver when:

- your dataset does not expose the built-in contract attrs
- your built-in sample identity should be stronger than the default one
- your logical sample key matters more than path or index
- you are attaching an `IterableDataset`

Example:

```python
def sample_id_resolver(dataset, index, sample):
    row = dataset.records[index]
    return f"record_id:{row['customer_id']}"
```

See [Custom Dataset Integration](custom-dataset-integration.md) for concrete
patterns.
