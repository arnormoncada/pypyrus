# Sample Identity Contract

PyPyrus stores one `sample_id` for each delivered sample.

Recommended usage:

- provide `sample_id_resolver=` in `attach(...)`

This is the strongest and most explicit way to control sample identity.

## Dataset Requirements

- map-style datasets should inherit `torch.utils.data.Dataset`
- iterable datasets should inherit `torch.utils.data.IterableDataset`

Datasets that only duck-type these interfaces are rejected at `attach(...)`.

## Resolution Order

For map-style datasets, PyPyrus resolves sample IDs in this order:

1. `sample_id_resolver=...`
2. built-in contract
3. fallback `index:<i>`

For iterable datasets:

1. `sample_id_resolver=...`

Iterable datasets require `sample_id_resolver=...`.

## Built-In Contracts

### File collections

Use this when one file corresponds to one sample.

Expose:

- `.samples`
- `.root`

Compatibility alias:

- `.imgs`

PyPyrus emits:

- `filepath:<relative-path>`

### Structured records

Use this when your dataset is really a collection of rows or records.

Expose one of:

- `.records`
- `.rows`
- `.record_ids`
- `.ids`

Preferred record key fields:

- `record_id`
- `id`
- `uuid`
- `key`

PyPyrus emits:

- `record_id:<value>` when a stable key is available
- `row:<index>` otherwise

## Normalized Schemes

PyPyrus currently emits:

- `filepath:<relative-path>`
- `record_id:<value>`
- `row:<value>`
- `index:<value>`

## Samples CLI

Use the stored sample ID directly:

```bash
pypyrus samples find <run_id> --sample-id record_id:cust_84291
```

If the same sample ID appears in multiple datasets in one run, `samples find`
can return matches from more than one dataset.