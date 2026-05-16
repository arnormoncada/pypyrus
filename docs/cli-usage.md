# CLI Usage

PyPyrus ships with a small inspection CLI:

- `runs list`
- `runs show`
- `compare`
- `batches show`
- `samples find`

Use `pypyrus --help` to see the top-level help.

## Database Selection

By default, the CLI reads:

- `./pypyrus.db`

You can override that with:

- `--db <path>`
- or `PYPYRUS_DB`

Use `--json` for structured output.

## `runs list`

List recorded runs:

```bash
pypyrus runs list
```

The default table shows:

- run ID
- status
- start time
- duration
- dataset count
- loader count
- roles present
- batch count

Use this to decide which run to inspect next.

## `runs show`

Inspect one run in detail:

```bash
pypyrus runs show <run_id>
```

This shows:

- run metadata
- datasets
- loaders
- transforms
- environment summary
- batch counts by role

It also shows sample-ID metadata per dataset:

- `sample_id_scheme`
- `sample_id_resolver`

## `compare`

Compare two runs:

```bash
pypyrus compare <run_a> <run_b>
```

The comparison is role-aware.

It reports:

- roles compared
- whether dataset identities match
- whether batch streams match
- a reason when they do not match
- first divergence when available

Use this when you want to know whether two runs saw the same data stream.

## `batches show`

Inspect one delivered batch:

```bash
pypyrus batches show <run_id> --step 12
```

Important semantics:

- `--step` means the run-global batch position
- internally this is `global_sequence`
- each run-global step identifies at most one batch in a run

The output includes:

- role
- loader ID
- dataset ID
- batch fingerprint
- sample IDs when stored

## `samples find`

Find whether a sample was used in a run.

### Sample ID lookup

```bash
pypyrus samples find <run_id> --sample-id index:3
```

Semantics:

- `--sample-id` is required
- lookup is run-wide across all datasets recorded in the run
- if the same sample ID appears under multiple datasets, the result reports all
  matching dataset IDs and roles

### Output

Sample lookup reports:

- whether the sample was found
- how many occurrences were found
- matching run-global steps
- matching roles
- matching loaders
- matching datasets
- first and last occurrence

## Typical Workflow

```bash
pypyrus runs list
pypyrus runs show <run_id>
pypyrus compare <run_a> <run_b>
pypyrus batches show <run_id> --step 12
pypyrus samples find <run_id> --sample-id record_id:ufo_42
```

Related pages:

- [SDK Quickstart](sdk-quickstart.md)
- [Sample Identity Contract](sample-identity-contract.md)
