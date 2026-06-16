---
name: pypyrus-repo
description: Use when working in the PyPyrus repository on SDK, CLI, sample identity resolution, docs, examples, or tests. Covers the public API (`Run`, `attach`), sample ID contracts, CLI semantics, docs split, and repo-specific implementation conventions.
---

# PyPyrus Repo

Use this skill for work in the PyPyrus codebase.

## Public Surface

Treat the main public SDK as:

- `pypyrus.Run`
- `pypyrus.attach`

Current attach signature:

- `attach(loader, run, role=..., sample_id_resolver=None)`

Current CLI workflows:

- `runs list`
- `runs show`
- `compare`
- `batches show`
- `samples find`

## Sample ID Conventions

These are PyPyrus conventions, not PyTorch-wide standards.

Resolution order:

1. user `sample_id_resolver`
2. file collection
3. structured record
4. fallback `index:<i>`

Canonical contracts:

- file collection:
  - recommended: `.samples` + `.root`
  - compatibility alias: `.imgs`
- structured record:
  - recommended: `.records`
  - compatibility aliases: `.rows`, `.record_ids`, `.ids`
  - preferred key fields: `record_id`, `id`, `uuid`, `key`

Important semantics:

- file lookup is interpreted with dataset context
- effective exact identity is `dataset_id + sample_id`
- bare `--sample-id` may match multiple datasets in a run

## CLI Semantics To Preserve

- `batches show --step` means run-global `global_sequence`
- `samples find --sample-id ...` is the only sample lookup path
- `samples find --sample-id ...` searches across the whole run
- `runs show` should expose `sample_id_scheme` and `sample_id_resolver`

When changing reporting behavior, keep CLI help text and docs in sync.

## Docs Split

User-facing docs:

- `README.md`
- `docs/sdk-quickstart.md`
- `docs/sample-identity-contract.md`
- `docs/cli-usage.md`
- `docs/custom-dataset-integration.md`

Internal working docs:

- `docs/internal-working-notes.md`
- `docs/sample-id-resolver-architecture.md`

Do not mix internal planning material into user docs.

## Reference Experiments

Main experiment references currently cover:

- file collections: `experiments/plant_seedlings/`
- structured records: `experiments/ufo_sightings/`
- explicit resolver / tabular experiment: `experiments/forest_covertype/`

Examples should stay concise and demo-oriented. Prefer narrow, pragmatic model choices over heavyweight abstractions.

## Repo Conventions

- prefer narrow built-in contracts plus explicit escape hatches
- favor behavior-preserving cleanup over broad refactors
- centralize contract definitions when semantics matter
- update docs when public behavior changes

## Validation Notes

Prefer:

- `py_compile` for changed Python files
- targeted CLI smoke checks
- focused tests around touched behavior

Be aware of current environment caveats seen in this repo:

- some local `torch` runtime paths may fail with OpenMP shared-memory issues
- installed package copies may diverge from the repo checkout if the environment is stale
- schema files for SQLite must be packaged with `pypyrus.storage`

## Project plan
Use the `pypyrys_project_plan.pdf` file for a high-level overview of what we want this repo to be and why, including motivation and research questions. For detailed experiment planning and execution tracking, see `experiments/experiment-plan.md`.
