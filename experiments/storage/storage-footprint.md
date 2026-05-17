# Storage Footprint

This suite runs instrumented workload(s) and summarizes the resulting PyPyrus
SQLite database footprint.

## Workloads

- `plant`: plant-seedlings ImageFolder benchmark
- `covtype`: forest-covertype CSV benchmark
- `all`: run both sequentially

## Reported values

For each generated database, the suite records:

- database size in bytes and MiB
- run ID and run name
- stored event count
- batch count
- dataset registration count
- loader count
- transform count
- environment snapshot count
- approximate bytes per batch
- approximate bytes per event

## Local example

From the repo root:

```bash
python experiments/storage/run_storage_footprint.py \
  --workload all \
  --reset-output
```

## Cluster submission

```bash
bsub < scripts/batch_run_storage_footprint_gpu.sh
```

Useful overrides:

- `WORKLOAD`
- `OUTPUT_ROOT`
- `PLANT_EPOCHS`
- `COVTYPE_EPOCHS`
- `PLANT_BATCH_SIZE`
- `COVTYPE_BATCH_SIZE`
- `SEED`
