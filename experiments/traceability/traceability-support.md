# Traceability Support

This suite runs one instrumented workload at a time and extracts three compact
evidence artifacts from the resulting PyPyrus database:

1. run overview
2. one selected delivered batch
3. one sample lookup

The goal is to demonstrate that the recorded provenance supports practical
inspection tasks without relying on raw database dumps.

## Workloads

- `plant`: file-based ImageFolder example
- `covtype`: record-based CSV example
- `all`: run both sequentially

## Output artifacts

For each workload, the suite writes:

- `traceability_summary.json`
- `run_overview.txt`
- `selected_batch.txt`
- `sample_lookup.txt`

The selected sample is the first sample ID from the first recorded `train`
batch, so the lookup is guaranteed to correspond to an observed delivered
sample.

## Local example

From the repo root:

```bash
python experiments/traceability/run_traceability_support.py \
  --workload all \
  --reset-output
```

## Cluster submission

```bash
bsub < scripts/batch_run_traceability_gpu.sh
```

Useful overrides:

- `WORKLOAD`
- `OUTPUT_ROOT`
- `PLANT_EPOCHS`
- `COVTYPE_EPOCHS`
- `PLANT_BATCH_SIZE`
- `COVTYPE_BATCH_SIZE`
- `SEED`
