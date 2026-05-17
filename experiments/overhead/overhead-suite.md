# Runtime Overhead

This suite measures the runtime cost of PyPyrus instrumentation under repeated,
paired runs.

## Primary workload

- `plant`: file-based plant-seedlings ImageFolder benchmark

## Secondary workload

- `covtype`: record-based forest-covertype CSV benchmark

## Method

For each pair:

1. run the workload with instrumentation enabled
2. run the same workload without instrumentation
3. keep data, model, and loader settings fixed

Warm-up pairs are excluded from the reported summary.

## Local examples

From the repo root:

```bash
bash experiments/plant_seedlings/run_instrumentation_compare.sh
```

```bash
bash experiments/forest_covertype/run_instrumentation_compare.sh
```

## Cluster submission

Primary plant benchmark:

```bash
bsub < scripts/batch_run_overhead_gpu.sh
```

Covtype secondary benchmark:

```bash
WORKLOAD=covtype bsub < scripts/batch_run_overhead_gpu.sh
```

Useful overrides:

- `EPOCHS`
- `PAIRS`
- `WARMUP_PAIRS`
- `BATCH_SIZE`
- `NUM_WORKERS`
- `TIMING_FILE`
