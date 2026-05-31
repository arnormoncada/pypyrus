# Reproducibility Suite (`A1–A3`)

This document describes the main reproducibility and divergence-detection
experiment setup for PyPyrus.

The suite focuses on three controlled cases over the plant-seedlings workload:

- `A1`: baseline positive-control match
- `A2`: shuffle divergence
- `A3`: dataset-content divergence

`A4` transform-change divergence is intentionally excluded from this suite.

## Purpose

The suite is designed to answer the core correctness questions of the thesis:

- can PyPyrus confirm a provenance-level match when the training setup is held
  fixed?
- can PyPyrus detect divergence when one controlled variable changes?

The suite does not try to prove:

- identical model weights
- identical floating-point execution
- identical optimizer trajectories

It instead evaluates the data-stream provenance story:

- dataset identity
- loader role structure
- delivered batch stream
- sample-level traceability checks

## Cases

### `A1` Baseline match

Two runs are executed with the same configuration and the same seed.

Expected result:
- dataset identity match
- batch-stream match
- no divergence step

Run names:
- `a1-match-a`
- `a1-match-b`

### `A2` Shuffle divergence

Two runs are executed with the same configuration but different seeds.

Expected result:
- dataset identity match
- batch-stream mismatch
- first divergence step reported

Run names:
- `a2-shuffle-s<seed-a>`
- `a2-shuffle-s<seed-b>`

### `A3` Dataset-content divergence

One run uses the original plant dataset and one run uses a deterministic
derived copy in which the lexicographically first training image is flipped
horizontally and overwritten at the same relative path.

The modified run uses the derived `train/` split together with the original
`test/` split so the divergence is isolated to the changed training data.

Expected result:
- train dataset identity mismatch
- test dataset identity match
- role structure unchanged
- batch stream may still match

Run names:
- `a3-data-base`
- `a3-data-mod`

## Local usage

Run the full suite:

```bash
python experiments/repro/run_repro_suite.py \
  --case all \
  --data-root experiments/plant_seedlings/data/split \
  --epochs 3 \
  --reset-output
```

Run a single case:

```bash
python experiments/repro/run_repro_suite.py \
  --case a2 \
  --data-root experiments/plant_seedlings/data/split \
  --seed-a 42 \
  --seed-b 99 \
  --epochs 3 \
  --reset-output
```

## Cluster usage

The default cluster wrapper is:

```bash
bsub < scripts/batch_run_repro_suite_gpu.sh
```

Override case or seeds at submission time:

```bash
CASE=a2 SEED_A=42 SEED_B=99 bsub < scripts/batch_run_repro_suite_gpu.sh
```

## Default cluster assumptions

The BSUB wrapper standardizes on:

- queue: `gpua100`
- environment: `pypyrus-v100`
- single node
- one GPU
- fixed CPU and memory request

All `A1–A3` runs should use the same cluster settings unless the experiment is
explicitly studying a different variable.

## Outputs

By default, the suite writes per-case artifacts under:

```text
experiments/results/repro_suite/
```

Each case produces:

- one SQLite database
- one JSON evidence bundle
- printed compare output with run IDs and run names

When `--case all` is used, a suite summary JSON is also written.

## Interpretation

Interpret the results narrowly:

- `A1` validates the positive-control provenance match
- `A2` validates batch-stream divergence detection
- `A3` validates dataset-identity divergence detection

These experiments support reproducibility and traceability claims at the
data-stream boundary. They do not, by themselves, prove end-to-end numerical
training equivalence.
