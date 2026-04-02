# Baseline Reproducibility Match Experiment

This document defines the baseline positive-case reproducibility experiment for
PyPyrus.

It should be read alongside [experiment-plan.md](/Users/arnormoncada/Documents/DTU/V26/pypyrus/experiments/experiment-plan.md).

## Purpose

Demonstrate that when the experiment setup is held fixed, PyPyrus reports a
full provenance match at the data-stream boundary.

This experiment is the baseline positive control for the broader
reproducibility suite.

It is not trying to prove:

- identical model weights
- identical floating-point execution
- identical optimizer trajectories

It is trying to prove:

- the same dataset identity was used
- the same declared transform pipeline was used
- the same batch stream was delivered
- PyPyrus can confirm that no provenance-level divergence occurred

## Core Question

If the same training script is run twice with the same dataset, the same
declared preprocessing, the same loader configuration, and the same seed, does
PyPyrus report that the two runs match?

## Script Basis

Primary basis:

- `examples/plant_seedlings/train_mobilenetv3_small.py`
- runner: `experiments/run_baseline_reproducibility_match.py`

Why this script:

- it is already wired into PyPyrus
- it uses a file-collection dataset that PyPyrus resolves automatically
- it has a simple seed-driven configuration surface
- it is easy to explain in a thesis/demo

The UFO example can be used later as a structured-record analogue, but it
should not replace the first baseline experiment.

## Controlled Variables

The two runs must match on all of the following:

- code revision
- dataset contents
- dataset path
- transforms
- model architecture
- optimizer configuration
- number of epochs
- batch size
- number of workers
- shuffle setting
- random seed
- PyPyrus version/config

These runs should be treated as nominally identical at the data-stream level.

## Independent Variable

There is no intended experimental change between the two runs.

The only difference should be that the script is executed twice, producing two
separate run IDs.

## Procedure

### 1. Prepare the environment

- use one fixed repository revision
- use one fixed Python environment
- ensure the plant seedlings dataset is unchanged between runs
- clear or record the output database path used by PyPyrus

### 2. Execute the experiment runner

Run the dedicated experiment script with a fixed configuration, for example:

```bash
python experiments/run_baseline_reproducibility_match.py \
  --data-root examples/plant_seedlings/data/split \
  --epochs 3 \
  --seed 42 \
  --reset-db
```

Record:

- run IDs
- CLI command used
- relevant config values
- output database path
- JSON evidence path

The runner executes the same training configuration twice and writes both runs
to the same PyPyrus database.

### 3. Inspect both runs

Capture:

```bash
pypyrus runs show <run_a>
pypyrus runs show <run_b>
```

The inspection should confirm that the runs expose the same high-level data
pipeline story.

### 4. Compare the runs

Capture:

```bash
pypyrus compare <run_a> <run_b>
```

If role-targeted compare output exists or is useful later, it can be captured
as supplementary evidence, but the core baseline should use the simplest
compare path first.

### 5. Spot-check a few batch and sample views

Capture a few representative steps:

```bash
pypyrus batches show <run_a> --step <n>
pypyrus batches show <run_b> --step <n>
```

Also capture one or two sample lookups using a known file path:

```bash
pypyrus samples find <run_a> \
  --file <class>/<filename> \
  --dataset-path examples/plant_seedlings/data/split/train

pypyrus samples find <run_b> \
  --file <class>/<filename> \
  --dataset-path examples/plant_seedlings/data/split/train
```

These are supporting checks. The primary evidence is still the run-level
compare result.

## Measurements To Capture

### Required

- run IDs
- dataset fingerprints
- dataset roles
- sample ID scheme/resolver metadata
- total batch counts
- compare result
- first divergence status

### Supporting

- selected `batches show` outputs at the same run-global step
- selected `samples find` outputs for the same file-backed sample

## Expected Result

PyPyrus should report a full provenance match between the two runs.

Concretely, the evidence should show:

- matching dataset identity
- matching declared transform story
- matching loader role structure
- matching batch counts
- no reported divergence in the captured batch stream

## Success Criteria

The experiment is considered successful if:

1. `pypyrus compare <run_a> <run_b>` reports a match or no meaningful
   provenance-level differences
2. `runs show` for both runs presents the same dataset and loader story
3. spot-checked batch/sample views are consistent with the compare result

## Interpretation

If this experiment succeeds, it supports the claim that PyPyrus can confirm a
reproducibility match at the data-stream boundary for a controlled repeated
run.

This should be interpreted narrowly:

- it validates the provenance comparison story
- it does not, by itself, prove bitwise-identical training behavior

## Deliverables

The baseline experiment should produce:

- the two run IDs
- the exact commands used
- `runs show` output for both runs
- `compare` output for the run pair
- one or two supporting `batches show` captures
- one or two supporting `samples find` captures
- a short written interpretation of the result

## Follow-On Relation

This experiment is the baseline for the later divergence experiments.

The next experiments should deliberately violate one controlled variable at a
time, for example:

- shuffle only
- transforms only
- dataset contents only
- loader structure only
