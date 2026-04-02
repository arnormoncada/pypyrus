# PyPyrus Experiment Plan

This document defines the experiment set for the next phase of PyPyrus.

The goal is to produce evidence for two claims:

1. PyPyrus can detect and explain data-stream-level reproducibility matches and divergences.
2. PyPyrus introduces bounded overhead in runtime and storage.

This is a planning document for the thesis/demo phase. It defines what should be
measured, how it should be measured, and what outputs each experiment should
produce.

---

## 1. Core Claim Boundary

PyPyrus is not trying to prove:

- identical model weights
- identical floating-point execution
- identical internal augmentation randomness
- identical optimizer state trajectories

PyPyrus is trying to prove:

- the same dataset identity was used
- the same declared transform pipeline was used
- the same batch stream was delivered, or
- the first concrete point where the batch stream diverged

So the experiment suite should evaluate PyPyrus at the **data-stream boundary**.

---

## 2. Experiment Families

The experiment suite should have three families:

1. Reproducibility match experiments
2. Reproducibility divergence experiments
3. Overhead experiments

Each family should be small, repeatable, and easy to explain in a thesis or
demo.

---

## 3. Reproducibility Match Experiments

These experiments should show that PyPyrus reports a full match when the data
pipeline is effectively unchanged.

### 3.1 Fixed Seed, Fixed Config, Same Dataset

This is the baseline positive-control reproducibility experiment.

Detailed spec:

- [baseline-reproducibility-match.md](/Users/arnormoncada/Documents/DTU/V26/pypyrus/experiments/baseline-reproducibility-match.md)

Purpose:

- demonstrate the positive case
- show that PyPyrus can confirm a stable batch stream across repeated runs

High-level success condition:

- `compare` reports a full provenance match for two nominally identical runs

Primary script basis:

- `examples/plant_seedlings/train_mobilenetv3_small.py`

### 3.2 Same Run Shape Under Multi-Loader Setup

Purpose:

- show that PyPyrus handles multiple loaders/roles correctly even when they
  share dataset identity

Setup:

- one run with `train` and `val` loaders
- both derived from the same logical dataset identity where applicable

Measurements:

- loaders listed distinctly
- roles separated cleanly
- batch stream preserved at the run-global sequence level

Expected result:

- `runs show` and `batches show` make the run structure understandable

---

## 4. Reproducibility Divergence Experiments

These experiments should show that PyPyrus detects and explains concrete
differences at the dataset/transform/batch level.

### 4.1 Changed Shuffle Only

Purpose:

- demonstrate divergence caused only by sampling order

Setup:

- same dataset
- same transforms
- same batch size
- different seed affecting train-loader shuffle

Preferred script basis:

- plant seedlings example with `--seed`

Measurements:

- dataset identity still matches
- transform declaration still matches
- batch stream diverges
- first divergence step/sequence is reported

Expected result:

- `compare` reports dataset match but batch-stream mismatch

### 4.2 Changed Transform Only

Purpose:

- demonstrate divergence caused by a different declared preprocessing pipeline

Setup:

- same dataset
- same seed
- different transform pipeline or transform parameters

Measurements:

- transform declaration differs
- dataset identity may still match
- batch fingerprint stream may or may not diverge depending on what the current
  batch fingerprint includes

Important note:

- this experiment is mainly about transform declaration evidence, not claiming
  PyPyrus sees every tensor-level change

Expected result:

- `runs show` exposes the changed transform declaration
- `compare` should reflect the changed dataset/stream story as honestly as the
  current fingerprint model allows

### 4.3 Changed Dataset Contents Only

Purpose:

- demonstrate that PyPyrus detects dataset-version drift

Setup:

- same code
- same seed
- same transforms
- dataset contents modified between runs

Possible modifications:

- add/remove one file for the plant seedlings dataset
- alter one record for a structured-record dataset

Measurements:

- dataset fingerprint changes
- `compare` reports dataset identity mismatch

Expected result:

- dataset mismatch is surfaced clearly before or alongside batch mismatch

### 4.4 Changed Loader Role / Loader Structure

Purpose:

- show that loader topology changes are visible

Setup:

- one run with only `train`
- one run with `train` + `test`
- or one run where role assignment differs

Measurements:

- loader count
- role list
- compare behavior when roles differ

Expected result:

- CLI makes the structural difference obvious

### 4.5 Sample-Level Lookup Demonstration

Purpose:

- demonstrate the novel value of `samples find`

Setup:

- use the plant seedlings example for file-backed lookup
- optionally use the UFO example for direct `record_id:*` lookup

Measurements:

- direct `sample_id` lookup works
- file lookup works with dataset scoping
- first/last occurrence and matching steps are intelligible

Expected result:

- PyPyrus can answer “was this sample used in this run?”

---

## 5. Overhead Experiments

These experiments should quantify the cost of using PyPyrus.

Two dimensions matter:

1. runtime overhead
2. storage overhead

### 5.1 Runtime Overhead

Purpose:

- quantify how much training-loop or loader throughput slows down when PyPyrus
  is attached

Primary measurement:

- wall-clock training time

Secondary measurements:

- time per epoch
- batches per second
- samples per second

Baseline comparison:

- without PyPyrus
- with PyPyrus attached

Recommended setup:

- use the same training script with a simple flag to enable/disable attachment
- run multiple repetitions per condition
- use a fixed seed
- use the same hardware and environment

Workloads to include:

- a small image workload
  - plant seedlings fast path
- a structured-record/text workload
  - UFO fast path

Important variables to sweep:

- batch size
- `num_workers`
- dataset size or run duration

Recommended minimum sweep:

- `num_workers = 0`
- `num_workers = 2`
- small vs medium batch size

Reporting:

- absolute runtime
- percent slowdown relative to baseline

Acceptance framing:

- PyPyrus should show bounded and explainable overhead, not necessarily zero or
  negligible overhead under every condition

### 5.2 Storage Overhead

Purpose:

- quantify how much SQLite provenance data PyPyrus produces

Primary measurements:

- final DB size on disk
- DB size per run
- DB size per delivered batch

Secondary measurements:

- number of events stored
- compressed sample ID blob size distribution if useful

Baseline comparison:

- not “without PyPyrus” in a DB sense, but rather compare storage growth across
  workloads and run lengths

Recommended normalization metrics:

- bytes per batch
- bytes per sample
- bytes per epoch

Workloads to include:

- image/file collection example
- structured-record example

Suggested sweeps:

- short run
- medium run
- multi-loader run

Optional deeper measurement:

- break down SQLite file growth by event class if worth the effort

Acceptance framing:

- show that provenance size grows predictably and remains practical for the MVP

### 5.3 Combined Overhead Table

For the thesis/demo, produce one summary table with:

- workload
- condition
- runtime
- percent slowdown
- final DB size
- bytes per batch

This should become the main quantitative overhead artifact.

---

## 6. Methodology Rules

To keep the experiments defensible:

- use fixed seeds where the point is comparison, unless the experiment is
  explicitly about changing shuffle
- run each timing condition multiple times and report mean plus spread
- avoid mixing code changes with experiment-variable changes
- keep hardware constant within a comparison group
- keep database location and storage medium constant within a comparison group
- record the exact command used for each run

For timing experiments:

- use warm-up runs where needed
- do not compare a cold-download transformer run against a warm local baseline
- prefer the fast model path as the main overhead benchmark because it reduces
  dependency noise

---

## 7. Deliverables Per Experiment

Each experiment should produce:

1. a short experiment description
2. the command(s) used
3. the run IDs
4. relevant CLI outputs or captured summaries
5. a short conclusion

For overhead experiments, also produce:

6. a small result table
7. a plot if useful, but the table is the minimum artifact

---

## 8. Immediate Implementation Order

The experiment phase should be built in this order:

1. positive reproducibility match experiment
2. changed shuffle only
3. changed dataset contents only
4. sample lookup demonstration
5. runtime overhead benchmark
6. storage overhead benchmark
7. optional transform-only and loader-structure experiments

This order gives early thesis/demo value while keeping the first benchmarks
easy to reason about.

---

## 9. What Needs To Be Implemented Next

The next code work should support this experiment plan directly:

- one or more repeatable experiment scripts under `experiments/`
- a simple enable/disable switch for PyPyrus in benchmark scripts
- a benchmark harness or notebook-free script for runtime/storage summaries
- result collection conventions so runs and outputs are easy to compare later

The experiment scripts should be small and reproducible, not a generic
benchmarking framework.
