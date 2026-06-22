# Thesis Experiment Plan

## Scope

The evaluation should stay tightly aligned with the thesis research questions.
The goal is not to benchmark every supported dataset shape or every PyPyrus
feature independently. The goal is to produce a small set of experiments that
directly support the core claims:

1. PyPyrus can capture and compare training-time data provenance at the
   data-stream boundary.
2. The captured provenance is useful for traceability and divergence diagnosis.
3. The instrumentation introduces bounded overhead and practical storage cost.

The evaluation should therefore focus on a few controlled studies with clear
positive/negative controls, rather than many small feature demonstrations.

---

## Workloads

### Primary workload: Plant seedlings

Use the plant seedlings ImageFolder benchmark as the primary workload for the
reproducibility and runtime-overhead results.

Why:
- lowest execution risk
- strongest current script support
- clear file-based sample identity story
- best fit for repeated paired runs on HPC

### Secondary workload: Forest Covertype

Use forest covertype as the structured-record workload and as the larger-scale
dataset for storage and scaling evidence.

Why:
- provides a record-based sample identity case
- contrasts with the smaller file-based workload
- helps show that the design is not limited to one dataset shape

The dataset-type distinction should be treated as supporting context within the
main experiments, not as a separate experiment family of its own.

---

## Core Evaluation Questions

### RQ1 and RQ2: Provenance capture, reproducibility, and divergence detection

Show that PyPyrus can:
- confirm a positive reproducibility match when conditions are held fixed
- detect and localize divergence when one controlled variable changes

### RQ3: Runtime overhead

Show that PyPyrus adds measurable but bounded execution overhead under repeated,
paired runs.

### RQ4: Traceability support

Show that the generated provenance records expose the information needed to:
- identify which dataset instance was used
- inspect which samples appeared in which delivered batches
- trace a sample occurrence back to run, role, and step

This should be framed as support for traceability objectives, not as complete
regulatory compliance.

---

## Main Experiments

## A. Reproducibility and Divergence Detection

This is the main correctness section of the evaluation.

### A1. Baseline match (positive control)

Goal:
Show that two nominally identical runs produce a full provenance match.

Workload:
- plant seedlings

Protocol:
1. Run the same configuration twice with fixed seed, fixed data, and fixed
   loader settings.
2. Compare the two runs using the standard comparison workflow.
3. Record dataset identity match, batch-stream match, and first-divergence
   status.

Expected result:
- dataset identity matches
- batch stream matches
- no reported divergence

### A2. Shuffle divergence

Goal:
Show that changing only the seed changes the delivered batch stream while
leaving dataset identity unchanged.

Workload:
- plant seedlings

Protocol:
1. Run configuration A with seed `S1`.
2. Run configuration B with seed `S2`.
3. Keep data, transforms, and loader settings otherwise fixed.
4. Compare the two runs.

Expected result:
- dataset identity matches
- batch stream mismatch is reported
- first divergence step is surfaced

### A3. Dataset-content divergence

Goal:
Show that changing the underlying data changes dataset identity and is surfaced
by PyPyrus.

Workload:
- plant seedlings or forest covertype, depending on which controlled data
  variant is easiest to construct reproducibly

Protocol:
1. Create a controlled modified dataset variant.
2. Run baseline and modified configurations.
3. Compare the two runs and inspect run metadata.

Expected result:
- dataset fingerprint mismatch is reported
- the mismatch is attributable to dataset identity rather than only batch order

---

## B. Runtime Overhead

This is the main performance section of the evaluation.

### B1. Primary paired benchmark

Goal:
Estimate the runtime overhead introduced by instrumentation under repeated
paired trials.

Primary workload:
- plant seedlings

Protocol:
1. Alternate runs with and without instrumentation.
2. Exclude warm-up pairs.
3. Keep job settings, data, and training configuration fixed.
4. Use sufficiently long runs so startup costs do not dominate.

Primary metrics:
- mean paired delta in seconds
- median paired delta in seconds
- mean paired percent overhead
- 95% confidence interval for mean delta and mean percent
- variability summary (stddev or IQR)

Interpretation:
- paired results are the primary estimate
- unpaired averages should be secondary context only

### B2. Sensitivity / scaling check

Goal:
Assess whether the overhead conclusion remains directionally stable under a
larger record-based workload.

Secondary workload:
- forest covertype

Purpose:
- show that the overhead story is not tied only to a small file-based dataset
- provide a small scaling contrast between a smaller and larger workload

Optional parameter variation:
- batch size


---

## C. Traceability Demonstration

This should be a short applied-value section, not a large standalone benchmark.

Goal:
Demonstrate that the recorded provenance can be used to inspect sample- and
batch-level lineage in practice.

Use:
- plant seedlings for a file-based example
- forest covertype for a record-based example

Demonstrate:
1. run overview
2. batch inspection at a chosen step
3. sample lookup for one known sample in each workload

Outputs:
- one concise file-based trace example
- one concise record-based trace example

This section should illustrate practical utility, not report a large lookup
success-rate study unless such a metric becomes especially easy to collect.

---

## D. Storage Footprint

This is a supportive evaluation component.

Goal:
Characterize the practical storage cost of recorded provenance.

Use:
- plant seedlings for the smaller workload
- forest covertype for the larger workload

Measurements:
- database size after each run
- approximate bytes per batch

Report this as a simple scaling-oriented result. It does not need a large
independent subsection unless the findings are especially strong.

---

## Final Evaluation Set

The minimum thesis-ready experiment set should be:

1. Baseline match
2. Shuffle divergence
3. Dataset-content divergence
4. Primary paired overhead benchmark
5. Short traceability demonstration using one file-based and one record-based
   workload
6. Short storage-footprint characterization


---