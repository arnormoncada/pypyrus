# Thesis Experiment Plan

## Positioning

Primary reported benchmark: **Plant seedlings (MobileNetV3)**.

Rationale:
- Strongest current support in repository scripts and timing workflow
- Lowest execution risk for reproducible paired trials on HPC
- Clearest narrative for a focused experiment/results chapter

Secondary benchmark: **UFO sightings** as a transferability/traceability validation, not co-primary.

Larger dataset: include only as a **scaling subsection** if compute/time permits.

---

## High-Level Experiment Plan

### 1. Reproducibility & Divergence Detection (Core Claim)
Goal: show PyPyrus correctly identifies matching runs and pinpoints divergence causes.

Planned studies:
1. Baseline match (positive control): same config/seed/data -> full match expected
2. Shuffle-only divergence: seed changed -> dataset identity match, batch stream divergence
3. Transform-only divergence: transform pipeline changed -> transform mismatch and likely batch divergence
4. Dataset-content divergence: data changed -> dataset fingerprint mismatch
5. Loader topology divergence: train-only vs train+test roles -> topology mismatch
6. Sample-level lookup demonstration: query traceability by sample/file

Primary outputs:
- Match/mismatch status per component (dataset identity, transform declarations, batch stream)
- First divergence step index
- Qualitative explanation of divergence source

---

### 2. Overhead & Performance (Core Claim)
Goal: quantify runtime overhead introduced by instrumentation.

Design:
1. Paired alternating runs (`with`, then `without`) under identical conditions
2. Warm-up pairs excluded from measured results
3. Longer runs (increase epochs so runtime is less startup-dominated)
4. Fixed cluster/job settings per condition

Primary outputs:
- Mean/median paired delta in seconds
- Mean paired percent overhead
- 95% CI for mean paired delta and mean paired percent
- Variability indicators (stddev/IQR)

---

### 3. Traceability Quality (Applied Value Claim)
Goal: demonstrate practical provenance utility for users.

Planned studies:
1. Sample lookup success for file-based data (plant seedlings)
2. Sample lookup success for record-based data (UFO)
3. Role/step-level inspection of delivered batches

Primary outputs:
- Lookup success rate
- Example provenance traces from run -> role -> step -> sample
- Query latency (optional if feasible)

---

### 4. Storage Footprint (Supportive Claim)
Goal: characterize storage growth and practical cost.

Planned studies:
1. DB size vs number of recorded batches/runs
2. Bytes per batch / bytes per sample trend (approximate)

Primary outputs:
- Table of run count, event count, DB size
- Simple scaling trendline

---

## Experiment/Testing Plan Per Experiment Type

## A) Reproducibility & Divergence

### A1. Baseline Match
Hypothesis: fixed seed/config/data yields full match.

Protocol:
1. Execute two independent runs with identical settings
2. Compare with CLI/API comparison
3. Record component-level match flags

Acceptance:
- Full match true
- No divergence step

### A2. Shuffle Divergence
Hypothesis: changing only seed changes order, not dataset identity.

Protocol:
1. Run A with seed S1, run B with seed S2
2. Keep transforms/data fixed
3. Compare runs

Acceptance:
- Dataset identity match true
- Batch stream match false
- First divergence step reported

### A3. Transform Divergence
Hypothesis: transform change is detected and explains mismatch.

Protocol:
1. Run A baseline transforms
2. Run B modified transforms
3. Compare transform declarations and batch stream

Acceptance:
- Transform mismatch true
- Divergence detected and explainable

### A4. Dataset Content Divergence
Hypothesis: changing dataset contents changes dataset fingerprint.

Protocol:
1. Create controlled modified dataset variant
2. Run baseline vs modified
3. Compare dataset identity fields

Acceptance:
- Dataset fingerprint mismatch
- Comparison surfaces mismatch cause

### A5. Loader Topology Divergence
Hypothesis: role/topology changes are surfaced.

Protocol:
1. Run A with train loader only
2. Run B with train+test loaders
3. Compare role-level records

Acceptance:
- Role/topology mismatch visible in report

### A6. Sample Traceability
Hypothesis: sample-level provenance lookup is operational and useful.

Protocol:
1. Query sample IDs / file paths from delivered batches
2. Validate mapping to runs/steps/roles

Acceptance:
- High lookup success
- Human-readable lineage examples

---

## B) Overhead & Performance

### B1. Main Runtime Benchmark (Primary)
Workload: plant seedlings.

Protocol:
1. Use paired alternating measured runs
2. Exclude warm-up pairs
3. Run 20-30 measured pairs (budget allowing)
4. Use longer epochs (target multi-minute runs)

Metrics:
- Paired deltas: `delta_i = with_i - without_i`
- Mean/median delta (s)
- Mean percent delta
- 95% bootstrap CI for mean delta and mean %

Acceptance:
- Bounded overhead estimate with uncertainty interval

### B2. Sensitivity Checks
Protocol:
1. Repeat with different `num_workers` values
2. Optional batch-size variants

Metrics:
- Overhead change by setting

Acceptance:
- Stable directionality of overhead across settings

---

## C) Traceability Quality

### C1. File-Based Traceability
Workload: plant seedlings.

Protocol:
1. Select random delivered samples
2. Query provenance via CLI/API

Metrics:
- Success rate
- Completeness of returned context

### C2. Record-Based Traceability
Workload: UFO sightings.

Protocol:
1. Select random record IDs
2. Resolve run/step provenance

Metrics:
- Success rate
- Consistency with expected labels/records

---

## D) Storage Footprint

### D1. Growth Characterization
Protocol:
1. Run short, medium, long workloads
2. Record DB size and event counts after each

Metrics:
- DB size growth per run
- Approx bytes per event/batch

Acceptance:
- Practical storage budget statement

---

## Cluster Settings & Controls (Applied Across Experiments)

1. Same partition/QoS and resource request across conditions
2. Fixed seeds for controlled experiments
3. Same dataset snapshot and code revision
4. Record node/job metadata with results
5. Avoid mixing markedly different load windows when possible
6. Keep dataloader/training hyperparameters fixed unless explicitly varied

---

## Statistical Reporting Template

Report for each benchmark condition:
1. `n` pairs (after warm-up exclusion)
2. Mean and median paired delta (seconds)
3. Mean paired percent overhead
4. 95% CI of mean delta and mean percent (bootstrap)
5. Variability (stddev/IQR)
6. Outlier note and sensitivity check (if any)

Interpretation guidance:
- Use paired statistics as primary estimate
- Use unpaired means as secondary context only

---

## Proposed Chapter Structure (Experiment/Results)

1. Evaluation goals and hypotheses
2. Experimental setup (hardware/software/data/protocol)
3. Reproducibility and divergence results
4. Overhead and sensitivity results
5. Traceability demonstration results
6. Storage footprint results
7. Threats to validity
8. Summary of findings and implications

---

## Execution Priority

1. Primary: B1 + A1/A2/A3 (core claims)
2. Secondary: A4/A5 + C1/C2 (robustness and utility)
3. Optional: D1 + larger-dataset scaling subsection
