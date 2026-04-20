# Distributed Training Support (DDP) - Design Notes

This document summarizes what must change in PyPyrus to support distributed
training robustly, and why those changes are needed.

## Goal

Keep PyPyrus conceptually distribution-agnostic:

- single-process training = one data stream
- distributed training = multiple data streams under one logical run

The system should still answer the same core questions:

1. What data was used?
2. In what order was it delivered?
3. Where do two executions diverge?

## Current State and Gaps

### 1. Single writer timeline assumption

Current `BatchDeliveredEvent` includes `global_sequence` that is unique per run.
This implies one total order for all batches in a run.

Why this is a problem:

- DDP naturally produces parallel per-rank streams.
- There is no canonical global batch order across ranks unless we define one.

### 2. Missing distributed execution identity

Current events do not encode rank/process identity in first-class fields.

Why this is a problem:

- We cannot reliably answer "which rank saw this batch".
- We cannot compare per-rank streams across runs.

### 3. Shared SQLite writer contention

Current SQLite usage keeps transactions open until explicit `flush()`.

Why this is a problem:

- Multiple DDP processes writing to one DB file will contend for the write lock.
- Long transactions increase lock duration and timeout risk.

### 4. Run lifecycle is process-centric

`RunStart` / `RunEnd` represent one process-local lifecycle.

Why this is a problem:

- DDP needs either:
  - one logical run containing many process streams, or
  - one run per rank with a grouping relation.

## Required Changes

## A. Event model

Add distributed context to relevant events.

Recommended fields:

- `run_group_id` (logical training execution id)
- `rank` (global rank)
- `local_rank`
- `world_size`
- `node_rank` (optional)
- `stream_id` (stable identity for one ordered batch stream)
- `stream_step` (monotonic index within `stream_id`)

Where to add:

- `RunStartEvent` / `RunEndEvent`: distributed context (`rank`, `world_size`, etc.)
- `LoaderRegisteredEvent`: bind loader to `stream_id`
- `BatchDeliveredEvent`: include `stream_id` + `stream_step` and rank metadata

Why:

- makes per-rank provenance explicit
- removes dependence on one global timeline
- enables deterministic comparisons at stream level

## B. Database schema

Add fields/tables to represent logical runs and streams.

Suggested additions:

1. `runs`:
- add `run_group_id`
- add `rank`, `local_rank`, `world_size`, `node_rank`

2. `loaders`:
- add `stream_id`
- optional unique constraint: `(run_id, stream_id)`

3. `batch_delivered`:
- add `stream_id`, `stream_step`
- move primary uniqueness from run-global sequence to stream-local ordering
- keep `global_sequence` optional and non-authoritative

Recommended constraints:

- `UNIQUE(stream_id, stream_step)`
- optional `UNIQUE(run_id, global_sequence)` only if retained for presentation

Why:

- stream order is the true invariant in distributed execution
- uniqueness reflects actual semantics

## C. Storage and write strategy

Support a safe default for DDP writes.

Preferred MVP strategy:

- one DB file per rank/process
- merge/query layer provides logical run view across rank DBs

Alternative (higher complexity):

- single shared DB with:
  - WAL mode
  - busy timeout
  - shorter transactions / periodic commits
  - possibly centralized writer process

Why:

- per-rank DB avoids cross-process writer lock contention
- easiest reliable path for MVP

## D. Run identity strategy

Define one logical training id and one process-local run id.

Recommended:

- `run_group_id`: shared across all ranks in one distributed execution
- `run_id`: unique per process/rank

Why:

- preserves clean lifecycle semantics per process
- allows grouping for distributed analysis/reporting

## E. Query and comparison semantics

Update reporting APIs to compare streams, not assumed single sequence.

Changes:

- compare by `(role, stream_id, stream_step)` or `(role, rank, stream_step)`
- report divergence per stream and per role
- aggregate to logical run-level summaries

Why:

- aligns analysis with DDP reality
- avoids false mismatches caused by arbitrary interleaving

## F. CLI changes

Extend command output to display distributed context.

Minimum updates:

- `runs list`: show `run_group_id`, rank/world size
- `runs show`: show per-rank streams and counts
- `compare`: role-aware and stream-aware divergence output
- `batches show`: optional filters `--rank` and `--stream-id`

Why:

- preserves usability while adding distributed clarity

## G. Migrations and compatibility

Add schema migration(s) and compatibility behavior.

Guidelines:

- old single-process runs remain readable
- missing distributed fields are interpreted as single-stream defaults
- avoid breaking existing CLI JSON contracts unless versioned

Why:

- preserves existing experiments and tests

## Implementation Plan (Phased)

### Phase 1 (MVP-safe distributed support)

1. Add distributed metadata fields to events and schema.
2. Introduce `stream_id` + `stream_step`.
3. Default to per-rank DB path strategy.
4. Add stream-aware query helpers.
5. Add integration tests for multi-process simulated writes.

Outcome:

- robust provenance capture for DDP without shared-writer fragility.

### Phase 2 (Unified analysis and UX)

1. Add run-group aggregation in reporting layer.
2. Upgrade CLI output for stream/rank context.
3. Add compare diagnostics across streams.

Outcome:

- distributed runs become first-class in user workflow.

### Phase 3 (Optional shared DB mode)

1. Add explicit shared DB mode behind config.
2. Enable WAL + timeout + periodic commit tuning.
3. Consider centralized writer for high-throughput settings.

Outcome:

- optional advanced deployment path, not required for correctness.

## Test Additions Needed

1. Rank metadata persistence test.
2. Stream uniqueness test (`stream_id`, `stream_step`).
3. Distributed compare regression test (first divergence per stream).
4. Backward compatibility test for legacy runs.
5. Per-rank DB aggregation test (logical run view).

## Open Design Decisions

1. Do we define logical step alignment at dataloader boundary only,
   or also add strict training-step boundary instrumentation?
2. Should `stream_id` be deterministic from `(run_id, role, rank, loader_idx)`
   or random UUID stored in loader registration?
3. Is shared-DB mode needed in MVP, or can it be postponed?

## Recommended MVP Decision

- Treat distributed training as multiple ordered streams under one logical run.
- Implement per-rank DB capture first.
- Make compare/query stream-aware.
- Defer shared-DB writer complexity until needed.

This gives a defensible and reliable distributed provenance story while keeping
implementation risk manageable.
