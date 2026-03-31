# PyPyrus Sample ID Resolver Architecture

This note is the cleaned-up MVP architecture successor to
`docs/pypyrus-sample-id-resolver-brainstorm.md`.

The brainstorm note should remain as raw ideation. This document is the
implementation-facing version.

---

## Core framing

PyPyrus should not reason in terms of:

* image datasets
* text datasets
* tabular datasets

PyPyrus should reason in terms of:

* how individual samples are stably addressed at the source
* how that source address can be normalized into a stored `sample_id`

The design goal is:

> produce the best available normalized sample reference, falling back to
> dataset-local positional identity when necessary.

In plain language:

* a `sample_id` is the canonical identifier stored in provenance
* resolver logic exists only to derive or recover that identifier
* `samples find` should query by `sample_id`
* filepath lookup is a convenience resolver layered on top of `sample_id`,
  not a separate provenance model

That means PyPyrus should abstract over **source addressability pattern**, not
data modality.

---

## MVP resolver families

The current MVP should explicitly consider these 4 resolver families.

### 1. File collection

One file corresponds to one sample under a dataset root.

Examples:

* `ImageFolder`
* text corpora with one file per document
* audio clips in folders
* PDFs or JSON files stored one-per-sample

Preferred identity:

* relative filepath under the effective dataset root

Example mappings:

* `ImageFolder` -> `filepath:class_a/img_001.png`
* text file dataset -> `filepath:docs/report_12.txt`

CLI implication:

* filepath lookup should be first-class in the MVP for this family

This is the first implementation target for `samples find`.

### 2. Structured record source

Samples correspond to rows or records in a structured source.

Examples:

* CSV
* JSONL
* Parquet-like row/record sources
* table-like datasets exposed through a file or file set

Preferred identity:

* explicit record ID if available
* otherwise row index

Example mappings:

* CSV-like source -> `record_id:customer_84291`
* CSV fallback -> `row:183`
* JSONL fallback -> `row:91`

This family is architecturally in-scope for the MVP, but it is likely not the
first CLI implementation target.

### 3. Framework / logical dataset

The dataset exposes a stable logical access pattern even if the underlying
storage is abstracted away.

Examples:

* torchvision datasets
* framework-provided train/test splits
* datasets with built-in keys or stable logical indices

Preferred identity:

* built-in key if available
* otherwise split + index
* otherwise dataset-local index

Example mappings:

* MNIST-like framework dataset -> `logical:train#1234`
* keyed logical dataset -> `logical:validation#doc_0042`

For the MVP, MNIST-like usage belongs here rather than opening a separate
container/binary resolver family.

### 4. Fallback custom / in-memory

No source-transparent locator is available.

Examples:

* `TensorDataset`
* list-backed custom datasets
* synthetic datasets
* precomputed objects stored only in memory

Preferred identity:

* dataset-local index

Example mapping:

* `TensorDataset` -> `index:123`

This fallback must always work. It is what keeps PyPyrus usable even when no
stronger source reference exists.

### Deferred on purpose

The MVP should stop here.

Keyed-container datasets, database-backed datasets, and other more specialized
source families are intentionally deferred to avoid overdesign during the
current hardening phase.

---

## Identity quality and normalization

PyPyrus should treat sample references as having a simple quality ranking.

### 1. Strong source locator

Best case.

Examples:

* relative filepath
* explicit record ID
* built-in dataset key

These are the most human-meaningful and easiest to query later.

### 2. Structural locator

Still useful, but more positional.

Examples:

* row index
* split + index

These are acceptable for the MVP, but they are more brittle if the source
changes.

### 3. Fallback positional identity

Always available.

Examples:

* dataset-local index

This is the universal fallback and should never be blocked by the absence of a
stronger resolver.

### Normalized output schemes

For the current MVP, the architecture should standardize on these normalized
reference schemes:

* `filepath:<relative-path>`
* `record_id:<value>`
* `row:<value>`
* `logical:<split>#<index-or-key>`
* `index:<value>`

This is intentionally simple. The goal is not to design a large type system,
but to keep the stored `sample_id` interpretable, stable, and queryable.

---

## MVP architecture decisions

The following decisions should be treated as locked for the MVP.

### Core decisions

* the stored provenance field remains `sample_id`
* current default extraction remains index-based unless a better resolver is
  available
* resolver logic should be best-effort and strategy-based
* the MVP should use a small sequential resolution pipeline, not a generalized
  registry or plugin framework

### Attach-time identity extraction

The intended resolution order at attach time is:

1. user-provided resolver override, if present
2. file-collection resolver
3. framework/logical resolver
4. structured-record resolver when the dataset shape makes this straightforward
5. fallback `index:<i>`

The important invariant is:

* PyPyrus stores one normalized `sample_id`
* later CLI and reporting code should query that same identity, not invent a
  second representation

### CLI-time query resolution

The intended MVP query shape for `samples find` is:

* `--sample-id` is the generic path
* `--file` + `--dataset-path` is the first convenience path

For filepath lookup:

* the file query must resolve to the same normalized `sample_id` scheme used at
  attach time
* filepath lookup should require dataset fingerprint match before the result is
  trusted

This means filepath lookup is a resolver convenience, not a separate provenance
feature.

---

## Scope boundaries and wrapper handling

Wrappers matter, but the MVP should stay conservative.

For:

* `Subset`
* `ConcatDataset`
* filtered views
* remapped or wrapped datasets

the resolver should operate against the **effective attached dataset**.

Desired behavior:

* preserve parent/source references when they are easy to recover

MVP fallback:

* if provenance-preserving reconstruction is hard, fall back to effective
  dataset-local identity rather than invent unstable semantics

The current MVP boundary should be stated explicitly:

> the current implementation target is reliable file-based lookup plus generic
> sample-id lookup.

That means:

* file collection is the first concrete implementation target
* structured-record and framework/logical families are part of the architecture
  and future-compatible shape
* not every family needs immediate CLI implementation during this hardening
  phase

---

## MVP acceptance criteria

The first usable version of this layer should satisfy the following:

* direct `sample_id` lookup works regardless of dataset family
* filepath lookup is supported only for file-collection datasets in the first
  CLI version
* resolver output is deterministic for the same dataset and sample
* dataset fingerprint match is required before trusting filepath-based reverse
  lookup
* when no source-level identity is available, the system still produces a valid
  fallback `sample_id`
* wrapper datasets do not block provenance capture; they fall back safely if
  source recovery is not possible

---

## Practical MVP stance

The resolver should not be a giant universal dataset parser.

For the current thesis/MVP scope, the right approach is:

* normalize sample identity into one stored `sample_id`
* prefer strong source references when they exist
* fall back safely when they do not
* implement `samples find` around direct `sample_id` lookup first
* add filepath resolution as the first convenience layer for file collections

That is enough to make the sample-lookup feature real without going overboard.
