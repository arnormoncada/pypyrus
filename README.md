# PyPyrus A Data Provenance Layer for Transparent and Reproducible Machine Learning Systems

### Project structure:
```
pypyrus                
├─ docs                
├─ examples            
├─ experiments         
├─ pypyrus             
│  ├─ core -> core abstractions            
│  ├─ instrumentation -> library specific instrumentation (e.g. PyTorch, TensorFlow, JAX)
│  ├─ provenance -> event definitions and provenance logic
│  ├─ reporting  -> query and report generation logic      
│  ├─ storage -> store implementations (e.g. SQLite)                 
│  └─ utils -> helper function   
├─ scripts             
├─ tests               
├─ README.md           
└─ pyproject.toml      
```

### Core abstractions:

1. A "run" is the container that groups all provenacne events for one training execution. 

Fields:
- run_id (UUID)
- start_time (timestamp), end_time (timestamp)
- code_ref (git commit hash / dirty flag)
- config_ref (hash of config dict/file)
- envornment (python version, dependancies, hardware summary)
- tags (freeform)
Respnsibilities:
- own a `Store` 
- attach instrumentation
- ensure flush/close on exit (context manager)

2. Dataset identity (DatasetID + fingerprint):

We need a stable way to say what dataset was used, even if its not declared in the code.

DatasetID:
- name (human readable string)
- uri or path (path, s3 url, registry reference, etc)
- version (optional, if available)

Fingerprint:
- content hash strategy (configurable). Quick hash of file listings + sizes + mtimes. Or maybe strong chunk hashes / merkle tree ( slower but more robust to changes).
- The goal is to detect daataset changes even if the path is the same

3. Provenance events:
Everything that happens during a run that we want to track is an event. 
We'll keep them small and cheap.

**DatasetAccessEvent**:
- dataset_id (DatasetID), run_id (RunID)
- timestamp
- operation (read/sample,batch/getitem/iter)
- sample_index or range (optional, for fine grained tracking)
- worker_id/process_id/thread_id 
- count (if we do coalescing then per coalesced batch/window counts instead of per sample)

**TransformEvent**:
Records preprocessing steps applied to the data at a logical level
- transform_name, params_hash
- this gives us visibility into data augmentations and transformations history, i.e. what operations were applied and in what order.
- Could be emitted by instrumenting common transform libraries (e.g. torchvision.transforms), or by providing a decorator for user defined transforms. We could also "envelope" each data sample at access time so it can carry a transform history that gets updated on each transform call and then emitted when it reaches the model.

**EnvEvent**:
- captures environment details at runtime.
- python version, seeds, library versions, hardware details, etc.

4. Instrumentor (how we capture events):
- Interface:
 - attach(dataset) -> dataset_proxy
 - emits DatasetAccessEvents on access (getitem, iter, ???)
 - important design choice:
    - event coalescing: Dont log 1 event per sample (if too slow)
    - e.g., aggregate counts per worker per N milliseconds or per batch, then flush to store. 

5. Store (local provenacne store):
- Interface:
    - append_event(event)
    - flush() -> persist buffered events to disk
    - query_events(...) -> (by run, dataset, time window, op type, etc
- Baseline implementation: SQLite (easy indexing, portable, good MVP for thesis)

6. Query + Report builder:
Turn raw provenance into human-readable compliance/repro outputs.

Query layer:

- “Which datasets were used in run X?”

- “How many accesses per dataset?”

- “Which transforms happened before training?”

- “What changed between run A and run B?”

Report builder:

- outputs sections you can map to AI Act-ish needs:

- dataset identification + versioning

- access summary (counts, times, workers)
preprocessing/transform chain

- reproducibility bundle (code ref + config hash + seeds)