-- ----------------------------
-- BatchConsumed: the key reproducibility event
-- ----------------------------
CREATE TABLE IF NOT EXISTS batch_consumed (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL, -- FK to runs(run_id)
  dataset_id TEXT NOT NULL, -- FK to datasets(dataset_id)

  global_step INTEGER NOT NULL, -- monotonic step counter within a run; can be used to order events
  timestamp TEXT NOT NULL, -- ISO 8601 timestamp generated at event generation time; 

  batch_size INTEGER NOT NULL, -- number of samples in the batch
  batch_fingerprint TEXT NOT NULL,      -- hash of ordered sample IDs (or batch signature)
  sample_ids_json TEXT,                 -- OPTIONAL (debug/full mode) OR we can try and compress this
  rng_state_hash TEXT,                  -- OPTIONAL WILL PROBABLY REMOVE THIS; 

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,

  UNIQUE (run_id, global_step)
);

CREATE INDEX IF NOT EXISTS idx_batch_run_step ON batch_consumed(run_id, global_step);
CREATE INDEX IF NOT EXISTS idx_batch_fingerprint ON batch_consumed(batch_fingerprint);