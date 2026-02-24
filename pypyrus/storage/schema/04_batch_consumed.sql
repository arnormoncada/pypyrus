-- ----------------------------
-- BatchConsumed: the key reproducibility event
-- ----------------------------
CREATE TABLE IF NOT EXISTS batch_consumed (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,

  global_step INTEGER NOT NULL,
  timestamp TEXT NOT NULL,

  batch_size INTEGER NOT NULL,
  batch_fingerprint TEXT NOT NULL,      -- hash of ordered sample IDs (or batch signature)
  sample_ids_json TEXT,                 -- OPTIONAL (debug/full mode)
  rng_state_hash TEXT,                  -- OPTIONAL

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,

  UNIQUE (run_id, global_step)
);

CREATE INDEX IF NOT EXISTS idx_batch_run_step ON batch_consumed(run_id, global_step);
CREATE INDEX IF NOT EXISTS idx_batch_fingerprint ON batch_consumed(batch_fingerprint);