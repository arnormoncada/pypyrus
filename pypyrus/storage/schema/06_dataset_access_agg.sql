-- ----------------------------
-- DatasetAccessAgg: optional aggregated usage stats
-- (keep this coalesced, not per-sample unless debugging)
-- ----------------------------
CREATE TABLE IF NOT EXISTS dataset_access_agg (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,

  operation TEXT NOT NULL,              -- getitem/iter/batch_fetch
  worker_id INTEGER,                    -- optional
  process_id INTEGER,                   -- optional

  -- aggregated evidence
  count INTEGER NOT NULL,               -- number of accesses represented
  sample_ref_json TEXT,                 -- optional (idx/range/hash), keep small

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_access_agg_run ON dataset_access_agg(run_id);
CREATE INDEX IF NOT EXISTS idx_access_agg_dataset ON dataset_access_agg(dataset_id);