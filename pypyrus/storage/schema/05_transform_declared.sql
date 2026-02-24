-- ----------------------------
-- Transform pipeline: declared (not per-sample)
-- ----------------------------
CREATE TABLE IF NOT EXISTS transform_declared (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,

  transform_chain_id TEXT NOT NULL,     -- stable id within a run
  transform_list_json TEXT NOT NULL,    -- ordered list of transform names/classes
  params_hash TEXT NOT NULL,            -- hash over pipeline description
  seed_policy TEXT,                     -- global/per-worker/per-sample (optional)
  deterministic INTEGER,                -- 0/1 (best-effort)

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transform_declared_run ON transform_declared(run_id);
CREATE INDEX IF NOT EXISTS idx_transform_declared_dataset ON transform_declared(dataset_id);