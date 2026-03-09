PRAGMA foreign_keys = ON;

-- ----------------------------
-- Transform pipeline: declared (not per-sample)
-- One row per TransformDeclaredEvent.
-- ----------------------------
CREATE TABLE IF NOT EXISTS transform_declared (
  event_id TEXT PRIMARY KEY,      -- UUID; TransformDeclaredEvent.event_id
  run_id TEXT NOT NULL,           -- TransformDeclaredEvent.run_id
  dataset_id TEXT NOT NULL,       -- TransformDeclaredEvent.dataset_id
  timestamp TEXT NOT NULL,        -- ISO 8601; TransformDeclaredEvent.timestamp

  transform_chain_id TEXT NOT NULL, -- TransformDeclaredEvent.transform_chain_id
  transform_list_json TEXT NOT NULL, -- TransformDeclaredEvent.transform_list serialised as JSON array
  params_hash TEXT NOT NULL,        -- TransformDeclaredEvent.params_hash
  deterministic_flag INTEGER NOT NULL, -- TransformDeclaredEvent.deterministic_flag (0/1)
  seed_policy TEXT NOT NULL,        -- TransformDeclaredEvent.seed_policy

  FOREIGN KEY (run_id)     REFERENCES runs(run_id)         ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transform_declared_run     ON transform_declared(run_id);
CREATE INDEX IF NOT EXISTS idx_transform_declared_dataset ON transform_declared(dataset_id);
CREATE INDEX IF NOT EXISTS idx_transform_chain            ON transform_declared(transform_chain_id);