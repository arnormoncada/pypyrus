PRAGMA foreign_keys = ON;

-- ----------------------------
-- Loaders: one logical attached loader per run
-- ----------------------------
CREATE TABLE IF NOT EXISTS loaders (
  event_id TEXT PRIMARY KEY,      -- UUID; LoaderRegisteredEvent.event_id
  loader_id TEXT NOT NULL UNIQUE, -- stable ID for one attached loader within a run
  run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  role TEXT NOT NULL,
  registered_at TEXT NOT NULL,

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_loaders_run ON loaders(run_id);
CREATE INDEX IF NOT EXISTS idx_loaders_role ON loaders(run_id, role);
CREATE INDEX IF NOT EXISTS idx_loaders_dataset ON loaders(run_id, dataset_id);
