PRAGMA foreign_keys = ON;

-- ----------------------------
-- Link: which datasets were used in which runs
-- Derived from DatasetRegisteredEvent; one row per event.
-- ----------------------------
CREATE TABLE IF NOT EXISTS run_datasets (
  run_id TEXT NOT NULL,           -- DatasetRegisteredEvent.run_id
  dataset_id TEXT NOT NULL,       -- DatasetRegisteredEvent.dataset_id
  registered_at TEXT NOT NULL,    -- DatasetRegisteredEvent.timestamp

  PRIMARY KEY (run_id, dataset_id),
  FOREIGN KEY (run_id)     REFERENCES runs(run_id)              ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)     ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_datasets_run     ON run_datasets(run_id);
CREATE INDEX IF NOT EXISTS idx_run_datasets_dataset ON run_datasets(dataset_id);