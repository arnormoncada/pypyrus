-- ----------------------------
-- Link: which datasets were used in which runs
-- ----------------------------
CREATE TABLE IF NOT EXISTS run_datasets (
  run_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  role TEXT,                      -- e.g. train/val/test/unspecified (optional)
  registered_at TEXT NOT NULL,

  PRIMARY KEY (run_id, dataset_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_datasets_run ON run_datasets(run_id);