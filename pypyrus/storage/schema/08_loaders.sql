PRAGMA foreign_keys = ON;

-- ----------------------------
-- Loaders: one logical attached loader per run
-- ----------------------------
CREATE TABLE IF NOT EXISTS loaders (
  event_id TEXT PRIMARY KEY,      -- UUID; LoaderRegisteredEvent.event_id
  loader_id TEXT NOT NULL UNIQUE, -- stable ID for one attached loader within a run
  run_id TEXT NOT NULL,
  dataset_registration_event_id TEXT NOT NULL,
  registered_at TEXT NOT NULL,

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (dataset_registration_event_id) REFERENCES dataset_registrations(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_loaders_run ON loaders(run_id);
CREATE INDEX IF NOT EXISTS idx_loaders_registration ON loaders(run_id, dataset_registration_event_id);
