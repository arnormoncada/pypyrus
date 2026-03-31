PRAGMA foreign_keys = ON;

-- ----------------------------
-- Datasets: identity + fingerprint
-- One row per DatasetRegisteredEvent.  If the same logical dataset
-- is re-registered across runs the first write wins (INSERT OR IGNORE).
-- ----------------------------
CREATE TABLE IF NOT EXISTS datasets (
  event_id TEXT PRIMARY KEY,      -- UUID; DatasetRegisteredEvent.event_id
  dataset_id TEXT NOT NULL UNIQUE, -- stable dataset identity; DatasetRegisteredEvent.dataset_id

  name TEXT NOT NULL,             -- DatasetRegisteredEvent.name
  uri TEXT,                       -- DatasetRegisteredEvent.uri (nullable)
  version_hint TEXT,              -- DatasetRegisteredEvent.version_hint

  fingerprint TEXT,               -- DatasetRegisteredEvent.fingerprint (nullable)
  fingerprint_method TEXT,        -- DatasetRegisteredEvent.fingerprint_method
  sample_id_scheme TEXT,          -- DatasetRegisteredEvent.sample_id_scheme
  sample_id_resolver TEXT,        -- DatasetRegisteredEvent.sample_id_resolver

  registered_at TEXT NOT NULL     -- ISO 8601; DatasetRegisteredEvent.timestamp
);

CREATE INDEX IF NOT EXISTS idx_datasets_dataset_id  ON datasets(dataset_id);
CREATE INDEX IF NOT EXISTS idx_datasets_name        ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint ON datasets(fingerprint);
