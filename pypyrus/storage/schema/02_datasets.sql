PRAGMA foreign_keys = ON;

-- ----------------------------
-- Dataset registrations: one row per DatasetRegisteredEvent.
-- Canonical dataset identity is the fingerprint method + fingerprint pair.
-- ----------------------------
CREATE TABLE IF NOT EXISTS dataset_registrations (
  event_id TEXT PRIMARY KEY,       -- UUID; DatasetRegisteredEvent.event_id
  run_id TEXT NOT NULL,            -- DatasetRegisteredEvent.run_id
  dataset_id TEXT NOT NULL,        -- derived canonical identity (fingerprint_method:fingerprint)

  name TEXT NOT NULL,              -- DatasetRegisteredEvent.name
  uri TEXT,                        -- DatasetRegisteredEvent.uri (nullable)
  version_hint TEXT,               -- DatasetRegisteredEvent.version_hint
  role TEXT,                       -- DatasetRegisteredEvent.role (nullable)

  fingerprint TEXT NOT NULL,       -- DatasetRegisteredEvent.fingerprint
  fingerprint_method TEXT NOT NULL,-- DatasetRegisteredEvent.fingerprint_method
  sample_id_scheme TEXT,           -- DatasetRegisteredEvent.sample_id_scheme
  sample_id_resolver TEXT,         -- DatasetRegisteredEvent.sample_id_resolver

  registered_at TEXT NOT NULL,     -- ISO 8601; DatasetRegisteredEvent.timestamp

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_registrations_run ON dataset_registrations(run_id);
CREATE INDEX IF NOT EXISTS idx_dataset_registrations_dataset_id ON dataset_registrations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_registrations_role ON dataset_registrations(run_id, role);
CREATE INDEX IF NOT EXISTS idx_dataset_registrations_fingerprint ON dataset_registrations(fingerprint_method, fingerprint);
