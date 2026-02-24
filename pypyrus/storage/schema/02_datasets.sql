-- ----------------------------
-- Datasets: identity + fingerprint
-- ----------------------------
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,

  name TEXT NOT NULL,
  uri TEXT NOT NULL,
  version_hint TEXT,

  fingerprint TEXT NOT NULL,
  fingerprint_strategy TEXT NOT NULL,

  registered_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint ON datasets(fingerprint);