-- ----------------------------
-- Datasets: identity + fingerprint
-- ----------------------------
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY, -- UUID or id derived from uri, fingerprint, etc.

  name TEXT NOT NULL, -- human-friendly name (e.g. "CIFAR-10 train split")
  uri TEXT NOT NULL, -- where to find the dataset (e.g. local path, s3 uri, etc.)
  version_hint TEXT, -- optional version hint (e.g. HF revision, DVC commit, etc.)

  fingerprint TEXT NOT NULL, -- dataset fingerprint
  fingerprint_strategy TEXT NOT NULL, -- how the fingerprint was computed (e.g. "hash of all files", "hash of metadata + sample of contents", etc.)

  registered_at TEXT NOT NULL -- ISO 8601 timestamp
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint ON datasets(fingerprint);