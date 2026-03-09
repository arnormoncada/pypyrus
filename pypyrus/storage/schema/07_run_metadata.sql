PRAGMA foreign_keys = ON;

-- ----------------------------
-- Environment snapshot
-- One row per EnvironmentSnapshotEvent.
-- ----------------------------
CREATE TABLE IF NOT EXISTS environment_snapshot (
  event_id TEXT PRIMARY KEY,      -- UUID; EnvironmentSnapshotEvent.event_id
  run_id TEXT NOT NULL,           -- EnvironmentSnapshotEvent.run_id
  timestamp TEXT NOT NULL,        -- ISO 8601; EnvironmentSnapshotEvent.timestamp

  python_version TEXT NOT NULL,   -- EnvironmentSnapshotEvent.python_version
  library_versions_hash TEXT,     -- EnvironmentSnapshotEvent.library_versions_hash
  hardware_summary TEXT,          -- EnvironmentSnapshotEvent.hardware_summary
  cuda_version TEXT,              -- EnvironmentSnapshotEvent.cuda_version

  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_env_snapshot_run ON environment_snapshot(run_id);
