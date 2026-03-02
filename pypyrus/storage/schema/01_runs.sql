PRAGMA foreign_keys = ON;

-- ----------------------------
-- Runs: audit boundary
-- ----------------------------
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, -- UUID
  start_time TEXT NOT NULL, -- ISO 8601 timestamp
  end_time TEXT, -- ISO 8601 timestamp; NULL if still running

  -- reproducibility bundle
  git_commit TEXT, -- git commit hash (40 chars)
  git_dirty INTEGER,              -- 0/1
  config_hash TEXT,               -- hash of config file/dict
  env_hash TEXT,                  -- hash of env snapshot

  -- optional lightweight metadata
  tags_json TEXT                  -- optional; keep as JSON string
);

CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time);