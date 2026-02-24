PRAGMA foreign_keys = ON;

-- ----------------------------
-- Runs: audit boundary
-- ----------------------------
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  start_time TEXT NOT NULL,
  end_time TEXT,

  -- reproducibility bundle
  git_commit TEXT,
  git_dirty INTEGER,              -- 0/1
  config_hash TEXT,               -- hash of config file/dict
  env_hash TEXT,                  -- hash of env snapshot

  -- optional lightweight metadata
  tags_json TEXT                  -- optional; keep as JSON string
);

CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time);