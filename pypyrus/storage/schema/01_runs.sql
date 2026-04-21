PRAGMA foreign_keys = ON;

-- ----------------------------
-- Runs: audit boundary
-- Populated from RunStartEvent (on open) and RunEndEvent (on close).
-- ----------------------------
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,        -- UUID; RunStartEvent.run_id

  -- RunStartEvent fields
  start_time TEXT NOT NULL,       -- ISO 8601; RunStartEvent.timestamp
  code_ref TEXT,                  -- RunStartEvent.code_ref (e.g. git SHA, script path)
  config_ref TEXT,                -- RunStartEvent.config_ref
  config_json TEXT,               -- RunStartEvent.config_json serialized as JSON
  environment_hash TEXT,          -- RunStartEvent.environment_hash
  seed_summary_json TEXT,         -- RunStartEvent.seed_summary serialised as JSON

  -- RunEndEvent fields (NULL while run is still active)
  end_time TEXT,                  -- ISO 8601; RunEndEvent.timestamp
  status TEXT,                    -- 'success' / 'failure' / 'interrupted'
  event_count INTEGER             -- RunEndEvent.event_count
);

CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);