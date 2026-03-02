-- ----------------------------
-- Run metadata: config, environment, extra info
-- ----------------------------
CREATE TABLE IF NOT EXISTS run_metadata (
  run_id           TEXT NOT NULL PRIMARY KEY, -- FK to runs.run_id

  config_json      TEXT, -- full config JSON; NULL if not captured
  environment_json TEXT, -- full env snapshot JSON; NULL if not captured
  metadata_json    TEXT, -- any extra structured info; NULL if not provided

  FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
);
