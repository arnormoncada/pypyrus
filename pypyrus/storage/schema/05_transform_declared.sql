PRAGMA foreign_keys = ON;

-- ----------------------------
-- Transform pipeline: declared (not per-sample)
-- One row per TransformDeclaredEvent.
-- ----------------------------
CREATE TABLE IF NOT EXISTS transform_declared (
  event_id TEXT PRIMARY KEY,      -- UUID; TransformDeclaredEvent.event_id
  run_id TEXT NOT NULL,           -- TransformDeclaredEvent.run_id
  dataset_registration_event_id TEXT NOT NULL, -- TransformDeclaredEvent.dataset_registration_event_id
  timestamp TEXT NOT NULL,        -- ISO 8601; TransformDeclaredEvent.timestamp

  transform_list_json TEXT NOT NULL, -- TransformDeclaredEvent.transform_list serialised as JSON array
  params_hash TEXT NOT NULL,        -- TransformDeclaredEvent.params_hash
  introspection_level TEXT NOT NULL, -- TransformDeclaredEvent.introspection_level (full/partial)

  FOREIGN KEY (run_id)     REFERENCES runs(run_id)         ON DELETE CASCADE,
  FOREIGN KEY (dataset_registration_event_id) REFERENCES dataset_registrations(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transform_declared_run     ON transform_declared(run_id);
CREATE INDEX IF NOT EXISTS idx_transform_declared_registration ON transform_declared(dataset_registration_event_id);
