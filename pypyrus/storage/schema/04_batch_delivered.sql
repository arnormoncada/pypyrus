PRAGMA foreign_keys = ON;

-- ----------------------------
-- BatchDelivered: the key reproducibility event
-- One row per BatchDeliveredEvent.
-- ----------------------------
CREATE TABLE IF NOT EXISTS batch_delivered (
  event_id TEXT PRIMARY KEY,      -- UUID; BatchDeliveredEvent.event_id
  run_id TEXT NOT NULL,           -- BatchDeliveredEvent.run_id
  loader_id TEXT NOT NULL,        -- BatchDeliveredEvent.loader_id

  global_step INTEGER NOT NULL,     -- BatchDeliveredEvent.global_step (per-loader counter)
  global_sequence INTEGER NOT NULL, -- BatchDeliveredEvent.global_sequence (run-wide counter)
  timestamp TEXT NOT NULL,          -- ISO 8601; BatchDeliveredEvent.timestamp

  batch_size INTEGER NOT NULL,    -- BatchDeliveredEvent.batch_size
  batch_fingerprint TEXT NOT NULL, -- BatchDeliveredEvent.batch_fingerprint
  sample_ids_blob BLOB,           -- BatchDeliveredEvent.sample_ids_blob (gzip-compressed bytes)

  FOREIGN KEY (run_id)     REFERENCES runs(run_id)              ON DELETE CASCADE,
  FOREIGN KEY (loader_id)  REFERENCES loaders(loader_id)        ON DELETE CASCADE,

  UNIQUE (loader_id, global_step),
  UNIQUE (run_id, global_sequence)
);

CREATE INDEX IF NOT EXISTS idx_batch_loader_step  ON batch_delivered(loader_id, global_step);
CREATE INDEX IF NOT EXISTS idx_batch_sequence     ON batch_delivered(run_id, global_sequence);
CREATE INDEX IF NOT EXISTS idx_batch_fingerprint  ON batch_delivered(batch_fingerprint);
