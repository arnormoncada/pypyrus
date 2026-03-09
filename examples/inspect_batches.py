from pypyrus.reporting.queries import get_batches_for_run
from pypyrus.storage.sqlite_store import SQLiteStore


store = SQLiteStore("pypyrus.db")

run_ids = store.list_runs()
if not run_ids:
    print("No runs found.")
    raise SystemExit(0)

run_id = run_ids[1]  # Just pick the second run for demonstration. Adjust as needed.
batches = get_batches_for_run(store, run_id, include_sample_ids=True)

print(f"Run: {run_id}")
print(f"Num batches: {len(batches)}\n")

for batch in batches[:5]:
    print(
        f"step={batch['global_step']} "
        f"size={batch['batch_size']} "
        f"fingerprint={batch['batch_fingerprint'][:12]} "
        f"sample_ids={batch['sample_ids']}"
    )

store.close()