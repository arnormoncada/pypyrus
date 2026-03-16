from pypyrus.reporting.compare import compare_runs, format_run_comparison
from pypyrus.storage.sqlite_store import SQLiteStore


store = SQLiteStore("pypyrus.db")
run_ids = store.list_runs()

if len(run_ids) < 2:
    print("Need at least two runs to compare.")
    raise SystemExit(0)

run_id_a = run_ids[-2]
run_id_b = run_ids[-1]

result = compare_runs(store, run_id_a, run_id_b)
print(format_run_comparison(result))

store.close()