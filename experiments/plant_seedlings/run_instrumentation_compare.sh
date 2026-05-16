#!/usr/bin/env bash
set -euo pipefail

# All settings can be overridden via environment variables, e.g.:
# EPOCHS=10 PAIRS=20 WARMUP_PAIRS=1 bash experiments/plant_seedlings/run_instrumentation_compare.sh
DATA_ROOT="${DATA_ROOT:-experiments/plant_seedlings/data/split}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TIMING_FILE="${TIMING_FILE:-experiments/plant_seedlings/timings.txt}"
PAIRS="${PAIRS:-20}"
WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESET_TIMINGS="${RESET_TIMINGS:-1}"
WARMUP_TIMING_FILE="${TIMING_FILE}.warmup.tmp"

run_once() {
  local use_instrumentation="$1"
  local timing_file="$2"

  if [[ "$use_instrumentation" == "true" ]]; then
    "$PYTHON_BIN" experiments/plant_seedlings/train_mobilenetv3_small.py \
      --data-root "$DATA_ROOT" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --timing-file "$timing_file"
  else
    "$PYTHON_BIN" experiments/plant_seedlings/train_mobilenetv3_small.py \
      --data-root "$DATA_ROOT" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --no-instrumentation \
      --timing-file "$timing_file"
  fi
}

if [[ "$RESET_TIMINGS" == "1" ]]; then
  : > "$TIMING_FILE"
fi

echo "Configuration:"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  PAIRS=$PAIRS"
echo "  WARMUP_PAIRS=$WARMUP_PAIRS"
echo "  TIMING_FILE=$TIMING_FILE"
echo

if [[ "$WARMUP_PAIRS" -gt 0 ]]; then
  : > "$WARMUP_TIMING_FILE"
  echo "Running $WARMUP_PAIRS warm-up pair(s) (excluded from summary)..."
  for i in $(seq 1 "$WARMUP_PAIRS"); do
    echo
    echo "Warm-up $i/$WARMUP_PAIRS: WITH instrumentation"
    run_once true "$WARMUP_TIMING_FILE"

    echo "Warm-up $i/$WARMUP_PAIRS: WITHOUT instrumentation"
    run_once false "$WARMUP_TIMING_FILE"
  done
  rm -f "$WARMUP_TIMING_FILE"
fi

echo
echo "Running $PAIRS measured pair(s)..."
for i in $(seq 1 "$PAIRS"); do
  echo
  echo "Pair $i/$PAIRS: WITH instrumentation"
  run_once true "$TIMING_FILE"

  echo "Pair $i/$PAIRS: WITHOUT instrumentation"
  run_once false "$TIMING_FILE"
done

echo
echo "Done. Timings:"
cat "$TIMING_FILE"

echo
echo "Summary:"
"$PYTHON_BIN" experiments/plant_seedlings/summarize_timings.py --timing-file "$TIMING_FILE"