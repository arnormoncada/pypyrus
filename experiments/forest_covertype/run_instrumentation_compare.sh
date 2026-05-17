#!/usr/bin/env bash
set -euo pipefail

# All settings can be overridden via environment variables, e.g.:
# EPOCHS=3 PAIRS=10 WARMUP_PAIRS=1 bash experiments/forest_covertype/run_instrumentation_compare.sh
DATA_PATH="${DATA_PATH:-experiments/forest_covertype/data/covtype_with_sample_id.csv}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TEST_RATIO="${TEST_RATIO:-0.2}"
TIMING_FILE="${TIMING_FILE:-experiments/results/overhead/covtype_timings.txt}"
PAIRS="${PAIRS:-10}"
WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESET_TIMINGS="${RESET_TIMINGS:-1}"
BASE_SEED="${BASE_SEED:-100}"
WARMUP_TIMING_FILE="${TIMING_FILE}.warmup.tmp"

run_once() {
  local use_instrumentation="$1"
  local timing_file="$2"
  local seed="$3"

  if [[ "$use_instrumentation" == "true" ]]; then
    "$PYTHON_BIN" experiments/forest_covertype/train_covtype_mlp.py \
      --data-path "$DATA_PATH" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$HIDDEN_DIM" \
      --test-ratio "$TEST_RATIO" \
      --num-workers "$NUM_WORKERS" \
      --seed "$seed" \
      --timing-file "$timing_file"
  else
    "$PYTHON_BIN" experiments/forest_covertype/train_covtype_mlp.py \
      --data-path "$DATA_PATH" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$HIDDEN_DIM" \
      --test-ratio "$TEST_RATIO" \
      --num-workers "$NUM_WORKERS" \
      --seed "$seed" \
      --no-instrumentation \
      --timing-file "$timing_file"
  fi
}

if [[ "$RESET_TIMINGS" == "1" ]]; then
  : > "$TIMING_FILE"
fi

echo "Configuration:"
echo "  DATA_PATH=$DATA_PATH"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  HIDDEN_DIM=$HIDDEN_DIM"
echo "  TEST_RATIO=$TEST_RATIO"
echo "  PAIRS=$PAIRS"
echo "  WARMUP_PAIRS=$WARMUP_PAIRS"
echo "  BASE_SEED=$BASE_SEED"
echo "  TIMING_FILE=$TIMING_FILE"
echo

if [[ "$WARMUP_PAIRS" -gt 0 ]]; then
  : > "$WARMUP_TIMING_FILE"
  echo "Running $WARMUP_PAIRS warm-up pair(s) (excluded from summary)..."
  for i in $(seq 1 "$WARMUP_PAIRS"); do
    seed=$((BASE_SEED + i - 1))
    echo
    echo "Warm-up $i/$WARMUP_PAIRS: WITH instrumentation (seed=$seed)"
    run_once true "$WARMUP_TIMING_FILE" "$seed"

    echo "Warm-up $i/$WARMUP_PAIRS: WITHOUT instrumentation (seed=$seed)"
    run_once false "$WARMUP_TIMING_FILE" "$seed"
  done
  rm -f "$WARMUP_TIMING_FILE"
fi

echo
echo "Running $PAIRS measured pair(s)..."
for i in $(seq 1 "$PAIRS"); do
  seed=$((BASE_SEED + WARMUP_PAIRS + i - 1))
  echo
  echo "Pair $i/$PAIRS: WITH instrumentation (seed=$seed)"
  run_once true "$TIMING_FILE" "$seed"

  echo "Pair $i/$PAIRS: WITHOUT instrumentation (seed=$seed)"
  run_once false "$TIMING_FILE" "$seed"
done

echo
echo "Done. Timings:"
cat "$TIMING_FILE"

echo
echo "Summary:"
"$PYTHON_BIN" experiments/plant_seedlings/summarize_timings.py --timing-file "$TIMING_FILE"
