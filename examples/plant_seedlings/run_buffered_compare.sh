#!/usr/bin/env bash
set -euo pipefail

# All settings can be overridden via environment variables, e.g.:
# EPOCHS=10 PAIRS=20 WARMUP_PAIRS=1 BUFFERED_QUEUE_SIZE=64 \
# bash examples/plant_seedlings/run_buffered_compare.sh
DATA_ROOT="${DATA_ROOT:-examples/plant_seedlings/data/split}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TIMING_FILE="${TIMING_FILE:-examples/plant_seedlings/timings_buffered_compare.txt}"
PAIRS="${PAIRS:-20}"
WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
BUFFERED_QUEUE_SIZE="${BUFFERED_QUEUE_SIZE:-1024}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESET_TIMINGS="${RESET_TIMINGS:-1}"
WARMUP_TIMING_FILE="${TIMING_FILE}.warmup.tmp"

run_once() {
  local use_buffered="$1"
  local timing_file="$2"
  local mode_label="sync"

  if [[ "$use_buffered" == "true" ]]; then
    mode_label="buffered"
    "$PYTHON_BIN" examples/plant_seedlings/train_mobilenetv3_small.py \
      --data-root "$DATA_ROOT" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --buffered-queue \
      --buffered-queue-size "$BUFFERED_QUEUE_SIZE" \
      --timing-file "$timing_file"
  else
    "$PYTHON_BIN" examples/plant_seedlings/train_mobilenetv3_small.py \
      --data-root "$DATA_ROOT" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --timing-file "$timing_file"
  fi

  # Append mode label and elapsed from the last timing line emitted by the trainer.
  local trainer_line
  trainer_line="$(tail -n 1 "$timing_file")"
  local elapsed
  elapsed="$(echo "$trainer_line" | sed -n 's/.*elapsed_seconds=\([^ ]*\).*/\1/p')"
  if [[ -z "$elapsed" ]]; then
    echo "ERROR: Could not parse elapsed_seconds from timing line: $trainer_line" >&2
    exit 1
  fi
  printf "buffered=%s epochs=%s batch_size=%s num_workers=%s elapsed_seconds=%s\n" \
    "$use_buffered" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$elapsed" >> "$timing_file.buffered"

  # Keep the original trainer timing file for compatibility/debugging,
  # but analysis in this script uses the buffered-tagged companion file.
  echo "mode=$mode_label elapsed_seconds=$elapsed"
}

if [[ "$RESET_TIMINGS" == "1" ]]; then
  : > "$TIMING_FILE"
  : > "$TIMING_FILE.buffered"
fi

echo "Configuration:"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  PAIRS=$PAIRS"
echo "  WARMUP_PAIRS=$WARMUP_PAIRS"
echo "  BUFFERED_QUEUE_SIZE=$BUFFERED_QUEUE_SIZE"
echo "  TIMING_FILE=$TIMING_FILE"
echo "  BUFFERED_TIMING_FILE=$TIMING_FILE.buffered"
echo

if [[ "$WARMUP_PAIRS" -gt 0 ]]; then
  : > "$WARMUP_TIMING_FILE"
  : > "$WARMUP_TIMING_FILE.buffered"
  echo "Running $WARMUP_PAIRS warm-up pair(s) (excluded from summary)..."
  for i in $(seq 1 "$WARMUP_PAIRS"); do
    echo
    echo "Warm-up $i/$WARMUP_PAIRS: BUFFERED"
    run_once true "$WARMUP_TIMING_FILE"

    echo "Warm-up $i/$WARMUP_PAIRS: SYNC"
    run_once false "$WARMUP_TIMING_FILE"
  done
  rm -f "$WARMUP_TIMING_FILE" "$WARMUP_TIMING_FILE.buffered"
fi

echo
echo "Running $PAIRS measured pair(s)..."
for i in $(seq 1 "$PAIRS"); do
  echo
  echo "Pair $i/$PAIRS: BUFFERED"
  run_once true "$TIMING_FILE"

  echo "Pair $i/$PAIRS: SYNC"
  run_once false "$TIMING_FILE"
done

echo
echo "Done. Buffered-vs-sync timings:"
cat "$TIMING_FILE.buffered"

echo
echo "Summary:"
"$PYTHON_BIN" examples/plant_seedlings/summarize_buffered_timings.py --timing-file "$TIMING_FILE.buffered"
