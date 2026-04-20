#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="examples/plant_seedlings/data/split"
EPOCHS=3
BATCH_SIZE=32
NUM_WORKERS=2
TIMING_FILE="examples/plant_seedlings/timings.txt"
TRIALS=10

# Optional: reset previous timings
: > "$TIMING_FILE"

echo "Running $TRIALS alternating trials per mode..."
for i in $(seq 1 "$TRIALS"); do
  echo
  echo "Trial $i/$TRIALS: WITH instrumentation"
  python examples/plant_seedlings/train_mobilenetv3_small.py \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --timing-file "$TIMING_FILE"

  echo "Trial $i/$TRIALS: WITHOUT instrumentation"
  python examples/plant_seedlings/train_mobilenetv3_small.py \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --no-instrumentation \
    --timing-file "$TIMING_FILE"
done

echo
echo "Done. Timings:"
cat "$TIMING_FILE"

echo
echo "Summary:"
python examples/plant_seedlings/summarize_timings.py --timing-file "$TIMING_FILE"