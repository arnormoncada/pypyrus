#!/bin/bash
#BSUB -J pypyrus_storage
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o logs/pypyrus_storage_%J.out
#BSUB -e logs/pypyrus_storage_%J.err
#BSUB -B
#BSUB -N
#BSUB -u s241645@dtu.dk

set -euo pipefail

REPO_ROOT="${LSB_SUBCWD:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: pyproject.toml not found in submit directory: ${REPO_ROOT}" >&2
  echo "Submit from repo root, e.g. cd ~/git/pypyrus" >&2
  exit 1
fi

mkdir -p logs

module unload nvhpc >/dev/null 2>&1 || true

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate pypyrus-v100

export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "Host: $(hostname)"
echo "Working directory: $PWD"
lscpu | grep "Model name" || true
echo "GPU Model(s):"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
ALLOWED_GPU_REGEX="${ALLOWED_GPU_REGEX:-A100}"
if ! echo "${GPU_NAME}" | grep -Eiq "${ALLOWED_GPU_REGEX}"; then
  echo "ERROR: Unexpected GPU '${GPU_NAME}'. Allowed: ${ALLOWED_GPU_REGEX}" >&2
  exit 1
fi

echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable =', sys.executable)"

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda.is_available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this job environment")
print("device_name =", torch.cuda.get_device_name(0))
print("device_cc   =", torch.cuda.get_device_capability(0))
PY

python -m pip install --no-deps -e .

WORKLOAD="${WORKLOAD:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/results/storage_footprint}"
PLANT_DATA_ROOT="${PLANT_DATA_ROOT:-experiments/plant_seedlings/data/split}"
COVTYPE_DATA_PATH="${COVTYPE_DATA_PATH:-experiments/forest_covertype/data/covtype_with_sample_id.csv}"
PLANT_EPOCHS="${PLANT_EPOCHS:-10}"
COVTYPE_EPOCHS="${COVTYPE_EPOCHS:-10}"
PLANT_BATCH_SIZE="${PLANT_BATCH_SIZE:-32}"
COVTYPE_BATCH_SIZE="${COVTYPE_BATCH_SIZE:-256}"
PLANT_NUM_WORKERS="${PLANT_NUM_WORKERS:-2}"
COVTYPE_NUM_WORKERS="${COVTYPE_NUM_WORKERS:-0}"
COVTYPE_HIDDEN_DIM="${COVTYPE_HIDDEN_DIM:-128}"
COVTYPE_TEST_RATIO="${COVTYPE_TEST_RATIO:-0.2}"
SEED="${SEED:-42}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"

CMD=(
  python experiments/storage/run_storage_footprint.py
  --workload "${WORKLOAD}"
  --output-root "${OUTPUT_ROOT}"
  --plant-data-root "${PLANT_DATA_ROOT}"
  --covtype-data-path "${COVTYPE_DATA_PATH}"
  --plant-epochs "${PLANT_EPOCHS}"
  --covtype-epochs "${COVTYPE_EPOCHS}"
  --plant-batch-size "${PLANT_BATCH_SIZE}"
  --covtype-batch-size "${COVTYPE_BATCH_SIZE}"
  --plant-num-workers "${PLANT_NUM_WORKERS}"
  --covtype-num-workers "${COVTYPE_NUM_WORKERS}"
  --covtype-hidden-dim "${COVTYPE_HIDDEN_DIM}"
  --covtype-test-ratio "${COVTYPE_TEST_RATIO}"
  --seed "${SEED}"
)

if [[ "${RESET_OUTPUT}" == "1" ]]; then
  CMD+=(--reset-output)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
