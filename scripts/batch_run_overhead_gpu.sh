#!/bin/bash
#BSUB -J pypyrus_overhead
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o logs/pypyrus_overhead_%J.out
#BSUB -e logs/pypyrus_overhead_%J.err
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

WORKLOAD="${WORKLOAD:-plant}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/overhead}"
mkdir -p "${OUTPUT_DIR}"

case "${WORKLOAD}" in
  plant)
    DATA_ROOT="${DATA_ROOT:-experiments/plant_seedlings/data/split}"
    EPOCHS="${EPOCHS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    PAIRS="${PAIRS:-10}"
    WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
    TIMING_FILE="${TIMING_FILE:-${OUTPUT_DIR}/plant_timings.txt}"

    DATA_ROOT="${DATA_ROOT}" \
    EPOCHS="${EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    PAIRS="${PAIRS}" \
    WARMUP_PAIRS="${WARMUP_PAIRS}" \
    TIMING_FILE="${TIMING_FILE}" \
    PYTHON_BIN=python \
    RESET_TIMINGS=1 \
    bash experiments/plant_seedlings/run_instrumentation_compare.sh
    ;;
  covtype)
    DATA_PATH="${DATA_PATH:-experiments/forest_covertype/data/covtype_with_sample_id.csv}"
    EPOCHS="${EPOCHS:-3}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    NUM_WORKERS="${NUM_WORKERS:-0}"
    HIDDEN_DIM="${HIDDEN_DIM:-128}"
    TEST_RATIO="${TEST_RATIO:-0.2}"
    PAIRS="${PAIRS:-10}"
    WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
    BASE_SEED="${BASE_SEED:-100}"
    TIMING_FILE="${TIMING_FILE:-${OUTPUT_DIR}/covtype_timings.txt}"

    DATA_PATH="${DATA_PATH}" \
    EPOCHS="${EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    HIDDEN_DIM="${HIDDEN_DIM}" \
    TEST_RATIO="${TEST_RATIO}" \
    PAIRS="${PAIRS}" \
    WARMUP_PAIRS="${WARMUP_PAIRS}" \
    BASE_SEED="${BASE_SEED}" \
    TIMING_FILE="${TIMING_FILE}" \
    PYTHON_BIN=python \
    RESET_TIMINGS=1 \
    bash experiments/forest_covertype/run_instrumentation_compare.sh
    ;;
  *)
    echo "ERROR: Unsupported WORKLOAD='${WORKLOAD}'. Expected 'plant' or 'covtype'." >&2
    exit 1
    ;;
esac
