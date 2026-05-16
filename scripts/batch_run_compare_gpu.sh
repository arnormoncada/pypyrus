#!/bin/bash
#BSUB -J pypyrus_buffer_compare
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o logs/pypyrus_buffer_compare_%J.out
#BSUB -e logs/pypyrus_buffer_compare_%J.err
# Send an email when the job starts
#BSUB -B
# Send an email when the job finishes
#BSUB -N
# Specify the email address to send notifications
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

# Prevent accidental ~/.local package leakage into this run.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "Host: $(hostname)"
echo "Working directory: $PWD"
lscpu | grep "Model name" || true
echo "GPU Model(s):"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
ALLOWED_GPU_REGEX="${ALLOWED_GPU_REGEX:-Tesla V100|A100|A40}"
if ! echo "${GPU_NAME}" | grep -Eiq "${ALLOWED_GPU_REGEX}"; then
  echo "ERROR: Unexpected GPU '${GPU_NAME}'. Allowed: ${ALLOWED_GPU_REGEX}" >&2
  exit 1
fi

echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable =', sys.executable)"

# Sanity-check torch and CUDA before starting the long benchmark.
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

# Install package editable in active env without re-resolving dependencies each run.
python -m pip install --no-deps -e .

# Benchmark settings (override at submission time if needed).
EPOCHS="${EPOCHS:-10}"
PAIRS="${PAIRS:-10}"
WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TIMING_FILE="${TIMING_FILE:-experiments/plant_seedlings/timings.txt}"

DATA_ROOT="experiments/plant_seedlings/data/split" \
EPOCHS="${EPOCHS}" \
PAIRS="${PAIRS}" \
WARMUP_PAIRS="${WARMUP_PAIRS}" \
NUM_WORKERS="${NUM_WORKERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
TIMING_FILE="${TIMING_FILE}" \
PYTHON_BIN=python \
RESET_TIMINGS=1 \
bash experiments/plant_seedlings/run_buffered_compare.sh
