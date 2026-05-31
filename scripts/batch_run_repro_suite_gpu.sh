#!/bin/bash
#BSUB -J pypyrus_repro_suite
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o logs/pypyrus_repro_suite_%J.out
#BSUB -e logs/pypyrus_repro_suite_%J.err
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

CASE="${CASE:-all}"
DATA_ROOT="${DATA_ROOT:-experiments/plant_seedlings/data/split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/results/repro_suite}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-99}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"

CMD=(
  python experiments/repro/run_repro_suite.py
  --case "${CASE}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed-a "${SEED_A}"
  --seed-b "${SEED_B}"
)

if [[ "${RESET_OUTPUT}" == "1" ]]; then
  CMD+=(--reset-output)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
