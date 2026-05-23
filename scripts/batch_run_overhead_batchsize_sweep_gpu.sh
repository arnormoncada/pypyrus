#!/bin/bash
#BSUB -J pypyrus_overhead_bsweep
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o logs/pypyrus_overhead_bsweep_%J.out
#BSUB -e logs/pypyrus_overhead_bsweep_%J.err
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
lscpu | egrep 'Architecture|CPU\(s\)|Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|L1d cache|L1i cache|L2 cache|L3 cache'
echo "GPU Model(s):"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq
echo "Memory:"
free -h || grep MemTotal /proc/meminfo
echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv,noheader

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

DATA_PATH="${DATA_PATH:-experiments/forest_covertype/data/covtype_with_sample_id.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/overhead/batchsize_sweep}"
BATCH_SIZES="${BATCH_SIZES:-64 128 256 512 1024}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TEST_RATIO="${TEST_RATIO:-0.2}"
PAIRS="${PAIRS:-3}"
WARMUP_PAIRS="${WARMUP_PAIRS:-1}"
BASE_SEED="${BASE_SEED:-100}"
SUMMARY_TSV="${SUMMARY_TSV:-${OUTPUT_DIR}/covtype_batchsize_sweep_summary.tsv}"

mkdir -p "${OUTPUT_DIR}"

echo "Sweep configuration:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  BATCH_SIZES=${BATCH_SIZES}"
echo "  EPOCHS=${EPOCHS}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  HIDDEN_DIM=${HIDDEN_DIM}"
echo "  TEST_RATIO=${TEST_RATIO}"
echo "  PAIRS=${PAIRS}"
echo "  WARMUP_PAIRS=${WARMUP_PAIRS}"
echo "  BASE_SEED=${BASE_SEED}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SUMMARY_TSV=${SUMMARY_TSV}"
echo

printf 'batch_size\tpairs\twarmup_pairs\tepochs\tnum_workers\tmean_with_seconds\tmean_without_seconds\tmean_delta_seconds\tmean_percent\n' > "${SUMMARY_TSV}"

for batch_size in ${BATCH_SIZES}; do
  timing_file="${OUTPUT_DIR}/covtype_bs${batch_size}_timings.txt"
  summary_file="${OUTPUT_DIR}/covtype_bs${batch_size}_summary.txt"

  echo "============================================================"
  echo "Running covtype batch-size sweep for batch_size=${batch_size}"
  echo "  timing_file=${timing_file}"
  echo "  summary_file=${summary_file}"
  echo "============================================================"

  DATA_PATH="${DATA_PATH}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${batch_size}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  HIDDEN_DIM="${HIDDEN_DIM}" \
  TEST_RATIO="${TEST_RATIO}" \
  PAIRS="${PAIRS}" \
  WARMUP_PAIRS="${WARMUP_PAIRS}" \
  BASE_SEED="${BASE_SEED}" \
  TIMING_FILE="${timing_file}" \
  PYTHON_BIN=python \
  RESET_TIMINGS=1 \
  bash experiments/forest_covertype/run_instrumentation_compare.sh | tee "${summary_file}"

  python - <<'PY' "${timing_file}" "${batch_size}" "${PAIRS}" "${WARMUP_PAIRS}" "${EPOCHS}" "${NUM_WORKERS}" >> "${SUMMARY_TSV}"
import statistics
import sys
from pathlib import Path

from experiments.plant_seedlings.summarize_timings import parse_line

timing_file = Path(sys.argv[1])
batch_size = sys.argv[2]
pairs = sys.argv[3]
warmup_pairs = sys.argv[4]
epochs = sys.argv[5]
num_workers = sys.argv[6]

rows = []
for line in timing_file.read_text(encoding="utf-8").splitlines():
    row = parse_line(line)
    if row is not None:
        rows.append(row)

with_inst = [row.elapsed_seconds for row in rows if row.instrumentation]
without_inst = [row.elapsed_seconds for row in rows if not row.instrumentation]

if not with_inst or not without_inst:
    raise SystemExit(f"Missing timing rows in {timing_file}")

mean_with = statistics.mean(with_inst)
mean_without = statistics.mean(without_inst)
mean_delta = mean_with - mean_without
mean_pct = (mean_delta / mean_without) * 100.0

print(
    f"{batch_size}\t{pairs}\t{warmup_pairs}\t{epochs}\t{num_workers}\t"
    f"{mean_with:.6f}\t{mean_without:.6f}\t{mean_delta:.6f}\t{mean_pct:.2f}"
)
PY

  echo
done

echo "Sweep complete."
echo "Aggregate summary: ${SUMMARY_TSV}"
cat "${SUMMARY_TSV}"
