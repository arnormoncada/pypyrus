#!/bin/bash

#BSUB -J pypyrus_instr_compare
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o logs/pypyrus_instr_compare_%J.out
#BSUB -e logs/pypyrus_instr_compare_%J.err

set -euo pipefail

# When submitted as "bsub < script.sh", BASH_SOURCE may not point to the repo.
# LSB_SUBCWD is the submit directory and is the most reliable root anchor.
REPO_ROOT="${LSB_SUBCWD:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: pyproject.toml not found in submit directory: ${REPO_ROOT}" >&2
  echo "Submit from the pypyrus repo root, e.g. cd ~/git/pypyrus" >&2
  exit 1
fi

mkdir -p logs

# Some GPU queues preload nvhpc, which can conflict with cuda/11.8 in the conda init stack.
module unload nvhpc >/dev/null 2>&1 || true

# Initialize the DTU conda environment used on the course cluster.
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "Host: $(hostname)"
echo "Working directory: $PWD"
lscpu | grep "Model name" || true
echo "GPU Model(s):"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq

# Prefer uv when available; otherwise fall back to pip.
if command -v uv >/dev/null 2>&1; then
  uv pip install --system -e .
else
  python -m pip install --upgrade pip
  python -m pip install -e .
fi

bash examples/plant_seedlings/run_instrumentation_compare.sh