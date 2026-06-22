#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the repo root." >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  "numpy==1.26.4" \
  "torch==2.2.2" \
  "torchvision==0.17.2"
python -m pip install --no-deps -e .

echo "Installed PyPyrus into .venv with pinned NumPy/Torch/Torchvision versions."
