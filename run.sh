#!/usr/bin/env bash
# run.sh — set up an isolated environment, install dependencies, and run.
#
# Usage:
#   ./run.sh --csv DATASET.csv                       # default: onehot_seq + ridge (CPU, fast)
#   ./run.sh --csv DATASET.csv --features mutation --split position
#   ESM=1 ./run.sh --csv DATASET.csv --features esm  # also installs torch + fair-esm
#
# Any arguments you pass are forwarded straight to seq2function.py.
# Re-running is fast: the virtual environment is reused once created.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"
REQS="requirements.txt"

# ESM=1 pulls in the optional torch / fair-esm extras.
if [ "${ESM:-0}" != "0" ]; then
  REQS="requirements-esm.txt"
fi

if [ ! -d "$VENV" ]; then
  echo ">> Creating virtual environment in $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> Installing $REQS"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$REQS"

echo ">> Running seq2function.py"
python seq2function.py "$@"
