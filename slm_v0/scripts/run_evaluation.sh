#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "$ROOT/venv/bin/activate" ]]; then
  source "$ROOT/venv/bin/activate"
elif [[ -f "$ROOT/../../venv/bin/activate" ]]; then
  source "$ROOT/../../venv/bin/activate"
fi

python "$ROOT/scripts/evaluate_checkpoints.py" \
  --checkpoints-glob "prototype_laptop_epoch*.pt" \
  --category physics \
  --max-papers 200 \
  --eval-steps 20 \
  --batch-size 2 \
  --seq-len 128 \
  --min-tokens 12 \
  --output "$ROOT/checkpoints/best_checkpoint.json"
