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

python "$ROOT/stream_train.py" \
  --mode prototype \
  --tokenizer-model "$ROOT/../model_32k.json" \
  --category physics \
  --max-papers 200 \
  --batch-size 2 \
  --seq-len 128 \
  --epochs 1 \
  --d-model 128 \
  --n-layers 2 \
  --n-heads 4 \
  --max-steps-per-epoch 25 \
  --checkpoint-prefix prototype_laptop
