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

# Long-run prototype defaults for ~3-4h on M3 Pro based on measured ~2.0 sec/step.
# Override with environment variables as needed.
MAX_PAPERS="${MAX_PAPERS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-256}"
EPOCHS="${EPOCHS:-1}"
D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-6}"
MAX_STEPS="${MAX_STEPS:-6000}"          # ~3.3h at ~2 sec/step
MIN_TOKENS="${MIN_TOKENS:-12}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
STREAM_MAX_RETRIES="${STREAM_MAX_RETRIES:-100}"
STREAM_RETRY_BACKOFF_SECONDS="${STREAM_RETRY_BACKOFF_SECONDS:-5}"
CATEGORY="${CATEGORY:-physics}"
LR="${LR:-2e-4}"
PREFIX="${PREFIX:-prototype_long4h}"

echo "Starting long prototype run"
echo "Estimated time ~= MAX_STEPS * 2 sec (machine-dependent)"

time python "$ROOT/stream_train.py" \
  --mode prototype \
  --tokenizer-model "$ROOT/../model_32k.json" \
  --category "$CATEGORY" \
  --max-papers "$MAX_PAPERS" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --epochs "$EPOCHS" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --n-heads "$N_HEADS" \
  --lr "$LR" \
  --max-steps-per-epoch "$MAX_STEPS" \
  --min-tokens "$MIN_TOKENS" \
  --delay-seconds 1 \
  --stream-max-retries "$STREAM_MAX_RETRIES" \
  --stream-retry-backoff-seconds "$STREAM_RETRY_BACKOFF_SECONDS" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --checkpoint-prefix "$PREFIX"

echo "Run complete. Final summary: $ROOT/checkpoints/${PREFIX}_summary.json"
