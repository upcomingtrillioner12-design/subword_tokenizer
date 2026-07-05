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

# Tunables (override with env vars if needed)
RUNS="${RUNS:-3}"
MAX_PAPERS="${MAX_PAPERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-128}"
EPOCHS="${EPOCHS:-1}"
D_MODEL="${D_MODEL:-128}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
MAX_STEPS="${MAX_STEPS:-25}"
MIN_TOKENS="${MIN_TOKENS:-12}"
CATEGORY="${CATEGORY:-physics}"
PREFIX_BASE="${PREFIX_BASE:-prototype_laptop_run}"

echo "Running ${RUNS} prototype trainings..."
for i in $(seq 1 "$RUNS"); do
  prefix="${PREFIX_BASE}${i}"
  echo ""
  echo "=== Training run ${i}/${RUNS} | prefix=${prefix} ==="
  python "$ROOT/stream_train.py" \
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
    --max-steps-per-epoch "$MAX_STEPS" \
    --min-tokens "$MIN_TOKENS" \
    --checkpoint-prefix "$prefix"
done

echo ""
echo "=== Evaluating checkpoints and selecting best ==="
python "$ROOT/scripts/evaluate_checkpoints.py" \
  --checkpoints-glob "${PREFIX_BASE}*_epoch*.pt" \
  --category "$CATEGORY" \
  --max-papers "$MAX_PAPERS" \
  --eval-steps 20 \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --min-tokens "$MIN_TOKENS" \
  --output "$ROOT/checkpoints/best_checkpoint_${PREFIX_BASE}.json"

echo ""
echo "Done. Best-checkpoint report:"
echo "  $ROOT/checkpoints/best_checkpoint_${PREFIX_BASE}.json"
