#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[phase2] Root: $ROOT_DIR"
source venv/bin/activate

CONFIG_PATH="config/phase2_lora_config.yaml"

echo "[phase2] Step 1/3: prepare offline corpus"
python scripts/prepare_offline_corpus.py \
  --category physics \
  --max-papers 50000 \
  --seq-len 256 \
  --output-dir data/offline_physics \
  --tokenizer-model subword_tokenizer/model_32k.json

echo "[phase2] Step 2/3: LoRA fine-tune"
python scripts/phase2_lora_finetune.py --config "$CONFIG_PATH"

echo "[phase2] Step 3/3: evaluate LoRA adapters"
python scripts/evaluate_lora_checkpoints.py \
  --config "$CONFIG_PATH" \
  --checkpoints-dir checkpoints/phase2_lora \
  --pattern "lora_adapter_*.pt" \
  --eval-split val \
  --eval-steps 200 \
  --output checkpoints/phase2_lora/phase2_evaluation_report.json

echo "[phase2] Completed."
