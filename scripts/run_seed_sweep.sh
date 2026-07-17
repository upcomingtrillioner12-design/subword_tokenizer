#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

CONFIGS=(
  "config/phase5_full_integration_eval.yaml"
  "config/phase5_finetuned_cross_encoder_eval.yaml"
  "config/phase5_ablation_no_iter_original.yaml"
  "config/phase5_ablation_no_iter_finetuned.yaml"
)
SEEDS=(1 2 3)
OUTDIR="results/rag_generation_eval/seed_runs"
mkdir -p "$OUTDIR"

for cfg in "${CONFIGS[@]}"; do
  cfg_name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    out_file="$OUTDIR/${cfg_name}_seed${seed}.json"
    echo "Running: $cfg_name, seed=$seed -> $out_file"
    python3 scripts/run_rag_generation_evaluation.py \
      --config "$cfg" \
      --seed "$seed" \
      --output "$out_file" \
      || echo "⚠️  Run failed: $cfg_name seed=$seed (check script args match your actual CLI)"
  done
done

echo "✅ Seed sweep complete. Check $OUTDIR for real result files."
