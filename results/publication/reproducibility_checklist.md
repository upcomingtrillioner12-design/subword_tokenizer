# Reproducibility Checklist

## Code / Config
- [x] Evaluation script: `scripts/run_rag_generation_evaluation.py`
- [x] Reranker pair collection: `scripts/collect_stem_preference_pairs.py`
- [x] Reranker fine-tuning: `scripts/finetune_cross_encoder.py`
- [x] Phase 5 configs available under `config/`

## Data
- [x] Base benchmark: `data/phase5_combined_100qa.json`
- [x] Hard benchmark extension: `data/phase5_combined_hard_500qa.json`
- [x] Preference labels (1000+): `data/stem_preference_pairs_1000.jsonl`

## Models / Checkpoints
- [x] Base SLM checkpoint path configured
- [x] LoRA best adapter path configured
- [x] Fine-tuned reranker v2: `checkpoints/cross_encoder_finetuned_task10_v2.pt`

## Results Artifacts
- [x] Seed-run outputs: `results/rag_generation_eval/seed_runs/`
- [x] Significance summary: `results/rag_generation_eval/seed_runs/seed_significance_summary.md`
- [x] Publication table: `results/publication/tables/phase5_seed_summary_table.csv`
- [x] Human eval template: `results/human_eval/human_eval_template.csv`
- [x] Human eval protocol summary: `results/human_eval/human_eval_summary.md`

## Notes
- Complete human annotation scoring is pending manual evaluator input.
- Full n=3 for every config should be maintained as additional runs are produced.
