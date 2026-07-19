# Paper Outline (arXiv-ready)

## Title
Iterative Retrieval-Augmented Generation with Learned Reranking for Physics QA

## Abstract
- Problem: ceiling effects in small benchmark, need robust retrieval+generation validation.
- Method: LoRA SLM + hybrid retrieval + cross-encoder reranking + calibrated uncertainty + optional iteration.
- Results: strong exact-match, uncertainty calibration, and ablation deltas with reproducible artifacts.

## 1. Introduction
- Motivation for domain-specific RAG with reproducibility constraints.
- Contributions list.

## 2. Related Work
- RAG, rerankers, calibration, small-model domain adaptation.

## 3. System Architecture
- Tokenizer / LM / retrieval / reranking / uncertainty / iteration.
- Link: `doc/project_reference.md`.

## 4. Data and Benchmarks
- Base set: `data/phase5_combined_100qa.json`.
- Hard set: `data/phase5_combined_hard_500qa.json`.

## 5. Methods
- Retrieval fusion and reranking scoring strategy.
- Cross-encoder fine-tuning dataset: `data/stem_preference_pairs_1000.jsonl`.
- Checkpoint: `checkpoints/cross_encoder_finetuned_task10_v2.pt`.

## 6. Experimental Setup
- Configs: `config/phase5_*.yaml`.
- Seed-run aggregation: `results/rag_generation_eval/seed_runs/`.

## 7. Results
- Main metrics table from `results/publication/tables/phase5_seed_summary_table.csv`.
- Figures from `results/publication/figures/`.
- Human-eval protocol: `results/human_eval/human_eval_summary.md`.

## 8. Ablations
- Iteration OFF/ON and reranker original/fine-tuned comparisons.

## 9. Limitations
- Ceiling on base benchmark.
- Full n=3 for every config should remain an ongoing target.

## 10. Conclusion
- Practical reproducible stack and next steps.
