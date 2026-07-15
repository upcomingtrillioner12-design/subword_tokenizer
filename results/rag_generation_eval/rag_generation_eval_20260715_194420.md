# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-15 19:44:20
- Questions: 2
- Elapsed: 35.65s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.0000 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.9373 |
| avg_factual_consistency | 0.9373 |
| avg_numeric_unit_consistency | 0.0000 |
| avg_uncertainty_score | 0.8000 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### adv_physics_001 (unknown)

- Query: What is Planck's constant value in SI units (J·s)?
- Expected: 6.62607015 × 10^-34
- Generated: 6.62607015 × 10^-34
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.895, unc=0.800

### adv_physics_002 (unknown)

- Query: What is the speed of sound in air at 20°C in m/s?
- Expected: 343
- Generated: 343
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.979, unc=0.800
