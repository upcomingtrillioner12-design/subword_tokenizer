# Phase 4 Task 5 Scaling + Faithfulness Report

## Corpus Scaling Comparison (MC-likelihood)

| Metric | Physics corpus + rerank | General corpus + advanced rerank | General corpus no rerank |
|---|---:|---:|---:|
| mc_exact_rate | 1.0000 | 1.0000 | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 | 1.0000 | 1.0000 |
| avg_token_f1 | 1.0000 | 1.0000 | 1.0000 |
| avg_contains_expected | 1.0000 | 1.0000 | 1.0000 |
| avg_faithfulness | 0.3060 | 0.1581 | 0.1064 |

## Generation Faithfulness Tradeoff (General corpus, generation mode)

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.0064 |
| avg_contains_expected | 0.0500 |
| avg_faithfulness | 0.9830 |
| mc_exact_rate | 0.0000 |
| mc_semantic_or_better_rate | 0.0000 |

### Notes
- MC-likelihood evaluation remains perfect (all 20/20) at general-corpus scale, with and without reranking.
- Advanced reranking improves interpretability (component scores) but does not change MC ranking on this dataset.
- Strict grounding settings in generation mode dramatically increase context faithfulness but reduce direct-answer accuracy.