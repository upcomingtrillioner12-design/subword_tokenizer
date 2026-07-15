# Phase 4 Task 8 / Task 7 Follow-up: Embedding Comparison (Adversarial Subset)

Date: 2026-07-15

## Setup
- Dataset: 20-question balanced adversarial subset
  - misleading_context: 6
  - near_miss_distractor: 6
  - unanswerable: 8
- Corpus: 800 sampled chunks from dense metadata (fixed seed)
- Metrics: Precision@5, Precision@10

## Models Evaluated
1. all-mpnet-base-v2
2. hkunlp/instructor-base
3. allenai/scibert_scivocab_uncased

## Important Runtime Note
- `hkunlp/instructor-base` used fallback behavior because `InstructorEmbedding` package is not installed in the environment.
- So its run reflects fallback sentence-transformer behavior, not full instructor pipeline.

## Results
| Rank | Model | Precision@5 | Precision@10 |
|---|---|---:|---:|
| 1 | allenai/scibert_scivocab_uncased | 0.050 | 0.100 |
| 2 | all-mpnet-base-v2 | 0.050 | 0.050 |
| 3 | hkunlp/instructor-base* | 0.050 | 0.050 |

\* fallback mode due to missing InstructorEmbedding.

## Interpretation
- SciBERT outperformed baseline on this subset for retrieval hit rate at top-10 (+0.05 absolute).
- MPNet and Instructor(fallback) tied.
- Given adversarial data difficulty and lexical hit scoring, gains are modest but directional.

## Recommendation
- Use `allenai/scibert_scivocab_uncased` as Task 9 candidate embedding.
- If we want a fair instructor comparison, install `InstructorEmbedding` and rerun.

## Artifacts
- Raw JSON: results/rag_generation_eval/phase4_task8_embedding_eval_20260715_193001.json
- Script: scripts/evaluate_embeddings.py
- Subset: data/phase4_task8_adversarial_subset_20qa.json
