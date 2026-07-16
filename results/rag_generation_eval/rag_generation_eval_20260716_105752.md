# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-16 10:57:52
- Questions: 5
- Elapsed: 80.56s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.2924 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.8842 |
| avg_factual_consistency | 0.8842 |
| avg_numeric_unit_consistency | 1.0000 |
| avg_uncertainty_score | 0.4564 |
| avg_iterations | 1.2000 |
| iterative_trigger_rate | 0.2000 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### physics_001 (stem)

- Query: What is the de Broglie wavelength relation for a particle?
- Expected: lambda equals h over p
- Generated: lambda equals h over p
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.967, unc=0.320, iter=1

### physics_002 (stem)

- Query: According to the Born rule, what does the square of the wavefunction magnitude represent?
- Expected: probability density
- Generated: probability density
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.883, unc=0.800, iter=2

### physics_003 (stem)

- Query: Which commutator corresponds to position and momentum in one dimension?
- Expected: x and p commute to i hbar
- Generated: x and p commute to i hbar
- Metrics: f1=1.000, contains=1.0, faith=0.429, sem=1.000, entail=0.847, unc=0.229, iter=1

### physics_004 (stem)

- Query: What principle limits simultaneous precision of position and momentum?
- Expected: heisenberg uncertainty principle
- Generated: heisenberg uncertainty principle
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.979, unc=0.533, iter=1

### physics_005 (stem)

- Query: In special relativity, what happens to time for a fast-moving clock?
- Expected: time dilation
- Generated: time dilation
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.746, unc=0.400, iter=1
