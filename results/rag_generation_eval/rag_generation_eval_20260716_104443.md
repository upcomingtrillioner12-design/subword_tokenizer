# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-16 10:44:42
- Questions: 10
- Elapsed: 268.67s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.2833 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.8930 |
| avg_factual_consistency | 0.8930 |
| avg_numeric_unit_consistency | 0.3000 |
| avg_uncertainty_score | 0.6040 |
| avg_iterations | 1.6000 |
| iterative_trigger_rate | 0.6000 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### adv_physics_001 (unknown)

- Query: What is Planck's constant value in SI units (J·s)?
- Expected: 6.62607015 × 10^-34
- Generated: 6.62607015 × 10^-34
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.926, unc=0.800, iter=2

### adv_physics_002 (unknown)

- Query: What is the speed of sound in air at 20°C in m/s?
- Expected: 343
- Generated: 343
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.957, unc=0.800, iter=2

### adv_biology_001 (unknown)

- Query: How many nucleotides are in a complete turn of the DNA double helix?
- Expected: 10
- Generated: 10
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.719, unc=0.800, iter=2

### adv_math_002 (unknown)

- Query: What is the determinant of the matrix [[2, 3], [4, 5]]?
- Expected: -2
- Generated: -2
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.869, unc=0.800, iter=2

### adv_cs_001 (unknown)

- Query: What is the time complexity of merge sort in the worst case?
- Expected: O(n log n)
- Generated: O(n log n)
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.853, unc=0.200, iter=1

### adv_physics_003 (unknown)

- Query: What is the magnetic field produced by a long straight wire carrying current I at distance r?
- Expected: B = μ₀I / (2πr)
- Generated: B = μ₀I / (2πr)
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.903, unc=0.160, iter=1

### adv_chemistry_001 (unknown)

- Query: What is the molecular weight of ethanol (C2H5OH) in g/mol?
- Expected: 46.07
- Generated: 46.07
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.886, unc=0.800, iter=2

### adv_math_001 (unknown)

- Query: What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?
- Expected: 3x^2 + 4x - 5
- Generated: 3x^2 + 4x - 5
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.910, unc=0.480, iter=1

### adv_earth_science_002 (unknown)

- Query: What is the highest temperature ever recorded on Earth's surface?
- Expected: 54.0°C
- Generated: 54.0°C
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.966, unc=0.400, iter=1

### adv_physics_004 (unknown)

- Query: What is the gravitational acceleration at Earth's surface?
- Expected: 9.81 m/s²
- Generated: 9.81 m/s²
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.940, unc=0.800, iter=2
