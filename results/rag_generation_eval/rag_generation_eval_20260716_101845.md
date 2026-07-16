# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-16 10:18:45
- Questions: 20
- Elapsed: 332.07s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.1667 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.8984 |
| avg_factual_consistency | 0.8984 |
| avg_numeric_unit_consistency | 0.5500 |
| avg_uncertainty_score | 0.7020 |
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

### adv_biology_001 (unknown)

- Query: How many nucleotides are in a complete turn of the DNA double helix?
- Expected: 10
- Generated: 10
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.719, unc=0.800

### adv_math_002 (unknown)

- Query: What is the determinant of the matrix [[2, 3], [4, 5]]?
- Expected: -2
- Generated: -2
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.953, unc=0.800

### adv_cs_001 (unknown)

- Query: What is the time complexity of merge sort in the worst case?
- Expected: O(n log n)
- Generated: O(n log n)
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.853, unc=0.200

### adv_physics_003 (unknown)

- Query: What is the magnetic field produced by a long straight wire carrying current I at distance r?
- Expected: B = μ₀I / (2πr)
- Generated: B = μ₀I / (2πr)
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.903, unc=0.160

### adv_chemistry_001 (unknown)

- Query: What is the molecular weight of ethanol (C2H5OH) in g/mol?
- Expected: 46.07
- Generated: 46.07
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.935, unc=0.800

### adv_math_001 (unknown)

- Query: What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?
- Expected: 3x^2 + 4x - 5
- Generated: 3x^2 + 4x - 5
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.910, unc=0.480

### adv_earth_science_002 (unknown)

- Query: What is the highest temperature ever recorded on Earth's surface?
- Expected: 54.0°C
- Generated: 54.0°C
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.966, unc=0.400

### adv_physics_004 (unknown)

- Query: What is the gravitational acceleration at Earth's surface?
- Expected: 9.81 m/s²
- Generated: 9.81 m/s²
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.962, unc=0.800

### adv_biology_003 (unknown)

- Query: What percentage of the human genome is protein-coding?
- Expected: 1.5%
- Generated: 1.5%
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.887, unc=0.800

### adv_earth_science_003 (unknown)

- Query: What is the current atmospheric CO₂ concentration in ppm (2024)?
- Expected: 420
- Generated: 420
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.970, unc=0.800

### adv_chemistry_002 (unknown)

- Query: What is the activation energy for the SN2 reaction of chloromethane with hydroxide ion in aprotic solvent?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.549, unc=0.800

### adv_biology_002 (unknown)

- Query: What is the current population of Arctic foxes in the Siberian tundra as of 2024?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.943, unc=0.800

### adv_earth_science_001 (unknown)

- Query: What is the exact mineral composition percentage of the deepest crustal layer under the Mariana Trench?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.938, unc=0.800

### adv_cs_002 (unknown)

- Query: What is the optimal index strategy for a specific query on tables with 5.2B rows and cardinality distribution X?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.883, unc=0.800

### adv_chemistry_003 (unknown)

- Query: What is the exact standard free energy of formation for a hypothetical compound XY₄ at 37°C?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.875, unc=0.800

### adv_biology_004 (unknown)

- Query: What is the exact number of mitochondria in a human liver cell from a specific individual under specific conditions?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.941, unc=0.800

### adv_math_004 (unknown)

- Query: What is the exact probability that a randomly selected integer has a specific Diophantine property?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.960, unc=0.800

### adv_earth_science_004 (unknown)

- Query: What is the exact current velocity at a specific point in the Gulf Stream on July 15, 2026?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.948, unc=0.800
