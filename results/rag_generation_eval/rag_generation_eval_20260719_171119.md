# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-19 17:11:19
- Questions: 30
- Elapsed: 395.02s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.2165 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.8698 |
| avg_factual_consistency | 0.8698 |
| avg_numeric_unit_consistency | 1.0000 |
| avg_uncertainty_score | 0.5952 |
| avg_iterations | 1.0000 |
| iterative_trigger_rate | 0.0000 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### physics_001 (stem)

- Query: What is the de Broglie wavelength relation for a particle?
- Expected: lambda equals h over p
- Generated: lambda equals h over p
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.900, unc=0.320, iter=1

### physics_002 (stem)

- Query: According to the Born rule, what does the square of the wavefunction magnitude represent?
- Expected: probability density
- Generated: probability density
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.877, unc=0.800, iter=1

### physics_003 (stem)

- Query: Which commutator corresponds to position and momentum in one dimension?
- Expected: x and p commute to i hbar
- Generated: x and p commute to i hbar
- Metrics: f1=1.000, contains=1.0, faith=0.429, sem=1.000, entail=0.847, unc=0.229, iter=1

### physics_004 (stem)

- Query: What principle limits simultaneous precision of position and momentum?
- Expected: heisenberg uncertainty principle
- Generated: heisenberg uncertainty principle
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.981, unc=0.800, iter=1

### physics_005 (stem)

- Query: In special relativity, what happens to time for a fast-moving clock?
- Expected: time dilation
- Generated: time dilation
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.877, unc=0.400, iter=1

### physics_006 (stem)

- Query: What geometric object is curved by mass-energy in general relativity?
- Expected: spacetime
- Generated: spacetime
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.936, unc=0.800, iter=1

### physics_007 (stem)

- Query: What boundary of a black hole marks no return for light?
- Expected: event horizon
- Generated: event horizon
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.853, unc=0.800, iter=1

### physics_008 (stem)

- Query: Which law gives divergence of electric field in terms of charge density?
- Expected: gauss law
- Generated: gauss law
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.982, unc=0.400, iter=1

### physics_009 (stem)

- Query: What is the force on charge q moving with velocity v in fields E and B called?
- Expected: lorentz force
- Generated: lorentz force
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.977, unc=0.400, iter=1

### physics_010 (stem)

- Query: State the trend of entropy in an isolated system according to the second law.
- Expected: entropy increases
- Generated: entropy increases
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.739, unc=0.800, iter=1

### chemistry_001 (stem)

- Query: How many electrons can occupy an orbital with quantum number n equals 3?
- Expected: eighteen electrons
- Generated: eighteen electrons
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.899, unc=0.800, iter=1

### chemistry_002 (stem)

- Query: What does the principal quantum number n determine in an atom?
- Expected: electron shell size and energy
- Generated: electron shell size and energy
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.315, unc=0.640, iter=1

### chemistry_003 (stem)

- Query: Which type of bond involves sharing of electron pairs between atoms?
- Expected: covalent bond
- Generated: covalent bond
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.922, unc=0.400, iter=1

### chemistry_004 (stem)

- Query: What is the geometry of a carbon atom with four single bonds?
- Expected: tetrahedral
- Generated: tetrahedral
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.864, unc=0.800, iter=1

### chemistry_005 (stem)

- Query: What substance is produced when an acid reacts with a base?
- Expected: salt and water
- Generated: salt and water
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.982, unc=0.267, iter=1

### chemistry_006 (stem)

- Query: In an exothermic reaction, what is released to the surroundings?
- Expected: heat energy
- Generated: heat energy
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.869, unc=0.400, iter=1

### chemistry_007 (stem)

- Query: What factor increases the rate of a chemical reaction by lowering activation energy?
- Expected: catalyst
- Generated: catalyst
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.945, unc=0.800, iter=1

### chemistry_008 (stem)

- Query: What is the pH of a neutral aqueous solution at standard conditions?
- Expected: seven
- Generated: seven
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.942, unc=0.000, iter=1

### chemistry_009 (stem)

- Query: In a reversible reaction at equilibrium, what is the relationship between forward and reverse rates?
- Expected: they are equal
- Generated: they are equal
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.783, unc=0.267, iter=1

### chemistry_010 (stem)

- Query: What is the term for loss of electrons in a redox reaction?
- Expected: oxidation
- Generated: oxidation
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.918, unc=0.800, iter=1

### biology_001 (stem)

- Query: What is the basic unit of life?
- Expected: cell
- Generated: cell
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.958, unc=0.800, iter=1

### biology_002 (stem)

- Query: Which organelle is responsible for producing energy in a cell?
- Expected: mitochondrion
- Generated: mitochondrion
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.863, unc=0.800, iter=1

### biology_003 (stem)

- Query: What is the molecule that carries genetic information in most organisms?
- Expected: deoxyribonucleic acid
- Generated: deoxyribonucleic acid
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.638, unc=0.800, iter=1

### biology_004 (stem)

- Query: How many DNA base pairs are in the human genome approximately?
- Expected: three billion base pairs
- Generated: three billion base pairs
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.949, unc=0.400, iter=1

### biology_005 (stem)

- Query: What is the primary mechanism of evolution proposed by Darwin?
- Expected: natural selection
- Generated: natural selection
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.849, unc=0.400, iter=1

### biology_006 (stem)

- Query: What term describes the role of an organism in its environment?
- Expected: ecological niche
- Generated: ecological niche
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.867, unc=0.800, iter=1

### biology_007 (stem)

- Query: What do plants use to convert light energy into chemical energy?
- Expected: photosynthesis
- Generated: photosynthesis
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.932, unc=0.800, iter=1

### biology_008 (stem)

- Query: What type of white blood cell directly attacks infected cells?
- Expected: cytotoxic t lymphocyte
- Generated: cytotoxic t lymphocyte
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.968, unc=0.533, iter=1

### biology_009 (stem)

- Query: What is the kingdom of organisms that includes humans?
- Expected: animalia
- Generated: animalia
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.721, unc=0.800, iter=1

### biology_010 (stem)

- Query: How many nucleotides are in a codon?
- Expected: three nucleotides
- Generated: three nucleotides
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.941, unc=0.800, iter=1
