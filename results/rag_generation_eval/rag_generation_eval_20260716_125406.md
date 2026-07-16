# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-16 12:54:06
- Questions: 102
- Elapsed: 2501.36s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.2201 |
| avg_semantic_similarity | 1.0000 |
| avg_bertscore_f1 | 1.0000 |
| avg_entailment_score | 0.8656 |
| avg_factual_consistency | 0.8656 |
| avg_numeric_unit_consistency | 0.8288 |
| avg_uncertainty_score | 0.6019 |
| avg_iterations | 1.6471 |
| iterative_trigger_rate | 0.6471 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### physics_001 (stem)

- Query: What is the de Broglie wavelength relation for a particle?
- Expected: lambda equals h over p
- Generated: lambda equals h over p
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.936, unc=0.320, iter=1

### physics_002 (stem)

- Query: According to the Born rule, what does the square of the wavefunction magnitude represent?
- Expected: probability density
- Generated: probability density
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.884, unc=0.800, iter=2

### physics_003 (stem)

- Query: Which commutator corresponds to position and momentum in one dimension?
- Expected: x and p commute to i hbar
- Generated: x and p commute to i hbar
- Metrics: f1=1.000, contains=1.0, faith=0.429, sem=1.000, entail=0.847, unc=0.229, iter=1

### physics_004 (stem)

- Query: What principle limits simultaneous precision of position and momentum?
- Expected: heisenberg uncertainty principle
- Generated: heisenberg uncertainty principle
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.915, unc=0.533, iter=2

### physics_005 (stem)

- Query: In special relativity, what happens to time for a fast-moving clock?
- Expected: time dilation
- Generated: time dilation
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.877, unc=0.400, iter=1

### physics_006 (stem)

- Query: What geometric object is curved by mass-energy in general relativity?
- Expected: spacetime
- Generated: spacetime
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.943, unc=0.800, iter=2

### physics_007 (stem)

- Query: What boundary of a black hole marks no return for light?
- Expected: event horizon
- Generated: event horizon
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.866, unc=0.800, iter=2

### physics_008 (stem)

- Query: Which law gives divergence of electric field in terms of charge density?
- Expected: gauss law
- Generated: gauss law
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.973, unc=0.400, iter=1

### physics_009 (stem)

- Query: What is the force on charge q moving with velocity v in fields E and B called?
- Expected: lorentz force
- Generated: lorentz force
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.977, unc=0.400, iter=1

### physics_010 (stem)

- Query: State the trend of entropy in an isolated system according to the second law.
- Expected: entropy increases
- Generated: entropy increases
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.950, unc=0.800, iter=2

### chemistry_001 (stem)

- Query: How many electrons can occupy an orbital with quantum number n equals 3?
- Expected: eighteen electrons
- Generated: eighteen electrons
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.523, unc=0.800, iter=2

### chemistry_002 (stem)

- Query: What does the principal quantum number n determine in an atom?
- Expected: electron shell size and energy
- Generated: electron shell size and energy
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.314, unc=0.640, iter=2

### chemistry_003 (stem)

- Query: Which type of bond involves sharing of electron pairs between atoms?
- Expected: covalent bond
- Generated: covalent bond
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.895, unc=0.800, iter=2

### chemistry_004 (stem)

- Query: What is the geometry of a carbon atom with four single bonds?
- Expected: tetrahedral
- Generated: tetrahedral
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.871, unc=0.800, iter=2

### chemistry_005 (stem)

- Query: What substance is produced when an acid reacts with a base?
- Expected: salt and water
- Generated: salt and water
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.955, unc=0.267, iter=1

### chemistry_006 (stem)

- Query: In an exothermic reaction, what is released to the surroundings?
- Expected: heat energy
- Generated: heat energy
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.869, unc=0.400, iter=1

### chemistry_007 (stem)

- Query: What factor increases the rate of a chemical reaction by lowering activation energy?
- Expected: catalyst
- Generated: catalyst
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.840, unc=0.800, iter=2

### chemistry_008 (stem)

- Query: What is the pH of a neutral aqueous solution at standard conditions?
- Expected: seven
- Generated: seven
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.912, unc=0.000, iter=1

### chemistry_009 (stem)

- Query: In a reversible reaction at equilibrium, what is the relationship between forward and reverse rates?
- Expected: they are equal
- Generated: they are equal
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.783, unc=0.267, iter=1

### chemistry_010 (stem)

- Query: What is the term for loss of electrons in a redox reaction?
- Expected: oxidation
- Generated: oxidation
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.944, unc=0.800, iter=2

### biology_001 (stem)

- Query: What is the basic unit of life?
- Expected: cell
- Generated: cell
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.937, unc=0.000, iter=2

### biology_002 (stem)

- Query: Which organelle is responsible for producing energy in a cell?
- Expected: mitochondrion
- Generated: mitochondrion
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.916, unc=0.800, iter=2

### biology_003 (stem)

- Query: What is the molecule that carries genetic information in most organisms?
- Expected: deoxyribonucleic acid
- Generated: deoxyribonucleic acid
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.639, unc=0.800, iter=2

### biology_004 (stem)

- Query: How many DNA base pairs are in the human genome approximately?
- Expected: three billion base pairs
- Generated: three billion base pairs
- Metrics: f1=1.000, contains=1.0, faith=0.750, sem=1.000, entail=0.907, unc=0.200, iter=1

### biology_005 (stem)

- Query: What is the primary mechanism of evolution proposed by Darwin?
- Expected: natural selection
- Generated: natural selection
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.849, unc=0.400, iter=1

### biology_006 (stem)

- Query: What term describes the role of an organism in its environment?
- Expected: ecological niche
- Generated: ecological niche
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.783, unc=0.800, iter=2

### biology_007 (stem)

- Query: What do plants use to convert light energy into chemical energy?
- Expected: photosynthesis
- Generated: photosynthesis
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.875, unc=0.800, iter=2

### biology_008 (stem)

- Query: What type of white blood cell directly attacks infected cells?
- Expected: cytotoxic t lymphocyte
- Generated: cytotoxic t lymphocyte
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.964, unc=0.533, iter=1

### biology_009 (stem)

- Query: What is the kingdom of organisms that includes humans?
- Expected: animalia
- Generated: animalia
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.870, unc=0.800, iter=2

### biology_010 (stem)

- Query: How many nucleotides are in a codon?
- Expected: three nucleotides
- Generated: three nucleotides
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.956, unc=0.400, iter=2

### mathematics_001 (stem)

- Query: What is the solution to the equation two x plus three equals seven?
- Expected: x equals two
- Generated: x equals two
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.421, unc=0.267, iter=1

### mathematics_002 (stem)

- Query: What is the standard form of a linear equation in two variables?
- Expected: a x plus b y equals c
- Generated: a x plus b y equals c
- Metrics: f1=1.000, contains=1.0, faith=0.143, sem=1.000, entail=0.920, unc=0.229, iter=1

### mathematics_003 (stem)

- Query: What is the sum of angles in a triangle?
- Expected: one hundred eighty degrees
- Generated: one hundred eighty degrees
- Metrics: f1=1.000, contains=1.0, faith=0.250, sem=1.000, entail=0.620, unc=0.600, iter=2

### mathematics_004 (stem)

- Query: What is the area formula for a circle?
- Expected: pi r squared
- Generated: pi r squared
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.901, unc=0.267, iter=1

### mathematics_005 (stem)

- Query: What is the derivative of x squared?
- Expected: two x
- Generated: two x
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.962, unc=0.000, iter=1

### mathematics_006 (stem)

- Query: What does the definite integral represent geometrically?
- Expected: area under the curve
- Generated: area under the curve
- Metrics: f1=1.000, contains=1.0, faith=0.250, sem=1.000, entail=0.862, unc=0.600, iter=2

### mathematics_007 (stem)

- Query: What does the standard deviation measure?
- Expected: spread of data from the mean
- Generated: spread of data from the mean
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.738, unc=0.267, iter=1

### mathematics_008 (stem)

- Query: What is the probability of rolling a six on a fair die?
- Expected: one sixth
- Generated: one sixth
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.950, unc=0.400, iter=1

### mathematics_009 (stem)

- Query: What does matrix multiplication compute between two vectors?
- Expected: dot product
- Generated: dot product
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.729, unc=0.800, iter=2

### mathematics_010 (stem)

- Query: What is a prime number?
- Expected: number divisible only by one and itself
- Generated: number divisible only by one and itself
- Metrics: f1=1.000, contains=1.0, faith=0.714, sem=1.000, entail=0.960, unc=0.229, iter=1

### earth_science_001 (stem)

- Query: What are the three main types of rocks?
- Expected: igneous sedimentary metamorphic
- Generated: igneous sedimentary metamorphic
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.973, unc=0.800, iter=2

### earth_science_002 (stem)

- Query: What scale measures the hardness of minerals?
- Expected: mohs hardness scale
- Generated: mohs hardness scale
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.936, unc=0.267, iter=1

### earth_science_003 (stem)

- Query: What is the most abundant mineral in Earth's crust?
- Expected: feldspar
- Generated: feldspar
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.853, unc=0.800, iter=2

### earth_science_004 (stem)

- Query: What weather system has the lowest atmospheric pressure?
- Expected: hurricane
- Generated: hurricane
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.887, unc=0.800, iter=2

### earth_science_005 (stem)

- Query: What causes ocean tides?
- Expected: gravitational force of the moon
- Generated: gravitational force of the moon
- Metrics: f1=1.000, contains=1.0, faith=0.400, sem=1.000, entail=0.972, unc=0.480, iter=1

### earth_science_006 (stem)

- Query: What is the movement of tectonic plates called?
- Expected: plate tectonics
- Generated: plate tectonics
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.901, unc=0.400, iter=1

### earth_science_007 (stem)

- Query: What gas is primarily responsible for current global warming?
- Expected: carbon dioxide
- Generated: carbon dioxide
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.647, unc=0.800, iter=2

### earth_science_008 (stem)

- Query: What is the layer of Earth between the crust and outer core?
- Expected: mantle
- Generated: mantle
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.591, unc=0.800, iter=2

### earth_science_009 (stem)

- Query: What scale measures the energy released by earthquakes?
- Expected: richter scale
- Generated: richter scale
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.958, unc=0.400, iter=1

### earth_science_010 (stem)

- Query: What is the closest star to Earth besides the Sun?
- Expected: proxima centauri
- Generated: proxima centauri
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.931, unc=0.800, iter=2

### computer_science_001 (stem)

- Query: What is the time complexity of binary search?
- Expected: logarithmic time
- Generated: logarithmic time
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.936, unc=0.400, iter=1

### computer_science_002 (stem)

- Query: What data structure uses LIFO principle?
- Expected: stack
- Generated: stack
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.942, unc=0.800, iter=2

### computer_science_003 (stem)

- Query: What does SQL stand for?
- Expected: structured query language
- Generated: structured query language
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.950, unc=0.800, iter=2

### computer_science_004 (stem)

- Query: What is a function in programming?
- Expected: reusable block of code
- Generated: reusable block of code
- Metrics: f1=1.000, contains=1.0, faith=0.250, sem=1.000, entail=0.783, unc=0.600, iter=2

### computer_science_005 (stem)

- Query: What does TCP stand for?
- Expected: transmission control protocol
- Generated: transmission control protocol
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.753, unc=0.800, iter=2

### computer_science_006 (stem)

- Query: What type of encryption uses two keys?
- Expected: asymmetric encryption
- Generated: asymmetric encryption
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.900, unc=0.800, iter=2

### computer_science_007 (stem)

- Query: What is the Church-Turing thesis about?
- Expected: computability of algorithms
- Generated: computability of algorithms
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.926, unc=0.533, iter=1

### computer_science_008 (stem)

- Query: What is supervised learning?
- Expected: learning from labeled data
- Generated: learning from labeled data
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.846, unc=0.400, iter=1

### computer_science_009 (stem)

- Query: What does AI stand for?
- Expected: artificial intelligence
- Generated: artificial intelligence
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.950, unc=0.800, iter=2

### computer_science_010 (stem)

- Query: What is a deadlock in operating systems?
- Expected: circular wait for resources
- Generated: circular wait for resources
- Metrics: f1=1.000, contains=1.0, faith=0.250, sem=1.000, entail=0.896, unc=0.600, iter=2

### adv_physics_001 (adversarial)

- Query: What is Planck's constant value in SI units (J·s)?
- Expected: 6.62607015 × 10^-34
- Generated: 6.62607015 × 10^-34
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.859, unc=0.800, iter=2

### adv_physics_002 (adversarial)

- Query: What is the speed of sound in air at 20°C in m/s?
- Expected: 343
- Generated: 343
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.973, unc=0.800, iter=2

### adv_chemistry_001 (adversarial)

- Query: What is the molecular weight of ethanol (C2H5OH) in g/mol?
- Expected: 46.07
- Generated: 46.07
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.886, unc=0.800, iter=2

### adv_chemistry_002 (adversarial)

- Query: What is the activation energy for the SN2 reaction of chloromethane with hydroxide ion in aprotic solvent?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.923, unc=0.800, iter=2

### adv_biology_001 (adversarial)

- Query: How many nucleotides are in a complete turn of the DNA double helix?
- Expected: 10
- Generated: 10
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.830, unc=0.800, iter=2

### adv_biology_002 (adversarial)

- Query: What is the current population of Arctic foxes in the Siberian tundra as of 2024?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.930, unc=0.800, iter=2

### adv_math_001 (adversarial)

- Query: What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?
- Expected: 3x^2 + 4x - 5
- Generated: 3x^2 + 4x - 5
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.910, unc=0.480, iter=1

### adv_math_002 (adversarial)

- Query: What is the determinant of the matrix [[2, 3], [4, 5]]?
- Expected: -2
- Generated: -2
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.953, unc=0.800, iter=2

### adv_earth_science_001 (adversarial)

- Query: What is the exact mineral composition percentage of the deepest crustal layer under the Mariana Trench?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.905, unc=0.800, iter=2

### adv_earth_science_002 (adversarial)

- Query: What is the highest temperature ever recorded on Earth's surface?
- Expected: 54.0°C
- Generated: 54.0°C
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.960, unc=0.400, iter=1

### adv_cs_001 (adversarial)

- Query: What is the time complexity of merge sort in the worst case?
- Expected: O(n log n)
- Generated: O(n log n)
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.853, unc=0.200, iter=1

### adv_cs_002 (adversarial)

- Query: What is the optimal index strategy for a specific query on tables with 5.2B rows and cardinality distribution X?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.890, unc=0.800, iter=2

### adv_physics_003 (adversarial)

- Query: What is the magnetic field produced by a long straight wire carrying current I at distance r?
- Expected: B = μ₀I / (2πr)
- Generated: B = μ₀I / (2πr)
- Metrics: f1=1.000, contains=1.0, faith=0.750, sem=1.000, entail=0.875, unc=0.160, iter=1

### adv_physics_004 (adversarial)

- Query: What is the gravitational acceleration at Earth's surface?
- Expected: 9.81 m/s²
- Generated: 9.81 m/s²
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.944, unc=0.800, iter=2

### adv_chemistry_003 (adversarial)

- Query: What is the exact standard free energy of formation for a hypothetical compound XY₄ at 37°C?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.886, unc=0.800, iter=2

### adv_chemistry_004 (adversarial)

- Query: What is the pH of a 0.1 M HCl solution?
- Expected: 1
- Generated: 1
- Metrics: f1=1.000, contains=1.0, faith=1.000, sem=1.000, entail=0.812, unc=0.000, iter=1

### adv_biology_003 (adversarial)

- Query: What percentage of the human genome is protein-coding?
- Expected: 1.5%
- Generated: 1.5%
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.919, unc=0.800, iter=2

### adv_biology_004 (adversarial)

- Query: What is the exact number of mitochondria in a human liver cell from a specific individual under specific conditions?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.974, unc=0.800, iter=2

### adv_math_003 (adversarial)

- Query: What is the next prime number after 1000?
- Expected: 1009
- Generated: 1009
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.918, unc=0.800, iter=2

### adv_math_004 (adversarial)

- Query: What is the exact probability that a randomly selected integer has a specific Diophantine property?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.960, unc=0.800, iter=2

### adv_earth_science_003 (adversarial)

- Query: What is the current atmospheric CO₂ concentration in ppm (2024)?
- Expected: 420
- Generated: 420
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.980, unc=0.800, iter=2

### adv_earth_science_004 (adversarial)

- Query: What is the exact current velocity at a specific point in the Gulf Stream on July 15, 2026?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.724, unc=0.800, iter=2

### adv_cs_003 (adversarial)

- Query: What is the time complexity of finding an element in a balanced binary search tree?
- Expected: O(log n)
- Generated: O(log n)
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.732, unc=0.267, iter=1

### adv_cs_004 (adversarial)

- Query: What is the optimal routing algorithm for a specific network topology with dynamic packet loss?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.954, unc=0.800, iter=2

### adv_physics_005 (adversarial)

- Query: What is the speed of light in vacuum?
- Expected: 3 × 10^8 m/s
- Generated: 3 × 10^8 m/s
- Metrics: f1=1.000, contains=1.0, faith=0.400, sem=1.000, entail=0.931, unc=0.600, iter=2

### adv_chemistry_005 (adversarial)

- Query: What is the exact orbital overlap integral for a specific molecular geometry?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.920, unc=0.800, iter=2

### adv_biology_005 (adversarial)

- Query: Approximately how many species exist on Earth?
- Expected: 8.7 million
- Generated: 8.7 million
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.968, unc=0.800, iter=2

### adv_earth_science_005 (adversarial)

- Query: What is the exact epicenter location of the next significant earthquake?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.964, unc=0.800, iter=2

### adv_math_005 (adversarial)

- Query: What is the circumference of a circle with radius 5?
- Expected: 10π
- Generated: 10π
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.945, unc=0.800, iter=2

### adv_cs_005 (adversarial)

- Query: What is the purpose of the softmax function in neural networks?
- Expected: Convert logits to probability distribution
- Generated: Convert logits to probability distribution
- Metrics: f1=1.000, contains=1.0, faith=0.200, sem=1.000, entail=0.775, unc=0.640, iter=2

### adv_physics_006 (adversarial)

- Query: What is the exact refractive index of a specific organic polymer blend at a particular wavelength?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.897, unc=0.800, iter=2

### adv_chemistry_006 (adversarial)

- Query: What is the concentration of a solution if 25 mL of 0.1 M HCl is diluted to 250 mL?
- Expected: 0.01 M
- Generated: 0.01 M
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.661, unc=0.400, iter=1

### adv_biology_006 (adversarial)

- Query: How many antibody isotypes does the human immune system produce?
- Expected: 5
- Generated: 5
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.911, unc=0.800, iter=2

### adv_earth_science_006 (adversarial)

- Query: What is the exact crystal structure parameter for a newly discovered mineral?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.857, unc=0.800, iter=2

### adv_math_006 (adversarial)

- Query: In a normal distribution, what percentage of data falls within 2 standard deviations?
- Expected: 95.45%
- Generated: 95.45%
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.965, unc=0.800, iter=2

### adv_cs_006 (adversarial)

- Query: What is the optimal encryption key length for protecting data until year 2100?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.900, unc=0.800, iter=2

### adv_physics_007 (adversarial)

- Query: What is the Reynolds number for water flowing through a pipe at standard conditions?
- Expected: depends on velocity and diameter
- Generated: depends on velocity and diameter
- Metrics: f1=1.000, contains=1.0, faith=0.600, sem=1.000, entail=0.922, unc=0.320, iter=1

### adv_chemistry_007 (adversarial)

- Query: What is the exact stability constant for a specific metal-ligand complex at pH 5.2?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.697, unc=0.800, iter=2

### adv_biology_007 (adversarial)

- Query: What is the Km value of the enzyme glucose oxidase for its substrate?
- Expected: ~1.4 mM
- Generated: ~1.4 mM
- Metrics: f1=1.000, contains=1.0, faith=0.667, sem=1.000, entail=0.403, unc=0.400, iter=1

### adv_earth_science_007 (adversarial)

- Query: What is the age of the Earth in years?
- Expected: 4.54 billion
- Generated: 4.54 billion
- Metrics: f1=1.000, contains=1.0, faith=0.333, sem=1.000, entail=0.942, unc=0.800, iter=2

### adv_math_007 (adversarial)

- Query: What is the cardinality of the continuum hypothesis solution set?
- Expected: CANNOT_BE_ANSWERED
- Generated: CANNOT_BE_ANSWERED
- Metrics: f1=1.000, contains=1.0, faith=0.000, sem=1.000, entail=0.851, unc=0.800, iter=2

### adv_cs_007 (adversarial)

- Query: What is the Chomsky hierarchy level of context-free languages?
- Expected: Type 2
- Generated: Type 2
- Metrics: f1=1.000, contains=1.0, faith=0.500, sem=1.000, entail=0.866, unc=0.400, iter=1
