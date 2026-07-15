# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-15 16:57:49
- Questions: 60
- Elapsed: 764.63s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.2250 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### physics_001 (unknown)

- Query: What is the de Broglie wavelength relation for a particle?
- Expected: lambda equals h over p
- Generated: lambda equals h over p
- Metrics: f1=1.000, contains=1.0, faith=0.200

### physics_002 (unknown)

- Query: According to the Born rule, what does the square of the wavefunction magnitude represent?
- Expected: probability density
- Generated: probability density
- Metrics: f1=1.000, contains=1.0, faith=0.000

### physics_003 (unknown)

- Query: Which commutator corresponds to position and momentum in one dimension?
- Expected: x and p commute to i hbar
- Generated: x and p commute to i hbar
- Metrics: f1=1.000, contains=1.0, faith=0.429

### physics_004 (unknown)

- Query: What principle limits simultaneous precision of position and momentum?
- Expected: heisenberg uncertainty principle
- Generated: heisenberg uncertainty principle
- Metrics: f1=1.000, contains=1.0, faith=0.333

### physics_005 (unknown)

- Query: In special relativity, what happens to time for a fast-moving clock?
- Expected: time dilation
- Generated: time dilation
- Metrics: f1=1.000, contains=1.0, faith=0.500

### physics_006 (unknown)

- Query: What geometric object is curved by mass-energy in general relativity?
- Expected: spacetime
- Generated: spacetime
- Metrics: f1=1.000, contains=1.0, faith=0.000

### physics_007 (unknown)

- Query: What boundary of a black hole marks no return for light?
- Expected: event horizon
- Generated: event horizon
- Metrics: f1=1.000, contains=1.0, faith=0.000

### physics_008 (unknown)

- Query: Which law gives divergence of electric field in terms of charge density?
- Expected: gauss law
- Generated: gauss law
- Metrics: f1=1.000, contains=1.0, faith=0.500

### physics_009 (unknown)

- Query: What is the force on charge q moving with velocity v in fields E and B called?
- Expected: lorentz force
- Generated: lorentz force
- Metrics: f1=1.000, contains=1.0, faith=0.000

### physics_010 (unknown)

- Query: State the trend of entropy in an isolated system according to the second law.
- Expected: entropy increases
- Generated: entropy increases
- Metrics: f1=1.000, contains=1.0, faith=0.000

### chemistry_001 (unknown)

- Query: How many electrons can occupy an orbital with quantum number n equals 3?
- Expected: eighteen electrons
- Generated: eighteen electrons
- Metrics: f1=1.000, contains=1.0, faith=0.000

### chemistry_002 (unknown)

- Query: What does the principal quantum number n determine in an atom?
- Expected: electron shell size and energy
- Generated: electron shell size and energy
- Metrics: f1=1.000, contains=1.0, faith=0.200

### chemistry_003 (unknown)

- Query: Which type of bond involves sharing of electron pairs between atoms?
- Expected: covalent bond
- Generated: covalent bond
- Metrics: f1=1.000, contains=1.0, faith=0.000

### chemistry_004 (unknown)

- Query: What is the geometry of a carbon atom with four single bonds?
- Expected: tetrahedral
- Generated: tetrahedral
- Metrics: f1=1.000, contains=1.0, faith=0.000

### chemistry_005 (unknown)

- Query: What substance is produced when an acid reacts with a base?
- Expected: salt and water
- Generated: salt and water
- Metrics: f1=1.000, contains=1.0, faith=0.667

### chemistry_006 (unknown)

- Query: In an exothermic reaction, what is released to the surroundings?
- Expected: heat energy
- Generated: heat energy
- Metrics: f1=1.000, contains=1.0, faith=0.500

### chemistry_007 (unknown)

- Query: What factor increases the rate of a chemical reaction by lowering activation energy?
- Expected: catalyst
- Generated: catalyst
- Metrics: f1=1.000, contains=1.0, faith=0.000

### chemistry_008 (unknown)

- Query: What is the pH of a neutral aqueous solution at standard conditions?
- Expected: seven
- Generated: seven
- Metrics: f1=1.000, contains=1.0, faith=1.000

### chemistry_009 (unknown)

- Query: In a reversible reaction at equilibrium, what is the relationship between forward and reverse rates?
- Expected: they are equal
- Generated: they are equal
- Metrics: f1=1.000, contains=1.0, faith=0.667

### chemistry_010 (unknown)

- Query: What is the term for loss of electrons in a redox reaction?
- Expected: oxidation
- Generated: oxidation
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_001 (unknown)

- Query: What is the basic unit of life?
- Expected: cell
- Generated: cell
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_002 (unknown)

- Query: Which organelle is responsible for producing energy in a cell?
- Expected: mitochondrion
- Generated: mitochondrion
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_003 (unknown)

- Query: What is the molecule that carries genetic information in most organisms?
- Expected: deoxyribonucleic acid
- Generated: deoxyribonucleic acid
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_004 (unknown)

- Query: How many DNA base pairs are in the human genome approximately?
- Expected: three billion base pairs
- Generated: three billion base pairs
- Metrics: f1=1.000, contains=1.0, faith=0.750

### biology_005 (unknown)

- Query: What is the primary mechanism of evolution proposed by Darwin?
- Expected: natural selection
- Generated: natural selection
- Metrics: f1=1.000, contains=1.0, faith=0.500

### biology_006 (unknown)

- Query: What term describes the role of an organism in its environment?
- Expected: ecological niche
- Generated: ecological niche
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_007 (unknown)

- Query: What do plants use to convert light energy into chemical energy?
- Expected: photosynthesis
- Generated: photosynthesis
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_008 (unknown)

- Query: What type of white blood cell directly attacks infected cells?
- Expected: cytotoxic t lymphocyte
- Generated: cytotoxic t lymphocyte
- Metrics: f1=1.000, contains=1.0, faith=0.333

### biology_009 (unknown)

- Query: What is the kingdom of organisms that includes humans?
- Expected: animalia
- Generated: animalia
- Metrics: f1=1.000, contains=1.0, faith=0.000

### biology_010 (unknown)

- Query: How many nucleotides are in a codon?
- Expected: three nucleotides
- Generated: three nucleotides
- Metrics: f1=1.000, contains=1.0, faith=0.500

### mathematics_001 (unknown)

- Query: What is the solution to the equation two x plus three equals seven?
- Expected: x equals two
- Generated: x equals two
- Metrics: f1=1.000, contains=1.0, faith=0.333

### mathematics_002 (unknown)

- Query: What is the standard form of a linear equation in two variables?
- Expected: a x plus b y equals c
- Generated: a x plus b y equals c
- Metrics: f1=1.000, contains=1.0, faith=0.143

### mathematics_003 (unknown)

- Query: What is the sum of angles in a triangle?
- Expected: one hundred eighty degrees
- Generated: one hundred eighty degrees
- Metrics: f1=1.000, contains=1.0, faith=0.250

### mathematics_004 (unknown)

- Query: What is the area formula for a circle?
- Expected: pi r squared
- Generated: pi r squared
- Metrics: f1=1.000, contains=1.0, faith=0.000

### mathematics_005 (unknown)

- Query: What is the derivative of x squared?
- Expected: two x
- Generated: two x
- Metrics: f1=1.000, contains=1.0, faith=0.500

### mathematics_006 (unknown)

- Query: What does the definite integral represent geometrically?
- Expected: area under the curve
- Generated: area under the curve
- Metrics: f1=1.000, contains=1.0, faith=0.250

### mathematics_007 (unknown)

- Query: What does the standard deviation measure?
- Expected: spread of data from the mean
- Generated: spread of data from the mean
- Metrics: f1=1.000, contains=1.0, faith=0.667

### mathematics_008 (unknown)

- Query: What is the probability of rolling a six on a fair die?
- Expected: one sixth
- Generated: one sixth
- Metrics: f1=1.000, contains=1.0, faith=0.000

### mathematics_009 (unknown)

- Query: What does matrix multiplication compute between two vectors?
- Expected: dot product
- Generated: dot product
- Metrics: f1=1.000, contains=1.0, faith=0.000

### mathematics_010 (unknown)

- Query: What is a prime number?
- Expected: number divisible only by one and itself
- Generated: number divisible only by one and itself
- Metrics: f1=1.000, contains=1.0, faith=0.714

### earth_science_001 (unknown)

- Query: What are the three main types of rocks?
- Expected: igneous sedimentary metamorphic
- Generated: igneous sedimentary metamorphic
- Metrics: f1=1.000, contains=1.0, faith=0.000

### earth_science_002 (unknown)

- Query: What scale measures the hardness of minerals?
- Expected: mohs hardness scale
- Generated: mohs hardness scale
- Metrics: f1=1.000, contains=1.0, faith=0.333

### earth_science_003 (unknown)

- Query: What is the most abundant mineral in Earth's crust?
- Expected: feldspar
- Generated: feldspar
- Metrics: f1=1.000, contains=1.0, faith=0.000

### earth_science_004 (unknown)

- Query: What weather system has the lowest atmospheric pressure?
- Expected: hurricane
- Generated: hurricane
- Metrics: f1=1.000, contains=1.0, faith=0.000

### earth_science_005 (unknown)

- Query: What causes ocean tides?
- Expected: gravitational force of the moon
- Generated: gravitational force of the moon
- Metrics: f1=1.000, contains=1.0, faith=0.400

### earth_science_006 (unknown)

- Query: What is the movement of tectonic plates called?
- Expected: plate tectonics
- Generated: plate tectonics
- Metrics: f1=1.000, contains=1.0, faith=0.500

### earth_science_007 (unknown)

- Query: What gas is primarily responsible for current global warming?
- Expected: carbon dioxide
- Generated: carbon dioxide
- Metrics: f1=1.000, contains=1.0, faith=0.000

### earth_science_008 (unknown)

- Query: What is the layer of Earth between the crust and outer core?
- Expected: mantle
- Generated: mantle
- Metrics: f1=1.000, contains=1.0, faith=0.000

### earth_science_009 (unknown)

- Query: What scale measures the energy released by earthquakes?
- Expected: richter scale
- Generated: richter scale
- Metrics: f1=1.000, contains=1.0, faith=0.500

### earth_science_010 (unknown)

- Query: What is the closest star to Earth besides the Sun?
- Expected: proxima centauri
- Generated: proxima centauri
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_001 (unknown)

- Query: What is the time complexity of binary search?
- Expected: logarithmic time
- Generated: logarithmic time
- Metrics: f1=1.000, contains=1.0, faith=0.500

### computer_science_002 (unknown)

- Query: What data structure uses LIFO principle?
- Expected: stack
- Generated: stack
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_003 (unknown)

- Query: What does SQL stand for?
- Expected: structured query language
- Generated: structured query language
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_004 (unknown)

- Query: What is a function in programming?
- Expected: reusable block of code
- Generated: reusable block of code
- Metrics: f1=1.000, contains=1.0, faith=0.250

### computer_science_005 (unknown)

- Query: What does TCP stand for?
- Expected: transmission control protocol
- Generated: transmission control protocol
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_006 (unknown)

- Query: What type of encryption uses two keys?
- Expected: asymmetric encryption
- Generated: asymmetric encryption
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_007 (unknown)

- Query: What is the Church-Turing thesis about?
- Expected: computability of algorithms
- Generated: computability of algorithms
- Metrics: f1=1.000, contains=1.0, faith=0.333

### computer_science_008 (unknown)

- Query: What is supervised learning?
- Expected: learning from labeled data
- Generated: learning from labeled data
- Metrics: f1=1.000, contains=1.0, faith=0.500

### computer_science_009 (unknown)

- Query: What does AI stand for?
- Expected: artificial intelligence
- Generated: artificial intelligence
- Metrics: f1=1.000, contains=1.0, faith=0.000

### computer_science_010 (unknown)

- Query: What is a deadlock in operating systems?
- Expected: circular wait for resources
- Generated: circular wait for resources
- Metrics: f1=1.000, contains=1.0, faith=0.250
