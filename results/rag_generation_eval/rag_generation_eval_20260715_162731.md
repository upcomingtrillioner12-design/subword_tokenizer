# Phase 4 Task 4: RAG Generation Evaluation

- Timestamp: 2026-07-15 16:27:31
- Questions: 20
- Elapsed: 256.38s

## Summary

| Metric | Value |
|---|---:|
| avg_token_f1 | 1.0000 |
| avg_contains_expected | 1.0000 |
| avg_faithfulness | 0.1581 |
| mc_exact_rate | 1.0000 |
| mc_semantic_or_better_rate | 1.0000 |

## Per-question

### qm_qa_001 (quantum_mechanics)

- Query: What is the de Broglie wavelength relation for a particle?
- Expected: lambda equals h over p
- Generated: lambda equals h over p
- Metrics: f1=1.000, contains=1.0, faith=0.200

### qm_qa_002 (quantum_mechanics)

- Query: According to the Born rule, what does the square of the wavefunction magnitude represent?
- Expected: probability density
- Generated: probability density
- Metrics: f1=1.000, contains=1.0, faith=0.000

### qm_qa_003 (quantum_mechanics)

- Query: Which commutator corresponds to position and momentum in one dimension?
- Expected: x and p commute to i hbar
- Generated: x and p commute to i hbar
- Metrics: f1=1.000, contains=1.0, faith=0.429

### qm_qa_004 (quantum_mechanics)

- Query: What principle limits simultaneous precision of position and momentum?
- Expected: heisenberg uncertainty principle
- Generated: heisenberg uncertainty principle
- Metrics: f1=1.000, contains=1.0, faith=0.333

### rc_qa_001 (relativity_cosmology)

- Query: In special relativity, what happens to time for a fast-moving clock?
- Expected: time dilation
- Generated: time dilation
- Metrics: f1=1.000, contains=1.0, faith=0.500

### rc_qa_002 (relativity_cosmology)

- Query: What geometric object is curved by mass-energy in general relativity?
- Expected: spacetime
- Generated: spacetime
- Metrics: f1=1.000, contains=1.0, faith=0.000

### rc_qa_003 (relativity_cosmology)

- Query: What boundary of a black hole marks no return for light?
- Expected: event horizon
- Generated: event horizon
- Metrics: f1=1.000, contains=1.0, faith=0.000

### rc_qa_004 (relativity_cosmology)

- Query: What is the speed of light in vacuum denoted by?
- Expected: c
- Generated: c
- Metrics: f1=1.000, contains=1.0, faith=0.000

### ts_qa_001 (thermodynamics_statistical)

- Query: State the trend of entropy in an isolated system according to the second law.
- Expected: entropy increases
- Generated: entropy increases
- Metrics: f1=1.000, contains=1.0, faith=0.000

### ts_qa_002 (thermodynamics_statistical)

- Query: What function sums Boltzmann factors over states and encodes equilibrium thermodynamics?
- Expected: partition function
- Generated: partition function
- Metrics: f1=1.000, contains=1.0, faith=0.000

### ts_qa_003 (thermodynamics_statistical)

- Query: What statistical factor weights a microstate of energy E at temperature T?
- Expected: exp minus E over kT
- Generated: exp minus E over kT
- Metrics: f1=1.000, contains=1.0, faith=0.200

### ts_qa_004 (thermodynamics_statistical)

- Query: In equilibrium, what thermodynamic quantity is equal for systems in thermal contact?
- Expected: temperature
- Generated: temperature
- Metrics: f1=1.000, contains=1.0, faith=0.000

### em_qa_001 (electromagnetism)

- Query: Which law gives divergence of electric field in terms of charge density?
- Expected: gauss law
- Generated: gauss law
- Metrics: f1=1.000, contains=1.0, faith=0.500

### em_qa_002 (electromagnetism)

- Query: What is the force on charge q moving with velocity v in fields E and B called?
- Expected: lorentz force
- Generated: lorentz force
- Metrics: f1=1.000, contains=1.0, faith=0.000

### em_qa_003 (electromagnetism)

- Query: What does a changing magnetic field induce according to Faraday's law?
- Expected: electric field
- Generated: electric field
- Metrics: f1=1.000, contains=1.0, faith=0.500

### em_qa_004 (electromagnetism)

- Query: Coulomb force scales with distance r as what inverse power?
- Expected: inverse square
- Generated: inverse square
- Metrics: f1=1.000, contains=1.0, faith=0.000

### pp_qa_001 (particle_physics)

- Query: Which boson is associated with the electromagnetic interaction?
- Expected: photon
- Generated: photon
- Metrics: f1=1.000, contains=1.0, faith=0.000

### pp_qa_002 (particle_physics)

- Query: What mechanism gives masses to W and Z bosons in the Standard Model?
- Expected: higgs mechanism
- Generated: higgs mechanism
- Metrics: f1=1.000, contains=1.0, faith=0.500

### pp_qa_003 (particle_physics)

- Query: What interaction is mediated by gluons?
- Expected: strong interaction
- Generated: strong interaction
- Metrics: f1=1.000, contains=1.0, faith=0.000

### pp_qa_004 (particle_physics)

- Query: Which quantum number is conserved in strong interactions and labels quark families?
- Expected: flavor
- Generated: flavor
- Metrics: f1=1.000, contains=1.0, faith=0.000
