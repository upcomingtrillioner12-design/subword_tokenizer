# RAG Evaluation Report: PHYSICS Corpus

**Timestamp:** 2026-07-14 14:11:44
**Embedding Model:** all-mpnet-base-v2
**Test Cases:** 8

## Retriever Comparison

| Retriever | Precision@1 | Precision@5 | Recall@1 | Recall@5 | Hit@1 | MRR |
|-----------|-------------|-------------|----------|----------|-------|-----|
| bm25 | 0.625 | 0.150 | 0.625 | 0.750 | 0.625 | 0.681 |
| dense | 0.625 | 0.150 | 0.625 | 0.750 | 0.625 | 0.708 |
| hybrid_rrf | 0.625 | 0.175 | 0.625 | 0.875 | 0.625 | 0.713 |
| hybrid_weighted_03 | 0.625 | 0.175 | 0.625 | 0.875 | 0.625 | 0.713 |

## Per-Query Results

### bm25

**Query q001:** What causes black holes to evaporate?
- Retrieved: ['0', '4', '21']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q002:** What is quantum entanglement and its implications?
- Retrieved: ['14', '12', '1']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.333

**Query q003:** Explain the Higgs mechanism and electroweak symmetry breaking.
- Retrieved: ['2', '17', '0']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q004:** How does gravity work according to Einstein?
- Retrieved: ['3', '22', '10']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q005:** What is dark matter and how do we detect it?
- Retrieved: ['4', '11', '13']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q006:** Explain wave-particle duality in quantum mechanics.
- Retrieved: ['5', '6', '24']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q007:** What is the standard model of particle physics?
- Retrieved: ['8', '2', '5']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.000

**Query q008:** How does supersymmetry extend the standard model?
- Retrieved: ['8', '2', '23']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.111

### dense

**Query q001:** What causes black holes to evaporate?
- Retrieved: ['0', '21', '3']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q002:** What is quantum entanglement and its implications?
- Retrieved: ['14', '1', '20']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.500

**Query q003:** Explain the Higgs mechanism and electroweak symmetry breaking.
- Retrieved: ['2', '8', '23']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q004:** How does gravity work according to Einstein?
- Retrieved: ['3', '0', '10']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q005:** What is dark matter and how do we detect it?
- Retrieved: ['4', '23', '13']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q006:** Explain wave-particle duality in quantum mechanics.
- Retrieved: ['5', '20', '24']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q007:** What is the standard model of particle physics?
- Retrieved: ['8', '23', '2']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.000

**Query q008:** How does supersymmetry extend the standard model?
- Retrieved: ['2', '8', '10']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.167

### hybrid_rrf

**Query q001:** What causes black holes to evaporate?
- Retrieved: ['0', '21', '6']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q002:** What is quantum entanglement and its implications?
- Retrieved: ['14', '1', '20']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.500

**Query q003:** Explain the Higgs mechanism and electroweak symmetry breaking.
- Retrieved: ['2', '17', '24']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q004:** How does gravity work according to Einstein?
- Retrieved: ['3', '10', '0']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q005:** What is dark matter and how do we detect it?
- Retrieved: ['4', '13', '3']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q006:** Explain wave-particle duality in quantum mechanics.
- Retrieved: ['5', '24', '20']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q007:** What is the standard model of particle physics?
- Retrieved: ['8', '2', '5']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.000

**Query q008:** How does supersymmetry extend the standard model?
- Retrieved: ['8', '2', '23']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.200

### hybrid_weighted_03

**Query q001:** What causes black holes to evaporate?
- Retrieved: ['0', '21', '6']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q002:** What is quantum entanglement and its implications?
- Retrieved: ['14', '1', '20']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.500

**Query q003:** Explain the Higgs mechanism and electroweak symmetry breaking.
- Retrieved: ['2', '17', '24']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q004:** How does gravity work according to Einstein?
- Retrieved: ['3', '10', '0']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q005:** What is dark matter and how do we detect it?
- Retrieved: ['4', '13', '3']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q006:** Explain wave-particle duality in quantum mechanics.
- Retrieved: ['5', '24', '20']
- Precision@1: 1.000
- Recall@5: 1.000
- MRR: 1.000

**Query q007:** What is the standard model of particle physics?
- Retrieved: ['8', '2', '5']
- Precision@1: 0.000
- Recall@5: 0.000
- MRR: 0.000

**Query q008:** How does supersymmetry extend the standard model?
- Retrieved: ['8', '2', '23']
- Precision@1: 0.000
- Recall@5: 1.000
- MRR: 0.200

