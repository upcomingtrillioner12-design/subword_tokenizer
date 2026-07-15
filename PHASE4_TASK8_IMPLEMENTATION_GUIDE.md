# Phase 4 Task 8: Domain-Specific Embeddings Implementation Guide

**Objective:** Improve adversarial QA performance (35% → 40-45%) using specialized embeddings  
**Estimated Duration:** 2-3 hours  
**Success Criteria:** Embedding comparison report showing embedding model rankings on adversarial subset

---

## Overview

Task 8 will evaluate 3 embedding models on 20-question adversarial subset to identify best model for semantic discrimination on near-miss distractors and misleading contexts.

### Models to Evaluate

| Model | Size | Strengths | Best For | Status |
|-------|------|-----------|----------|--------|
| all-mpnet-base-v2 | 110M params | General-purpose, fast, proven | Baseline | ✅ Current |
| instructor-embedding | 330M params | Domain instructions, flexible | Semantic understanding | 🟡 Ready |
| SciBERT | 110M params | Scientific domain-tuned | Physics/chemistry/biology | 🟡 Ready |

### Expected Performance Impact

```
Baseline (all-mpnet):     35% exact match
With SciBERT:            +5-8% → 40-43% exact match
With Instructor:         +3-5% → 38-40% exact match
Combined best strategy:   40-45% exact match
```

---

## Implementation Steps

### Step 1: Prepare Adversarial Subset (10 min)

Select balanced 20-question subset:
```python
# From phase4_task7_adversarial_qa_dataset.json
# Sample: 7-8 misleading_context, 6-7 near_miss_distractor, 5-6 unanswerable
# Ensures representation of all challenge types
```

**File to create:**
- `data/phase4_task8_adversarial_subset_20qa.json`

**Script to create subset:**
```python
import json
from pathlib import Path

dataset_path = Path('data/phase4_task7_adversarial_qa_dataset.json')
with open(dataset_path) as f:
    data = json.load(f)

# Group by type
by_type = {}
for q in data['questions']:
    q_type = q.get('type', 'unknown')
    if q_type not in by_type:
        by_type[q_type] = []
    by_type[q_type].append(q)

# Sample balanced (20 total = 7-8 per type)
subset = []
for q_type, questions in by_type.items():
    per_type = 20 // len(by_type)
    subset.extend(questions[:per_type])

subset = subset[:20]

# Save
subset_data = data.copy()
subset_data['questions'] = subset
with open('data/phase4_task8_adversarial_subset_20qa.json', 'w') as f:
    json.dump(subset_data, f, indent=2)

print(f"Created subset with {len(subset)} questions")
```

### Step 2: Create Embedding Comparison Script (20 min)

**File:** `scripts/evaluate_embeddings.py`

```python
#!/usr/bin/env python3
"""
Phase 4 Task 8: Compare embedding models on adversarial QA
Measures: retrieval precision@K, ranking quality, answer discrimination
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

# Use embedding_selector.py from Task 7
sys.path.insert(0, str(Path(__file__).parent))
from embedding_selector import EmbeddingComparator
from enhanced_metrics import MetricsAggregator

def load_corpus_and_embeddings(models_list: List[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load corpus and encode with all models"""
    
    # Load corpus
    corpus_path = Path('data/retrieval/bm25_corpus.json')  # Adjust path as needed
    with open(corpus_path) as f:
        corpus_data = json.load(f)
    
    corpus = [doc.get('text', '') for doc in corpus_data.get('documents', [])]
    print(f"✓ Loaded corpus: {len(corpus)} documents")
    
    # Initialize comparator
    comparator = EmbeddingComparator(models_list)
    
    # Encode corpus with all models
    print("🔄 Encoding corpus with all models...")
    corpus_embeddings = {}
    for model_name in models_list:
        print(f"  {model_name}...", end='', flush=True)
        embeddings = comparator.models[model_name].encode(corpus)
        corpus_embeddings[model_name] = embeddings
        print(f" ✓ ({embeddings.shape})")
    
    return corpus, corpus_embeddings, comparator

def evaluate_embedding_model(query: str, expected_answer: str, corpus: List[str],
                             corpus_embeddings: Dict[str, np.ndarray],
                             model_name: str, comparator) -> Dict[str, float]:
    """
    Evaluate single embedding model on query
    Returns: precision@5, precision@10, answer ranking position
    """
    
    # Encode query
    query_embedding = comparator.models[model_name].encode([query])[0]
    corpus_embs = corpus_embeddings[model_name]
    
    # Compute similarities
    similarities = corpus_embs @ query_embedding / (
        np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    
    # Get rankings
    top_indices = np.argsort(similarities)[::-1]
    top_docs = [corpus[i] for i in top_indices[:10]]
    
    # Check if expected answer in top-K
    answer_in_top5 = any(expected_answer.lower() in doc.lower() for doc in top_docs[:5])
    answer_in_top10 = any(expected_answer.lower() in doc.lower() for doc in top_docs)
    
    # Find first rank of expected answer
    answer_rank = -1
    for i, doc in enumerate(top_docs):
        if expected_answer.lower() in doc.lower():
            answer_rank = i + 1
            break
    
    return {
        'precision@5': float(answer_in_top5),
        'precision@10': float(answer_in_top10),
        'answer_rank': answer_rank,
        'answer_similarity': float(similarities[top_indices[0]]) if answer_rank > 0 else 0.0
    }

def run_evaluation(models_list: List[str], subset_size: int = 20) -> Dict:
    """
    Run full evaluation comparing embedding models
    """
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': models_list,
        'subset_size': subset_size,
        'model_results': {model: {} for model in models_list},
        'summary': {}
    }
    
    # Load data
    print("📊 Loading corpus and embeddings...")
    corpus, corpus_embeddings, comparator = load_corpus_and_embeddings(models_list)
    
    # Load questions
    subset_path = Path('data/phase4_task8_adversarial_subset_20qa.json')
    with open(subset_path) as f:
        subset_data = json.load(f)
    
    questions = subset_data.get('questions', [])[:subset_size]
    print(f"✓ Loaded {len(questions)} adversarial questions")
    
    # Evaluate each question with each model
    print(f"\n🔍 Evaluating {len(models_list)} models on {len(questions)} questions...")
    
    for model_name in models_list:
        results['model_results'][model_name] = {
            'questions': [],
            'metrics': {
                'avg_precision@5': 0.0,
                'avg_precision@10': 0.0,
                'avg_rank': 0.0,
                'found_answers': 0
            }
        }
        
        for q in questions:
            q_id = q.get('id', 'unknown')
            question = q.get('question', '')
            expected = q.get('expected_answer', '')
            
            eval_result = evaluate_embedding_model(
                question, expected, corpus, corpus_embeddings, model_name, comparator
            )
            
            results['model_results'][model_name]['questions'].append({
                'id': q_id,
                'results': eval_result
            })
        
        # Compute model-level metrics
        q_results = results['model_results'][model_name]['questions']
        precisions_5 = [q['results']['precision@5'] for q in q_results]
        precisions_10 = [q['results']['precision@10'] for q in q_results]
        ranks = [q['results']['answer_rank'] for q in q_results if q['results']['answer_rank'] > 0]
        
        results['model_results'][model_name]['metrics'] = {
            'avg_precision@5': float(np.mean(precisions_5)),
            'avg_precision@10': float(np.mean(precisions_10)),
            'avg_rank': float(np.mean(ranks)) if ranks else 0.0,
            'found_answers': len(ranks)
        }
        
        print(f"  ✓ {model_name}: p@5={np.mean(precisions_5):.2%}, p@10={np.mean(precisions_10):.2%}")
    
    # Compute overall summary
    model_p5_scores = {}
    for model in models_list:
        model_p5_scores[model] = results['model_results'][model]['metrics']['avg_precision@5']
    
    best_model = max(model_p5_scores, key=model_p5_scores.get)
    
    results['summary'] = {
        'best_model': best_model,
        'model_rankings': sorted(model_p5_scores.items(), key=lambda x: x[1], reverse=True),
        'improvement_over_baseline': {
            model: (score - model_p5_scores['all-mpnet-base-v2']) * 100
            for model, score in model_p5_scores.items()
        }
    }
    
    return results

if __name__ == '__main__':
    models = ['all-mpnet-base-v2', 'instructor-embedding', 'SciBERT']
    
    print("=" * 60)
    print("Phase 4 Task 8: Embedding Model Comparison")
    print("=" * 60)
    
    results = run_evaluation(models, subset_size=20)
    
    # Save results
    output_dir = Path('results/rag_generation_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'phase4_task8_embedding_comparison_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nBest Model: {results['summary']['best_model']}")
    print("\nModel Rankings (by Precision@5):")
    for rank, (model, score) in enumerate(results['summary']['model_rankings'], 1):
        baseline_delta = results['summary']['improvement_over_baseline'][model]
        delta_str = f"+{baseline_delta:.1f}pp" if baseline_delta > 0 else f"{baseline_delta:.1f}pp"
        print(f"  {rank}. {model}: {score:.2%} {delta_str}")
```

### Step 3: Create Instruction Templates (15 min)

**File:** `config/embedding_instructions.yaml`

```yaml
# Domain-specific instructions for instructor-embedding

instructions:
  general: "Represent this text for retrieval"
  
  physics_query: "Represent this physics question for semantic search and retrieval"
  physics_passage: "Represent this physics educational content for retrieval"
  
  chemistry_query: "Represent this chemistry question for molecular and chemical retrieval"
  chemistry_passage: "Represent this chemistry educational content for retrieval"
  
  biology_query: "Represent this biology/life science question for semantic search"
  biology_passage: "Represent this biology educational content for retrieval"
  
  mathematics_query: "Represent this mathematics problem for solution retrieval"
  mathematics_passage: "Represent this mathematics solution for retrieval"
  
  earth_science_query: "Represent this earth science question for information retrieval"
  earth_science_passage: "Represent this earth science content for retrieval"
  
  computer_science_query: "Represent this computer science problem for technical retrieval"
  computer_science_passage: "Represent this computer science content for retrieval"

# Usage: 
#   For query: instruction = instructions[domain + "_query"]
#   For passage: instruction = instructions[domain + "_passage"]
#   Encoder: model.encode([[instruction, text]])
```

### Step 4: Run Embedding Evaluation (30 min)

```bash
cd /Users/jdsingh/slm_v0/subword_tokenizer

# Run comparison
python3.12 scripts/evaluate_embeddings.py

# Expected output:
# ✓ Loaded corpus: 5305 documents
# 🔄 Encoding corpus with all models...
#   all-mpnet-base-v2... ✓ (5305, 768)
#   instructor-embedding... ✓ (5305, 768)
#   SciBERT... ✓ (5305, 768)
# 🔍 Evaluating 3 models on 20 questions...
#   ✓ all-mpnet-base-v2: p@5=XX.XX%, p@10=XX.XX%
#   ✓ instructor-embedding: p@5=XX.XX%, p@10=XX.XX%
#   ✓ SciBERT: p@5=XX.XX%, p@10=XX.XX%
# ✅ Results saved to results/rag_generation_eval/phase4_task8_embedding_comparison_TIMESTAMP.json
```

### Step 5: Generate Comparison Report (30 min)

**File:** `results/rag_generation_eval/phase4_task8_embedding_comparison.md`

Template structure:
```markdown
# Phase 4 Task 8: Embedding Model Comparison Report

## Executive Summary
- Best model: [winner]
- Improvement over baseline: [%]
- Recommendation: [model + expected full-dataset improvement]

## Detailed Results

### Model 1: all-mpnet-base-v2 (Baseline)
- Precision@5: X.XX%
- Precision@10: X.XX%
- Average answer rank: X
- Strengths: General-purpose, fast
- Limitations: Non-specialized for science

### Model 2: instructor-embedding
- Precision@5: X.XX% (+Y.YYpp vs baseline)
- Precision@10: X.XX%
- Average answer rank: X
- Strengths: Domain instruction capable
- Limitations: Requires instructions

### Model 3: SciBERT
- Precision@5: X.XX% (+Y.YYpp vs baseline)
- Precision@10: X.XX%
- Average answer rank: X
- Strengths: Scientific domain tuned
- Limitations: Limited to science domains

## Analysis

### Why [Winner] Performs Best
[Analysis of strong/weak question types]

### Predicted Full-Dataset Performance
- 20-Q subset accuracy: X%
- Expected 40-Q adversarial accuracy: X-Y% (interpolated)
- STEM dataset impact: [+/-] Z% (if evaluated)

## Recommendation

**Proceed with [Model] for Task 9**

Reasoning:
1. Best precision metrics on adversarial subset
2. Consistent performance across question types
3. [X]% improvement over current baseline
4. Fits within deployment budget constraints

## Next Steps
- Integrate [winner] into run_rag_generation_evaluation.py
- Run full 40-Q adversarial evaluation
- Measure exact match improvement
- Prepare for Task 9 (semantic metrics)

---
Report Generated: [timestamp]
Evaluated: 3 models × 20 questions = 60 comparisons
```

---

## Success Criteria

**Minimum (Task 8 Success):**
- ✅ Embedding comparison report generated
- ✅ Best model identified with confidence
- ✅ Predicted improvement quantified
- ✅ Clear recommendation for Task 9

**Stretch (Task 8+):**
- ✅ Run full 40-Q adversarial evaluation with winner
- ✅ Measure exact match improvement on full dataset
- ✅ Generate side-by-side before/after comparison

---

## Integration Checklist

- [ ] Create `data/phase4_task8_adversarial_subset_20qa.json`
- [ ] Create `scripts/evaluate_embeddings.py`
- [ ] Create `config/embedding_instructions.yaml`
- [ ] Run embedding comparison (gather results)
- [ ] Generate comparison report
- [ ] Update `run_rag_generation_evaluation.py` with model selection flag
- [ ] (Optional) Run full 40-Q evaluation with winning model
- [ ] Commit Task 8 work to main
- [ ] Update roadmap with results

---

## Fallback Plan

If model evaluation takes longer than expected:
1. **Reduce subset to 10 questions** (faster evaluation)
2. **Use cached corpus embeddings** (if available from previous runs)
3. **Skip instruction tuning** (use default encoder only)
4. **Generate qualitative report** (sample output instead of full)

Expected time savings: 15-20 minutes

---

## Resources

**Embedding Selector Code:**
- Location: `scripts/embedding_selector.py` ✅ (created Task 7)
- Usage: `EmbeddingComparator(['all-mpnet-base-v2', 'instructor-embedding', 'SciBERT'])`

**Adversarial Dataset:**
- Location: `data/phase4_task7_adversarial_qa_dataset.json` ✅ (created Task 7)
- Size: 40 questions (will subset to 20)

**Enhanced Metrics:**
- Location: `scripts/enhanced_metrics.py` ✅ (available)
- Available: precision@K, recall@K, F1, etc.

**Corpus & Indices:**
- BM25: `/data/retrieval/bm25_index.json`
- Dense: `/data/retrieval/dense_general/dense_index.faiss`

---

## Estimated Timeline

- [ ] 0-10 min: Setup (subset creation, script organization)
- [ ] 10-40 min: Evaluation (model encoding + question evaluation)
- [ ] 40-60 min: Analysis (report generation, visualization)
- [ ] 60-75 min: Refinement (optional full-dataset run, documentation)

**Total: 75 minutes** (~1.25 hours)

---

**Author:** AI Copilot  
**Date:** July 15, 2026  
**Status:** Ready for Implementation  
**Dependency:** Phase 4 Task 7 Complete ✅
