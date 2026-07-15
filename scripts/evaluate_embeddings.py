#!/usr/bin/env python3
"""
Phase 4 Task 8: Embedding Model Comparison - Retrieval Quality Analysis
Direct similarity scoring approach for fast evaluation
"""

import json
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    from embedding_selector import EmbeddingSelector
except ImportError:
    print("❌ embedding_selector.py not found")
    sys.exit(1)


def load_corpus_chunks():
    """Load corpus chunks from metadata"""
    metadata_path = Path('data/retrieval/dense_general/dense_index_metadata.jsonl')
    chunks = []
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    if text:
                        chunks.append(text[:1000])
                except:
                    continue
    
    return chunks


def compute_retrieval_score(query, expected_answer, model, chunks, top_k=10):
    """Compute retrieval quality metrics"""
    try:
        query_emb = model.encode([query])[0]
        
        # Sample for efficiency
        if len(chunks) > 500:
            sampled_indices = np.random.choice(len(chunks), 500, replace=False)
            sampled_chunks = [chunks[i] for i in sampled_indices]
        else:
            sampled_chunks = chunks
        
        chunk_embs = model.encode(sampled_chunks, batch_size=32)
        
        # Similarities
        similarities = chunk_embs @ query_emb / (
            np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_sims = similarities[top_indices]
        
        # Check for answer keyword
        answer_keyword = expected_answer.split()[0] if expected_answer != 'CANNOT_BE_ANSWERED' else None
        answer_found = False
        best_rank = -1
        
        if answer_keyword:
            for rank, idx in enumerate(top_indices):
                if answer_keyword.lower() in sampled_chunks[idx].lower():
                    answer_found = True
                    best_rank = rank + 1
                    break
        
        return {
            'max_similarity': float(top_sims[0]),
            'avg_top5': float(np.mean(top_sims[:5])),
            'avg_top10': float(np.mean(top_sims)),
            'answer_found': answer_found,
            'answer_rank': best_rank
        }
    
    except Exception as e:
        return {
            'max_similarity': 0.0,
            'avg_top5': 0.0,
            'avg_top10': 0.0,
            'answer_found': False,
            'answer_rank': -1
        }


def main():
    print("=" * 75)
    print("Phase 4 Task 8: Embedding Model Comparison - Retrieval Quality")
    print("=" * 75)
    
    # Load data
    print("\n📚 Loading corpus chunks...")
    chunks = load_corpus_chunks()
    print(f"  ✓ Loaded {len(chunks)} chunks")
    
    print("\n📖 Loading 20-question subset...")
    subset_path = Path('data/phase4_task8_adversarial_subset_20qa.json')
    with open(subset_path) as f:
        subset_data = json.load(f)
    
    questions = subset_data['questions'][:20]
    print(f"  ✓ Loaded {len(questions)} questions")
    
    # Initialize model
    print("\n🔧 Loading embedding model...")
    print("  all-mpnet-base-v2...", end='', flush=True)
    model = EmbeddingSelector('all-mpnet-base-v2')
    print(" ✓")
    
    # Evaluate
    print(f"\n🔍 Evaluating on {len(questions)} questions...\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'all-mpnet-base-v2',
        'subset_size': len(questions),
        'questions': [],
        'metrics': {
            'max_sim': [],
            'top5_sim': [],
            'top10_sim': [],
            'answers_found': 0,
            'answerable_count': 0
        }
    }
    
    for i, q in enumerate(questions, 1):
        q_id = q.get('id', f'q_{i}')
        question = q.get('question', '')
        expected = q.get('expected_answer', '')
        q_type = q.get('type', 'unknown')
        is_answerable = expected != 'CANNOT_BE_ANSWERED'
        
        score = compute_retrieval_score(question, expected, model, chunks)
        
        results['questions'].append({
            'id': q_id,
            'type': q_type,
            'is_answerable': is_answerable,
            'scores': score
        })
        
        results['metrics']['max_sim'].append(score['max_similarity'])
        results['metrics']['top5_sim'].append(score['avg_top5'])
        results['metrics']['top10_sim'].append(score['avg_top10'])
        results['metrics']['answerable_count'] += is_answerable
        
        if score['answer_found']:
            results['metrics']['answers_found'] += 1
        
        if i % 5 == 0:
            print(f"  [{i}/{len(questions)}] max_sim={score['max_similarity']:.4f}, "
                  f"avg_top5={score['avg_top5']:.4f}")
    
    # Summary
    results['summary'] = {
        'avg_max_similarity': float(np.mean(results['metrics']['max_sim'])),
        'avg_top5_similarity': float(np.mean(results['metrics']['top5_sim'])),
        'avg_top10_similarity': float(np.mean(results['metrics']['top10_sim'])),
        'answers_found': results['metrics']['answers_found'],
        'answerable_questions': results['metrics']['answerable_count'],
        'answer_retrieval_rate': float(results['metrics']['answers_found'] / max(1, results['metrics']['answerable_count']))
    }
    
    # Save
    output_dir = Path('results/rag_generation_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'phase4_task8_embedding_eval_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"\nModel: all-mpnet-base-v2")
    print(f"Questions: {results['summary']['answerable_questions']} answerable / {len(questions)} total")
    print(f"\nRetrieval Quality (average cosine similarity):")
    print(f"  Max similarity (best chunk):  {results['summary']['avg_max_similarity']:.4f}")
    print(f"  Avg top-5 similarity:         {results['summary']['avg_top5_similarity']:.4f}")
    print(f"  Avg top-10 similarity:        {results['summary']['avg_top10_similarity']:.4f}")
    print(f"\nAnswer Retrieval:")
    print(f"  Answers found in top-10:      {results['summary']['answers_found']}/{results['summary']['answerable_questions']}")
    print(f"  Retrieval success rate:       {results['summary']['answer_retrieval_rate']:.1%}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
