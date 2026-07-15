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


def compute_retrieval_score(query, expected_answer, model, sampled_chunks, sampled_embs, top_k=10):
    """Compute retrieval quality metrics"""
    try:
        query_emb = model.encode([query])[0]

        # Similarities
        similarities = sampled_embs @ query_emb / (
            np.linalg.norm(sampled_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
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
    
    # Fixed sample for fair comparison
    sample_size = min(800, len(chunks))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(chunks), size=sample_size, replace=False)
    sampled_chunks = [chunks[i] for i in sample_idx]

    models_to_test = [
        'all-mpnet-base-v2',
        'hkunlp/instructor-base',
        'allenai/scibert_scivocab_uncased',
    ]

    # Evaluate
    print(f"\n🔍 Evaluating {len(models_to_test)} models on {len(questions)} questions...\n")

    results = {
        'timestamp': datetime.now().isoformat(),
        'subset_size': len(questions),
        'corpus_sample_size': sample_size,
        'models': {}
    }

    for model_name in models_to_test:
        print(f"🔧 Loading {model_name}...", end='', flush=True)
        model = EmbeddingSelector(model_name)
        sampled_embs = model.encode(sampled_chunks, batch_size=32)
        print(" ✓")

        model_rows = []
        max_sim = []
        top5_sim = []
        top10_sim = []
        found = 0
        answerable = 0
        ranks = []

        for i, q in enumerate(questions, 1):
            q_id = q.get('id', f'q_{i}')
            question = q.get('question', '')
            expected = q.get('expected_answer', '')
            q_type = q.get('type', 'unknown')
            is_answerable = expected != 'CANNOT_BE_ANSWERED'

            score = compute_retrieval_score(question, expected, model, sampled_chunks, sampled_embs)

            model_rows.append({
                'id': q_id,
                'type': q_type,
                'is_answerable': is_answerable,
                'scores': score
            })

            max_sim.append(score['max_similarity'])
            top5_sim.append(score['avg_top5'])
            top10_sim.append(score['avg_top10'])
            answerable += int(is_answerable)

            if score['answer_found']:
                found += 1
                if score['answer_rank'] > 0:
                    ranks.append(score['answer_rank'])

            if i % 5 == 0:
                print(f"  [{i}/{len(questions)}] {model_name}")

        metrics = {
            'precision_at_5': float(np.mean([1.0 if (r['scores']['answer_rank'] > 0 and r['scores']['answer_rank'] <= 5) else 0.0 for r in model_rows])),
            'precision_at_10': float(np.mean([1.0 if r['scores']['answer_rank'] > 0 else 0.0 for r in model_rows])),
            'avg_rank_when_found': float(np.mean(ranks)) if ranks else -1.0,
            'avg_max_similarity': float(np.mean(max_sim)),
            'avg_top5_similarity': float(np.mean(top5_sim)),
            'avg_top10_similarity': float(np.mean(top10_sim)),
            'answers_found': found,
            'answerable_questions': answerable,
            'answer_retrieval_rate': float(found / max(1, answerable))
        }

        results['models'][model_name] = {
            'model_info': model.get_model_info(),
            'metrics': metrics,
            'questions': model_rows
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
    ranking = sorted(
        [(m, d['metrics']['precision_at_10']) for m, d in results['models'].items()],
        key=lambda x: x[1],
        reverse=True
    )
    for i, (model_name, p10) in enumerate(ranking, start=1):
        p5 = results['models'][model_name]['metrics']['precision_at_5']
        print(f"  {i}. {model_name}: p@5={p5:.3f}, p@10={p10:.3f}")

    results['summary'] = {
        'ranking_by_precision_at_10': ranking,
        'best_model': ranking[0][0] if ranking else None
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model: {results['summary']['best_model']}")
    print(f"Saved: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
