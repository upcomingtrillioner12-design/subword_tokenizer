#!/usr/bin/env python3
"""
Task 10.2b: Collect STEM Preference Pairs for Cross-Encoder Fine-Tuning

Loads STEM benchmark questions, retrieves top-5 documents per question,
labels them as relevant (1.0) or irrelevant (0.0) based on whether they
contain the expected answer, and saves preference pairs for fine-tuning.

Output: 60 questions × 5 docs = 300 preference pairs (80/20 train/val split)
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_retrieval import HybridRetriever


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _weak_relevance_score(expected_answer: str, doc_text: str) -> float:
    """Heuristic relevance score in [0,1] using exact match + token overlap."""
    exp = expected_answer.lower().strip()
    doc = doc_text.lower()
    if not exp or not doc:
        return 0.0

    # Strong exact phrase match
    if exp in doc:
        return 1.0

    exp_tokens = [t for t in _tokenize(exp) if len(t) > 2]
    doc_tokens = set(_tokenize(doc))
    if not exp_tokens:
        return 0.0

    overlap = sum(1 for t in exp_tokens if t in doc_tokens)
    recall = overlap / max(len(exp_tokens), 1)
    return float(recall)


def collect_preference_pairs(
    questions_file: str,
    output_file: str,
    k_docs: int = 5,
    verbose: bool = True
) -> List[Dict]:
    """
    Collect preference pairs from STEM benchmark questions.
    
    Args:
        questions_file: Path to STEM questions JSON
        output_file: Path to save preference pairs
        k_docs: Number of documents to label per question
        verbose: Print progress
    
    Returns:
        List of preference pair dicts
    """
    # Load STEM questions
    questions_path = Path(questions_file)
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_file}")
        print("\nSearching for STEM benchmark files...")
        import glob
        stem_files = glob.glob("data/*stem*.json") + glob.glob("data/phase4*stem*.json")
        print(f"Found potential files: {stem_files}")
        return []
    
    with open(questions_path) as f:
        data = json.load(f)
    
    questions = data.get("questions", data)
    if verbose:
        print(f"\nLoading {len(questions)} STEM questions from {questions_file}")
    
    # Initialize retriever with existing retrieval indices
    root = Path(__file__).resolve().parents[1]
    retriever = HybridRetriever(
        bm25_index_path=str(root / "data/retrieval/bm25_index.json"),
        dense_index_path=str(root / "data/retrieval/dense_general/dense_index.faiss"),
        dense_metadata_path=str(root / "data/retrieval/dense_general/dense_index_metadata.jsonl"),
        embedding_model="allenai/scibert_scivocab_uncased",
        alpha=0.5,
        fusion_method="rrf",
    )
    
    pairs = []
    total_relevant = 0
    total_irrelevant = 0
    
    for i, question in enumerate(questions, start=1):
        query = question.get("question", "")
        expected_answer = question.get("expected_answer", "")
        question_id = question.get("id", f"stem_q{i}")
        
        if not query or not expected_answer:
            if verbose and i % 10 == 0:
                print(f"  [{i}/{len(questions)}] Skipping question with missing fields")
            continue
        
        # Retrieve top-k documents
        try:
            retrieved_docs = retriever.retrieve(query, k=k_docs)
        except Exception as e:
            print(f"  ERROR retrieving for {question_id}: {e}")
            continue
        
        # Weak-label with heuristic score, then ensure at least one positive per question
        scored_docs = []
        for j, doc in enumerate(retrieved_docs):
            doc_text = doc.get("text", "")
            doc_id = doc.get("doc_id", f"{question_id}_doc_{j}")
            score = _weak_relevance_score(expected_answer, doc_text)
            scored_docs.append((j, doc_id, doc_text, score))

        # Positive if strong match; otherwise force top retrieved doc as pseudo-positive
        positive_indices = {j for j, _, _, s in scored_docs if s >= 0.34}
        if not positive_indices and scored_docs:
            positive_indices.add(0)

        for j, doc_id, doc_text, score in scored_docs:
            is_relevant = j in positive_indices
            label = 1.0 if is_relevant else 0.0

            if label > 0.5:
                total_relevant += 1
            else:
                total_irrelevant += 1

            pair = {
                "query": query,
                "document": doc_text[:500],
                "label": label,
                "question_id": question_id,
                "doc_id": doc_id,
                "is_relevant": bool(is_relevant),
                "weak_score": float(score),
                "expected_answer": expected_answer,
            }
            pairs.append(pair)
        
        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(questions)}] Collected {len(pairs)} pairs so far...")
    
    if verbose:
        print(f"\n✓ Collected {len(pairs)} preference pairs")
        print(f"  Relevant: {total_relevant} ({total_relevant/len(pairs)*100:.1f}%)")
        print(f"  Irrelevant: {total_irrelevant} ({total_irrelevant/len(pairs)*100:.1f}%)")
    
    # Save pairs
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    if verbose:
        print(f"✓ Saved pairs to {output_file}")
    
    return pairs


def print_pair_statistics(pairs: List[Dict]):
    """Print statistics about the collected preference pairs."""
    if not pairs:
        print("No pairs to analyze")
        return
    
    print("\n" + "="*70)
    print("PREFERENCE PAIR STATISTICS")
    print("="*70)
    
    total = len(pairs)
    relevant = sum(1 for p in pairs if p['label'] > 0.5)
    irrelevant = total - relevant
    
    unique_questions = len(set(p['question_id'] for p in pairs))
    
    print(f"\nTotal pairs: {total}")
    print(f"  Relevant (label=1.0): {relevant} ({relevant/total*100:.1f}%)")
    print(f"  Irrelevant (label=0.0): {irrelevant} ({irrelevant/total*100:.1f}%)")
    print(f"\nUnique questions: {unique_questions}")
    print(f"Average docs per question: {total/unique_questions:.1f}")
    
    # Train/val split
    train_size = int(0.8 * total)
    val_size = total - train_size
    print(f"\nTrain/val split (80/20):")
    print(f"  Training pairs: {train_size}")
    print(f"  Validation pairs: {val_size}")
    print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect STEM preference pairs for cross-encoder fine-tuning"
    )
    parser.add_argument(
        "--questions",
        default="data/phase4_stem_60qa.json",
        help="Path to STEM questions file"
    )
    parser.add_argument(
        "--output",
        default="data/cross_encoder_stem_preference_pairs.json",
        help="Path to save preference pairs"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per question"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PHASE 4 TASK 10.2b: COLLECT STEM PREFERENCE PAIRS")
    print("="*70)
    
    pairs = collect_preference_pairs(
        questions_file=args.questions,
        output_file=args.output,
        k_docs=args.k,
        verbose=args.verbose
    )
    
    if pairs:
        print_pair_statistics(pairs)
    else:
        print("\nERROR: No pairs collected. Check question file path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
