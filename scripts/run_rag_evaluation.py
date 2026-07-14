#!/usr/bin/env python3
"""
RAG Evaluation Test Suite

Executes comprehensive RAG evaluation on physics and general corpora.
Generates sample sets and evaluates retrieval quality using BM25, Dense, and Hybrid retrievers.

Usage:
    python run_rag_evaluation.py --corpus physics --output results/rag_evaluation/
    python run_rag_evaluation.py --corpus general --output results/rag_evaluation/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
import time
import sys
import math
from collections import Counter, defaultdict
import re

# Import custom modules
from dense_retrieval import DenseRetriever
from hybrid_retrieval import HybridRetriever
from rag_evaluator import RAGEvaluator


# BM25 Retriever (inline from retrieval_baseline.py)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def bm25_score(
    query_terms: List[str],
    index: Dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    N = index["meta"]["num_docs"]
    avgdl = index["meta"]["avgdl"]
    doc_lens = index["doc_lens"]
    doc_freq = index["doc_freq"]
    postings = index["postings"]

    qtf = Counter(query_terms)
    for term, q_count in qtf.items():
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings.get(term, []):
            dl = doc_lens[doc_id]
            denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            term_score = idf * (tf * (k1 + 1)) / max(denom, 1e-9)
            scores[doc_id] += q_count * term_score
    return scores


class SimpleBM25Retriever:
    """Wrapper around BM25 index for retrieval."""
    
    def __init__(self, index: Dict):
        self.index = index
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k results for query."""
        q_terms = tokenize(query)
        scores = bm25_score(q_terms, self.index)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        docs = self.index["docs"]
        for rank, (doc_id, score) in enumerate(ranked, 1):
            doc = docs[doc_id]
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "text": doc.get("text", ""),
                "score": score,
                "source": doc.get("title", "unknown")
            })
        
        return results



class PhysicsTestSet:
    """
    Physics domain test set with 8 reference queries and expected documents.
    """
    
    @staticmethod
    def get_test_cases() -> List[Dict]:
        """Return physics test cases with queries and expected doc IDs."""
        return [
            {
                "query_id": "q001",
                "query": "What causes black holes to evaporate?",
                "expected_answer": "Hawking radiation",
                "expected_doc_ids": [0],  # Hawking Radiation & Black Hole Thermodynamics
                "domain": "General Relativity",
                "difficulty": "medium"
            },
            {
                "query_id": "q002",
                "query": "What is quantum entanglement and its implications?",
                "expected_answer": "non-local correlations",
                "expected_doc_ids": [1],  # Quantum Entanglement & Bell Nonlocality
                "domain": "Quantum Mechanics",
                "difficulty": "high"
            },
            {
                "query_id": "q003",
                "query": "Explain the Higgs mechanism and electroweak symmetry breaking.",
                "expected_answer": "mass generation mechanism",
                "expected_doc_ids": [2],  # Higgs Mechanism & Electroweak Symmetry
                "domain": "Particle Physics",
                "difficulty": "high"
            },
            {
                "query_id": "q004",
                "query": "How does gravity work according to Einstein?",
                "expected_answer": "spacetime curvature",
                "expected_doc_ids": [3],  # General Relativity: Spacetime Curvature
                "domain": "General Relativity",
                "difficulty": "medium"
            },
            {
                "query_id": "q005",
                "query": "What is dark matter and how do we detect it?",
                "expected_answer": "gravitational lensing, galaxy rotation",
                "expected_doc_ids": [4],  # Dark Matter Detection & Galactic Rotation
                "domain": "Cosmology",
                "difficulty": "high"
            },
            {
                "query_id": "q006",
                "query": "Explain wave-particle duality in quantum mechanics.",
                "expected_answer": "photons and electrons have both properties",
                "expected_doc_ids": [5],  # Wave-Particle Duality in QM
                "domain": "Quantum Mechanics",
                "difficulty": "medium"
            },
            {
                "query_id": "q007",
                "query": "What is the standard model of particle physics?",
                "expected_answer": "describes fundamental particles and forces",
                "expected_doc_ids": [6],  # Standard Model & Fundamental Particles
                "domain": "Particle Physics",
                "difficulty": "high"
            },
            {
                "query_id": "q008",
                "query": "How does supersymmetry extend the standard model?",
                "expected_answer": "predicts superpartners for each particle",
                "expected_doc_ids": [7],  # Supersymmetry Beyond Standard Model
                "domain": "Theoretical Physics",
                "difficulty": "high"
            },
        ]


def evaluate_retriever_on_corpus(
    retriever_name: str,
    retriever,
    test_cases: List[Dict],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Evaluate a single retriever on test cases.
    
    Args:
        retriever_name: Name of retriever (for logging)
        retriever: Retriever instance (with retrieve() or query() method)
        test_cases: List of test cases with query and expected_doc_ids
        k_values: K values for @k metrics
    
    Returns:
        Dict with per-query and aggregated results
    """
    print(f"\n  Evaluating {retriever_name}...")
    
    results = {
        "retriever": retriever_name,
        "num_queries": len(test_cases),
        "k_values": k_values,
        "per_query_results": [],
        "aggregated_metrics": {}
    }
    
    # Evaluate each query
    for tc in test_cases:
        query = tc["query"]
        expected_doc_ids = set(tc.get("expected_doc_ids", []))
        
        # Retrieve (try both retrieve and query methods)
        if hasattr(retriever, 'retrieve'):
            retrieved_results = retriever.retrieve(query, k=max(k_values))
        elif hasattr(retriever, 'query'):
            retrieved_results = retriever.query(query, k=max(k_values))
        else:
            raise AttributeError(f"Retriever has no retrieve() or query() method")
        
        retrieved_ids = [str(r["doc_id"]) for r in retrieved_results]  # Convert to string
        
        # Convert expected IDs to strings for comparison
        expected_doc_ids_str = set(str(d) for d in expected_doc_ids)
        
        # Compute metrics manually (simple implementation)
        query_metrics = {}
        for k in k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            relevant_found = retrieved_at_k & expected_doc_ids_str
            
            query_metrics[f"precision@{k}"] = len(relevant_found) / k if k > 0 else 0.0
            query_metrics[f"recall@{k}"] = len(relevant_found) / len(expected_doc_ids_str) if expected_doc_ids_str else 0.0
            query_metrics[f"hit@{k}"] = 1.0 if relevant_found else 0.0
            
            if relevant_found:
                first_relevant_idx = min([i for i, doc_id in enumerate(retrieved_ids) if doc_id in expected_doc_ids_str]) + 1
                query_metrics["mrr"] = 1.0 / first_relevant_idx
            else:
                query_metrics["mrr"] = 0.0
        
        results["per_query_results"].append({
            "query_id": tc.get("query_id", "unknown"),
            "query": query,
            "expected_doc_ids": list(expected_doc_ids_str),
            "retrieved_ids": retrieved_ids[:5],  # Top 5
            "metrics": query_metrics
        })
    
    # Aggregate metrics
    from collections import defaultdict
    aggregated = defaultdict(list)
    
    for qr in results["per_query_results"]:
        for metric_name, score in qr["metrics"].items():
            aggregated[metric_name].append(score)
    
    for metric_name, scores in aggregated.items():
        results["aggregated_metrics"][metric_name] = sum(scores) / len(scores)
    
    return results


def evaluate_corpus(
    corpus_type: str,
    output_dir: str,
    embedding_model: str = "all-mpnet-base-v2"
) -> None:
    """
    Run comprehensive evaluation on a corpus.
    
    Args:
        corpus_type: "physics" or "general"
        output_dir: Directory to save results
        embedding_model: Embedding model for dense retrieval
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"RAG Evaluation: {corpus_type.upper()} Corpus")
    print(f"{'='*80}")
    
    # Load test cases
    if corpus_type == "physics":
        test_cases = PhysicsTestSet.get_test_cases()
        bm25_index_path = "data/retrieval/bm25_physics_papers_index.json"
        dense_index_path = "data/retrieval/dense_physics/dense_index.faiss"
        dense_metadata_path = "data/retrieval/dense_physics/dense_index_metadata.jsonl"
        output_prefix = "rag_physics"
    elif corpus_type == "general":
        # Load general corpus test cases from existing sample set
        with open("results/retrieval_baseline/retrieval_quality_sample_set.json", "r") as f:
            sample_data = json.load(f)
        
        # Convert to test cases format
        test_cases = []
        for q_data in sample_data.get("test_cases", [])[:8]:  # Use first 8 queries
            test_cases.append({
                "query_id": q_data.get("query_id"),
                "query": q_data.get("query"),
                "expected_answer": "various",
                "expected_doc_ids": [],  # General corpus: no specific expected docs for demo
                "domain": "Literature",
                "difficulty": "low"
            })
        
        bm25_index_path = "data/retrieval/bm25_index.json"
        # For general corpus, we might not have dense index yet, so handle gracefully
        dense_index_path = "data/retrieval/dense_general/dense_index.faiss"
        dense_metadata_path = "data/retrieval/dense_general/dense_index_metadata.jsonl"
        output_prefix = "rag_general"
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")
    
    print(f"\nTest cases: {len(test_cases)}")
    for tc in test_cases:
        print(f"  - {tc.get('query_id', 'unknown')}: {tc['query'][:50]}...")
    
    # Initialize retrievers
    print(f"\nInitializing retrievers...")
    
    # BM25
    with open(bm25_index_path, "r") as f:
        bm25_index = json.load(f)
    bm25_retriever = SimpleBM25Retriever(bm25_index)
    print(f"✓ BM25 retriever loaded")
    
    # Dense (only if exists)
    dense_retriever = None
    try:
        dense_retriever = DenseRetriever(model_name=embedding_model)
        dense_retriever.load_index(dense_index_path, dense_metadata_path)
        print(f"✓ Dense retriever loaded")
    except FileNotFoundError:
        print(f"⚠ Dense index not found at {dense_index_path}, skipping dense retrieval")
    
    # Hybrid (only if dense available)
    hybrid_retriever = None
    if dense_retriever:
        hybrid_retriever = HybridRetriever(
            bm25_index_path=bm25_index_path,
            dense_index_path=dense_index_path,
            dense_metadata_path=dense_metadata_path,
            embedding_model=embedding_model,
            alpha=0.5,
            fusion_method="rrf"
        )
        print(f"✓ Hybrid retriever loaded")
    
    # Run evaluations
    print(f"\nRunning evaluations...")
    evaluation_results = {
        "corpus": corpus_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_cases": len(test_cases),
        "embedding_model": embedding_model,
        "retrievers": {}
    }
    
    # Evaluate BM25
    bm25_results = evaluate_retriever_on_corpus("BM25", bm25_retriever, test_cases)
    evaluation_results["retrievers"]["bm25"] = bm25_results
    
    # Evaluate Dense
    if dense_retriever:
        dense_results = evaluate_retriever_on_corpus("Dense", dense_retriever, test_cases)
        evaluation_results["retrievers"]["dense"] = dense_results
    
    # Evaluate Hybrid
    if hybrid_retriever:
        hybrid_rrf_results = evaluate_retriever_on_corpus("Hybrid (RRF)", hybrid_retriever, test_cases)
        evaluation_results["retrievers"]["hybrid_rrf"] = hybrid_rrf_results
        
        # Also try weighted
        original_alpha = hybrid_retriever.alpha
        hybrid_retriever.alpha = 0.3  # More weight to BM25
        hybrid_weighted_results = evaluate_retriever_on_corpus("Hybrid (Weighted, α=0.3)", hybrid_retriever, test_cases)
        evaluation_results["retrievers"]["hybrid_weighted_03"] = hybrid_weighted_results
        hybrid_retriever.alpha = original_alpha
    
    # Save results
    json_path = output_path / f"{output_prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n✓ Saved JSON results to {json_path}")
    
    # Generate markdown report
    markdown_path = output_path / f"{output_prefix}_results.md"
    generate_markdown_report(evaluation_results, str(markdown_path))
    print(f"✓ Saved markdown report to {markdown_path}")
    
    # Print summary
    print_summary(evaluation_results)


def generate_markdown_report(results: Dict, output_path: str) -> None:
    """Generate markdown report from evaluation results."""
    with open(output_path, "w") as f:
        f.write(f"# RAG Evaluation Report: {results['corpus'].upper()} Corpus\n\n")
        
        f.write(f"**Timestamp:** {results['timestamp']}\n")
        f.write(f"**Embedding Model:** {results['embedding_model']}\n")
        f.write(f"**Test Cases:** {results['num_test_cases']}\n\n")
        
        # Summary table
        f.write("## Retriever Comparison\n\n")
        f.write("| Retriever | Precision@1 | Precision@5 | Recall@1 | Recall@5 | Hit@1 | MRR |\n")
        f.write("|-----------|-------------|-------------|----------|----------|-------|-----|\n")
        
        for retriever_name, retriever_results in results.get("retrievers", {}).items():
            metrics = retriever_results.get("aggregated_metrics", {})
            f.write(f"| {retriever_name} | ")
            f.write(f"{metrics.get('precision@1', 0):.3f} | ")
            f.write(f"{metrics.get('precision@5', 0):.3f} | ")
            f.write(f"{metrics.get('recall@1', 0):.3f} | ")
            f.write(f"{metrics.get('recall@5', 0):.3f} | ")
            f.write(f"{metrics.get('hit@1', 0):.3f} | ")
            f.write(f"{metrics.get('mrr', 0):.3f} |\n")
        
        f.write("\n## Per-Query Results\n\n")
        
        for retriever_name, retriever_results in results.get("retrievers", {}).items():
            f.write(f"### {retriever_name}\n\n")
            
            for qr in retriever_results.get("per_query_results", []):
                f.write(f"**Query {qr['query_id']}:** {qr['query']}\n")
                f.write(f"- Retrieved: {qr['retrieved_ids'][:3]}\n")
                f.write(f"- Precision@1: {qr['metrics'].get('precision@1', 0):.3f}\n")
                f.write(f"- Recall@5: {qr['metrics'].get('recall@5', 0):.3f}\n")
                f.write(f"- MRR: {qr['metrics'].get('mrr', 0):.3f}\n\n")


def print_summary(results: Dict) -> None:
    """Print evaluation summary to console."""
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY: {results['corpus'].upper()}")
    print(f"{'='*80}\n")
    
    print("Aggregated Metrics by Retriever:\n")
    
    for retriever_name, retriever_results in results.get("retrievers", {}).items():
        metrics = retriever_results.get("aggregated_metrics", {})
        print(f"{retriever_name}:")
        print(f"  Precision@1: {metrics.get('precision@1', 0):.3f}")
        print(f"  Precision@5: {metrics.get('precision@5', 0):.3f}")
        print(f"  Recall@1:    {metrics.get('recall@1', 0):.3f}")
        print(f"  Recall@5:    {metrics.get('recall@5', 0):.3f}")
        print(f"  Hit@1:       {metrics.get('hit@1', 0):.3f}")
        print(f"  MRR:         {metrics.get('mrr', 0):.3f}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Test Suite")
    parser.add_argument("--corpus", choices=["physics", "general"], default="physics",
                       help="Corpus to evaluate")
    parser.add_argument("--output", default="results/rag_evaluation",
                       help="Output directory for results")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                       help="Embedding model for dense retrieval")
    
    args = parser.parse_args()
    
    evaluate_corpus(args.corpus, args.output, args.model)


if __name__ == "__main__":
    main()
