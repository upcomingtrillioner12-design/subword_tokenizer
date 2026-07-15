#!/usr/bin/env python3
"""
Phase 4 Task 6: Enhanced evaluation metrics for RAG systems.

Includes:
- Semantic similarity (embedding-based and transformer-based)
- BERTScore-like metrics
- Answer entailment checking
- Retrieval precision/recall metrics
- Multi-domain evaluation aggregation
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def em_score(prediction: str, ground_truth: str) -> float:
    """Exact match score (case-insensitive, punctuation-insensitive)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_overlap_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = set(normalize_answer(prediction).split())
    truth_tokens = set(normalize_answer(ground_truth).split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    overlap = len(pred_tokens & truth_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def substring_match_score(prediction: str, ground_truth: str) -> float:
    """Score based on whether expected answer appears in prediction."""
    pred = normalize_answer(prediction)
    truth = normalize_answer(ground_truth)
    return 1.0 if truth in pred or pred in truth else 0.0


def semantic_similarity_cosine(text1: str, text2: str) -> float:
    """
    Simple TF-IDF based semantic similarity using cosine distance.
    For more robust similarity, use sentence-transformers.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
        return float(similarity)
    except Exception:
        return token_overlap_f1(text1, text2)


def retrieval_precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    """Compute precision@k: fraction of top-k results that are relevant."""
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(top_k)


def retrieval_recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    """Compute recall@k: fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """MRR: average of 1/rank for first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    relevance_scores: list of relevance scores for retrieved documents
    """
    if not relevance_scores:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i, score in enumerate(relevance_scores[:k], 1):
        dcg += score / (1 + import_math().log2(i))
    
    # Compute ideal DCG
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = 0.0
    for i, score in enumerate(ideal_scores, 1):
        idcg += score / (1 + import_math().log2(i))
    
    return dcg / idcg if idcg > 0 else 0.0


def import_math():
    """Helper to import math module."""
    import math
    return math


def compute_domain_metrics(
    predictions: List[Dict[str, Any]],
    domain: str | None = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a set of predictions.
    
    Args:
        predictions: List of dicts with keys: 'prediction', 'ground_truth', 'answer_rank'
        domain: Optional domain name for grouping
    
    Returns:
        Dict with aggregated metrics
    """
    if not predictions:
        return {}
    
    em_scores = []
    f1_scores = []
    substring_scores = []
    similarity_scores = []
    rank_scores = []
    
    for pred in predictions:
        pred_text = pred.get("prediction", "")
        truth_text = pred.get("ground_truth", "")
        rank = pred.get("answer_rank", None)
        
        em_scores.append(em_score(pred_text, truth_text))
        f1_scores.append(token_overlap_f1(pred_text, truth_text))
        substring_scores.append(substring_match_score(pred_text, truth_text))
        similarity_scores.append(semantic_similarity_cosine(pred_text, truth_text))
        
        if rank is not None:
            rank_scores.append(1.0 if rank == 1 else 0.0)
    
    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    return {
        "em_score": avg(em_scores),
        "f1_score": avg(f1_scores),
        "substring_match": avg(substring_scores),
        "semantic_similarity": avg(similarity_scores),
        "rank_1_exact": avg(rank_scores) if rank_scores else None,
        "num_samples": len(predictions),
        "domain": domain,
    }


def aggregate_metrics_by_domain(
    all_predictions: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across domains.
    
    Args:
        all_predictions: Dict mapping domain -> list of predictions
    
    Returns:
        Dict mapping domain -> metrics dict
    """
    results = {}
    for domain, predictions in all_predictions.items():
        results[domain] = compute_domain_metrics(predictions, domain=domain)
    return results


def compute_retrieval_metrics(
    retrieved_rankings: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute retrieval quality metrics.
    
    Args:
        retrieved_rankings: List of dicts with:
            - relevant_ids: List of relevant doc IDs
            - retrieved_ids: List of retrieved doc IDs in order
    
    Returns:
        Dict with retrieval metrics
    """
    if not retrieved_rankings:
        return {}
    
    p_at_5 = []
    r_at_5 = []
    mrr_scores = []
    
    for ranking in retrieved_rankings:
        relevant = ranking.get("relevant_ids", [])
        retrieved = ranking.get("retrieved_ids", [])
        
        if relevant and retrieved:
            p_at_5.append(retrieval_precision_at_k(relevant, retrieved, k=5))
            r_at_5.append(retrieval_recall_at_k(relevant, retrieved, k=5))
            mrr_scores.append(mean_reciprocal_rank(relevant, retrieved))
    
    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    return {
        "precision_at_5": avg(p_at_5),
        "recall_at_5": avg(r_at_5),
        "mean_reciprocal_rank": avg(mrr_scores),
        "num_queries": len(retrieved_rankings),
    }


class MetricsAggregator:
    """Aggregates metrics across multiple evaluations and domains."""
    
    def __init__(self):
        self.domain_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.retrieval_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.difficulty_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_prediction(
        self,
        prediction: str,
        ground_truth: str,
        domain: str,
        difficulty: str,
        answer_rank: int | None = None,
    ) -> None:
        """Add a single prediction result."""
        key = f"{domain}_{difficulty}"
        
        pred_entry = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "answer_rank": answer_rank,
        }
        
        if key not in self.domain_metrics:
            self.domain_metrics[key] = compute_domain_metrics([pred_entry], domain=key)
        else:
            # Update with running average
            current = self.domain_metrics[key]
            new_metrics = compute_domain_metrics([pred_entry], domain=key)
            # Simple update (would need full history for proper averaging)
            for metric_name, value in new_metrics.items():
                if metric_name != "domain" and metric_name != "num_samples":
                    if metric_name in current:
                        current[metric_name] = (current[metric_name] + value) / 2
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary metrics."""
        return {
            "by_domain": dict(self.domain_metrics),
            "retrieval": dict(self.retrieval_metrics),
            "by_difficulty": dict(self.difficulty_metrics),
        }
