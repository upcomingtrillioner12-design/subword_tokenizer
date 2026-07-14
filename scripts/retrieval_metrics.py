#!/usr/bin/env python3
"""
Retrieval Evaluation Metrics

Implements standard information retrieval metrics for evaluating retrieval quality:
- Precision@k, Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)

All metrics work with binary relevance (relevant/not relevant).
"""

from typing import List, Set, Dict, Tuple, Optional
import numpy as np


class RetrievalMetrics:
    """Compute standard IR evaluation metrics."""
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: int
    ) -> float:
        """
        Precision@k: fraction of top-k results that are relevant.
        
        Precision@k = |relevant ∩ retrieved| / k
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Cutoff position
        
        Returns:
            Precision@k in [0, 1]
        """
        retrieved_at_k = set(retrieved_ids[:k])
        relevant = retrieved_at_k & expected_ids
        return len(relevant) / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: int
    ) -> float:
        """
        Recall@k: fraction of relevant docs found in top-k.
        
        Recall@k = |relevant ∩ retrieved| / |relevant|
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Cutoff position
        
        Returns:
            Recall@k in [0, 1]
        """
        if not expected_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant = retrieved_at_k & expected_ids
        return len(relevant) / len(expected_ids)
    
    @staticmethod
    def mrr(
        retrieved_ids: List[str],
        expected_ids: Set[str]
    ) -> float:
        """
        Mean Reciprocal Rank: reciprocal of rank of first relevant item.
        
        MRR = 1 / rank_of_first_relevant
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
        
        Returns:
            MRR in [0, 1]
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in expected_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@k.
        
        DCG@k = sum_{i=1}^{k} rel(i) / log2(i+1)
        NDCG@k = DCG@k / iDCG@k
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Cutoff position
        
        Returns:
            NDCG@k in [0, 1]
        """
        # Compute DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            rel = 1.0 if doc_id in expected_ids else 0.0
            dcg += rel / np.log2(i + 1)
        
        # Compute ideal DCG (all relevant docs first)
        idcg = 0.0
        for i in range(1, min(k, len(expected_ids)) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_score(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: Optional[int] = None
    ) -> float:
        """
        Mean Average Precision: average of precisions at positions of relevant items.
        
        AP = sum_{i=1}^{n} P(i) * rel(i) / |relevant|
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Optional cutoff position (if None, use all)
        
        Returns:
            MAP in [0, 1]
        """
        if not expected_ids:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k] if k else retrieved_ids
        
        score = 0.0
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved_at_k, 1):
            if doc_id in expected_ids:
                relevant_found += 1
                precision_at_i = relevant_found / i
                score += precision_at_i
        
        return score / len(expected_ids)
    
    @staticmethod
    def f1_score(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: int
    ) -> float:
        """
        F1 Score: harmonic mean of precision and recall at k.
        
        F1 = 2 * (P * R) / (P + R)
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Cutoff position
        
        Returns:
            F1 in [0, 1]
        """
        precision = RetrievalMetrics.precision_at_k(retrieved_ids, expected_ids, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_ids, expected_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: List[str],
        expected_ids: Set[str],
        k: int
    ) -> float:
        """
        Hit Rate@k: whether at least one relevant doc is in top-k.
        
        Hit@k = 1 if |relevant ∩ retrieved| > 0 else 0
        
        Args:
            retrieved_ids: List of retrieved doc IDs (in order)
            expected_ids: Set of relevant doc IDs
            k: Cutoff position
        
        Returns:
            Hit rate in {0.0, 1.0}
        """
        retrieved_at_k = set(retrieved_ids[:k])
        return 1.0 if (retrieved_at_k & expected_ids) else 0.0


def evaluate_single_query(
    retrieved_ids: List[str],
    expected_ids: Set[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate a single query against expected results.
    
    Args:
        retrieved_ids: List of retrieved doc IDs
        expected_ids: Set of relevant doc IDs
        k_values: List of k values for @k metrics
    
    Returns:
        Dict of metric_name -> score
    """
    metrics = {}
    
    # MRR and MAP (not dependent on k)
    metrics["mrr"] = RetrievalMetrics.mrr(retrieved_ids, expected_ids)
    metrics["map"] = RetrievalMetrics.map_score(retrieved_ids, expected_ids)
    
    # Metrics for each k
    for k in k_values:
        metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved_ids, expected_ids, k)
        metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, expected_ids, k)
        metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, expected_ids, k)
        metrics[f"f1@{k}"] = RetrievalMetrics.f1_score(retrieved_ids, expected_ids, k)
        metrics[f"hit@{k}"] = RetrievalMetrics.hit_rate_at_k(retrieved_ids, expected_ids, k)
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries (averaging).
    
    Args:
        all_metrics: List of metric dicts (one per query)
    
    Returns:
        Aggregated metrics with averages
    """
    if not all_metrics:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for m in all_metrics:
        metric_names.update(m.keys())
    
    # Average each metric
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            aggregated[metric_name] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
    
    return aggregated


# Example usage
if __name__ == "__main__":
    # Test metrics
    retrieved = ["doc1", "doc3", "doc5", "doc7", "doc2", "doc8", "doc10"]
    expected = {"doc1", "doc2", "doc3"}
    
    metrics = evaluate_single_query(retrieved, expected)
    print("Single query metrics:")
    for name, score in sorted(metrics.items()):
        print(f"  {name:15s} = {score:.4f}")
    
    # Test aggregation
    print("\nTest aggregation:")
    all_metrics = [
        evaluate_single_query(retrieved, expected),
        evaluate_single_query(retrieved, expected)
    ]
    agg = aggregate_metrics(all_metrics)
    for name, score in sorted(agg.items()):
        print(f"  {name:20s} = {score:.4f}")
