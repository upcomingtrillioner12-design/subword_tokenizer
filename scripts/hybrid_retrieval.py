#!/usr/bin/env python3
"""
Hybrid Retrieval Engine

Combines sparse BM25 retrieval with dense semantic retrieval.
Supports multiple fusion strategies (RRF, linear weighted combination).

Features:
- BM25 retrieval for keyword matching
- Dense retrieval for semantic similarity
- Reciprocal Rank Fusion (RRF) for combining results
- Weighted linear combination with configurable alpha
- Normalized scoring for fair comparison
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

# Existing modules
from dense_retrieval import DenseRetriever


class SimpleBM25Retriever:
    """Wrapper around BM25 index for retrieval."""
    
    def __init__(self, index: Dict):
        self.index = index
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k results for query."""
        import re
        import math
        from collections import Counter
        
        TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
        
        def tokenize(text: str):
            return TOKEN_RE.findall(text.lower())
        
        q_terms = tokenize(query)
        
        # Compute BM25 scores
        scores = defaultdict(float)
        N = self.index["meta"]["num_docs"]
        avgdl = self.index["meta"]["avgdl"]
        doc_lens = self.index["doc_lens"]
        doc_freq = self.index["doc_freq"]
        postings = self.index["postings"]
        
        qtf = Counter(q_terms)
        for term, q_count in qtf.items():
            df = doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.get(term, []):
                dl = doc_lens[doc_id]
                k1, b = 1.5, 0.75
                denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
                term_score = idf * (tf * (k1 + 1)) / max(denom, 1e-9)
                scores[doc_id] += q_count * term_score
        
        # Rank and return
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



class HybridRetriever:
    """
    Hybrid retriever combining BM25 (sparse) and dense (semantic) retrieval.
    
    Attributes:
        bm25: BM25 retriever instance
        dense: Dense retriever instance
        alpha: Weight for dense scores (1-alpha for BM25)
        fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted" (linear combination)
    """
    
    def __init__(
        self,
        bm25_index_path: str,
        dense_index_path: str,
        dense_metadata_path: str,
        embedding_model: str = "all-mpnet-base-v2",
        alpha: float = 0.5,
        fusion_method: str = "rrf"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_index_path: Path to BM25 index JSON
            dense_index_path: Path to dense index .faiss file
            dense_metadata_path: Path to dense metadata .jsonl file
            embedding_model: Embedding model name for dense retriever
            alpha: Weight for dense scores in [0, 1]
                   - 0.0 = pure BM25
                   - 0.5 = equal weighting
                   - 1.0 = pure dense
            fusion_method: "rrf" or "weighted"
        """
        print("Initializing hybrid retriever...")
        print(f"  BM25 index: {bm25_index_path}")
        print(f"  Dense index: {dense_index_path}")
        print(f"  Fusion method: {fusion_method}, alpha={alpha}")
        
        # Load BM25 retriever
        with open(bm25_index_path, "r") as f:
            bm25_index = json.load(f)
        self.bm25 = SimpleBM25Retriever(bm25_index)
        
        # Load dense retriever
        self.dense = DenseRetriever(model_name=embedding_model)
        self.dense.load_index(dense_index_path, dense_metadata_path)
        
        self.alpha = alpha
        self.fusion_method = fusion_method
        print("✓ Hybrid retriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        k_candidate: Optional[int] = None,
        fusion_method: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve top-k documents using hybrid fusion.
        
        Args:
            query: Query text
            k: Number of final results to return
            k_candidate: Number of candidates from each retriever (default: 2*k)
            fusion_method: Override instance fusion method
            alpha: Override instance alpha weight
        
        Returns:
            List of dicts with keys: rank, doc_id, chunk_id, text, score, source,
                                    bm25_rank, dense_rank, bm25_score, dense_score
        """
        fusion_method = fusion_method or self.fusion_method
        alpha = alpha if alpha is not None else self.alpha
        k_candidate = k_candidate or max(10, 2 * k)
        
        # 1. Get BM25 results
        bm25_results = self.bm25.retrieve(query, k=k_candidate)
        
        # 2. Get dense results
        dense_results = self.dense.query(query, k=k_candidate)
        
        # 3. Fusion
        if fusion_method.lower() == "rrf":
            fused_results = self._fuse_rrf(bm25_results, dense_results, k)
        elif fusion_method.lower() == "weighted":
            fused_results = self._fuse_weighted(bm25_results, dense_results, k, alpha)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused_results
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        fusion_method: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Retrieve top-k documents for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            fusion_method: Override instance fusion method
            alpha: Override instance alpha weight
        
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, k=k, fusion_method=fusion_method, alpha=alpha))
        return results
    
    def benchmark_retrievers(
        self,
        queries: List[str],
        k: int = 5
    ) -> Dict:
        """
        Benchmark BM25 vs Dense vs Hybrid on test queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
        
        Returns:
            Dict with per-retriever and per-query comparison
        """
        results = {
            "bm25": [],
            "dense": [],
            "hybrid_rrf": [],
            "hybrid_weighted": []
        }
        
        print(f"\nBenchmarking on {len(queries)} queries...")
        
        for i, query in enumerate(queries, 1):
            # BM25
            bm25_res = self.bm25.query(query, k=k)
            results["bm25"].append(bm25_res)
            
            # Dense
            dense_res = self.dense.query(query, k=k)
            results["dense"].append(dense_res)
            
            # Hybrid RRF
            hybrid_rrf_res = self.retrieve(query, k=k, fusion_method="rrf")
            results["hybrid_rrf"].append(hybrid_rrf_res)
            
            # Hybrid weighted
            hybrid_weighted_res = self.retrieve(query, k=k, fusion_method="weighted", alpha=0.5)
            results["hybrid_weighted"].append(hybrid_weighted_res)
            
            if i % 5 == 0:
                print(f"  Benchmarked {i}/{len(queries)} queries")
        
        return results
    
    def set_fusion_weights(self, alpha: float) -> None:
        """
        Dynamically adjust BM25 vs dense balance.
        
        Args:
            alpha: New weight (0.0 = pure BM25, 1.0 = pure dense)
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in [0, 1]")
        self.alpha = alpha
        print(f"Updated fusion weight: alpha={alpha}")
    
    # ========== Fusion Strategies ==========
    
    def _fuse_rrf(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        k: int
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        
        Score(d) = sum over all systems: 1 / (60 + rank(d))
        
        Args:
            bm25_results: BM25 ranking
            dense_results: Dense ranking
            k: Final number of results to return
        
        Returns:
            Fused ranking
        """
        # Build mapping of doc_id -> (bm25_rank, dense_rank)
        doc_ranks = defaultdict(lambda: {"bm25": None, "dense": None})
        
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["doc_id"]
            doc_ranks[doc_id]["bm25"] = rank
            doc_ranks[doc_id]["bm25_score"] = result.get("score", 0)
            doc_ranks[doc_id]["chunk_id"] = result.get("chunk_id", result["doc_id"])
            doc_ranks[doc_id]["text"] = result.get("text", "")
            doc_ranks[doc_id]["source"] = result.get("source", "")
        
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["doc_id"]
            doc_ranks[doc_id]["dense"] = rank
            doc_ranks[doc_id]["dense_score"] = result.get("score", 0)
            doc_ranks[doc_id]["chunk_id"] = result.get("chunk_id", result["doc_id"])
            doc_ranks[doc_id]["text"] = result.get("text", "")
            doc_ranks[doc_id]["source"] = result.get("source", "")
        
        # Compute RRF scores
        rrf_scores = {}
        for doc_id, ranks in doc_ranks.items():
            score = 0.0
            if ranks["bm25"] is not None:
                score += 1.0 / (60 + ranks["bm25"])
            if ranks["dense"] is not None:
                score += 1.0 / (60 + ranks["dense"])
            rrf_scores[doc_id] = score
        
        # Sort and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for final_rank, (doc_id, score) in enumerate(sorted_docs[:k], 1):
            ranks = doc_ranks[doc_id]
            results.append({
                "rank": final_rank,
                "doc_id": doc_id,
                "chunk_id": ranks["chunk_id"],
                "text": ranks["text"],
                "score": score,
                "source": ranks["source"],
                "bm25_rank": ranks["bm25"],
                "dense_rank": ranks["dense"],
                "bm25_score": ranks.get("bm25_score", None),
                "dense_score": ranks.get("dense_score", None)
            })
        
        return results
    
    def _fuse_weighted(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        k: int,
        alpha: float
    ) -> List[Dict]:
        """
        Weighted linear combination.
        
        Score(d) = (1 - alpha) * norm_bm25(d) + alpha * norm_dense(d)
        
        Args:
            bm25_results: BM25 ranking
            dense_results: Dense ranking
            k: Final number of results to return
            alpha: Weight for dense (1-alpha for BM25)
        
        Returns:
            Fused ranking
        """
        # Normalize BM25 scores to [0, 1]
        bm25_scores = [r.get("score", 0) for r in bm25_results]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        
        # Normalize dense scores to [0, 1] (inverse since lower L2 is better)
        # L2 distances: lower is better, so we invert with max(0, 1 - score)
        dense_scores = [r.get("score", 0) for r in dense_results]
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_norm_scale = dense_max + 1.0  # Scale to [0, 1]
        
        # Build doc score map
        doc_scores = defaultdict(lambda: {"bm25": None, "dense": None})
        
        for result in bm25_results:
            doc_id = result["doc_id"]
            norm_score = (result.get("score", 0) - bm25_min) / bm25_range if bm25_range > 0 else 0
            doc_scores[doc_id]["bm25"] = norm_score
            doc_scores[doc_id]["bm25_raw"] = result.get("score", 0)
            doc_scores[doc_id]["chunk_id"] = result.get("chunk_id", result["doc_id"])
            doc_scores[doc_id]["text"] = result.get("text", "")
            doc_scores[doc_id]["source"] = result.get("source", "")
        
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["doc_id"]
            # L2 distance: smaller is better, so invert
            raw_score = result.get("score", 0)
            norm_score = max(0, 1.0 - (raw_score / dense_norm_scale))
            doc_scores[doc_id]["dense"] = norm_score
            doc_scores[doc_id]["dense_raw"] = raw_score
            doc_scores[doc_id]["dense_rank"] = rank
            doc_scores[doc_id]["chunk_id"] = result.get("chunk_id", result["doc_id"])
            doc_scores[doc_id]["text"] = result.get("text", "")
            doc_scores[doc_id]["source"] = result.get("source", "")
        
        # Compute weighted scores
        combined_scores = {}
        for doc_id, scores in doc_scores.items():
            bm25_norm = scores["bm25"] or 0.0
            dense_norm = scores["dense"] or 0.0
            combined = (1 - alpha) * bm25_norm + alpha * dense_norm
            combined_scores[doc_id] = combined
        
        # Sort and return top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for final_rank, (doc_id, score) in enumerate(sorted_docs[:k], 1):
            scores = doc_scores[doc_id]
            results.append({
                "rank": final_rank,
                "doc_id": doc_id,
                "chunk_id": scores.get("chunk_id"),
                "text": scores.get("text", ""),
                "score": score,
                "source": scores.get("source", ""),
                "bm25_rank": None,  # Not tracked in weighted fusion
                "dense_rank": scores.get("dense_rank", None),
                "bm25_score": scores.get("bm25_raw", None),
                "dense_score": scores.get("dense_raw", None)
            })
        
        return results


def main():
    """CLI interface for hybrid retrieval."""
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Engine")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query hybrid index")
    query_parser.add_argument("--bm25-index", required=True, help="Path to BM25 index JSON")
    query_parser.add_argument("--dense-index", required=True, help="Path to dense index .faiss")
    query_parser.add_argument("--dense-metadata", required=True, help="Path to dense metadata .jsonl")
    query_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    query_parser.add_argument("--q", required=True, help="Query text")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--alpha", type=float, default=0.5, help="Dense weight [0, 1]")
    query_parser.add_argument("--method", default="rrf", choices=["rrf", "weighted"],
                             help="Fusion method")
    
    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark retrievers")
    benchmark_parser.add_argument("--bm25-index", required=True, help="Path to BM25 index JSON")
    benchmark_parser.add_argument("--dense-index", required=True, help="Path to dense index .faiss")
    benchmark_parser.add_argument("--dense-metadata", required=True, help="Path to dense metadata .jsonl")
    benchmark_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    benchmark_parser.add_argument("--queries", required=True, help="Path to queries JSONL")
    benchmark_parser.add_argument("--k", type=int, default=5, help="Number of results")
    benchmark_parser.add_argument("--output", help="Output file for results JSON")
    
    args = parser.parse_args()
    
    if args.command == "query":
        retriever = HybridRetriever(
            bm25_index_path=args.bm25_index,
            dense_index_path=args.dense_index,
            dense_metadata_path=args.dense_metadata,
            embedding_model=args.model,
            alpha=args.alpha,
            fusion_method=args.method
        )
        
        print(f"\nQuery: {args.q}\n")
        print(f"Fusion: {args.method}, alpha={args.alpha}\n")
        
        results = retriever.retrieve(args.q, k=args.k)
        
        for result in results:
            print(f"[{result['rank']}] score={result['score']:.3f}", end="")
            if result["bm25_rank"]:
                print(f" | BM25 rank={result['bm25_rank']}", end="")
            if result["dense_rank"]:
                print(f" | Dense rank={result['dense_rank']}", end="")
            print(f" | {result['doc_id']}")
            print(f"    {result['text'][:100]}...\n")
    
    elif args.command == "benchmark":
        retriever = HybridRetriever(
            bm25_index_path=args.bm25_index,
            dense_index_path=args.dense_index,
            dense_metadata_path=args.dense_metadata,
            embedding_model=args.model
        )
        
        # Load queries
        queries = []
        with open(args.queries, "r") as f:
            for line in f:
                doc = json.loads(line)
                queries.append(doc.get("query", ""))
        
        # Run benchmark
        bench_results = retriever.benchmark_retrievers(queries, k=args.k)
        
        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(bench_results, f, indent=2)
            print(f"Saved benchmark results to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
