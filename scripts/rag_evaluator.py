#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Evaluator

Evaluates retrieval quality and can optionally evaluate end-to-end generation quality.
Provides metrics for both retrieval and RAG tasks.

Features:
- Retrieval quality evaluation (precision, recall, NDCG, etc.)
- RAG quality metrics (context relevance, answer relevance, faithfulness)
- Batch evaluation on multiple test cases
- Result aggregation and reporting
"""

import json
from typing import List, Dict, Set, Optional, Any
from pathlib import Path
from retrieval_metrics import evaluate_single_query, aggregate_metrics


class RAGEvaluator:
    """
    Evaluator for RAG (Retrieval-Augmented Generation) systems.
    
    Can evaluate:
    1. Retrieval quality alone (requires expected doc IDs)
    2. RAG quality with generation (requires generator and expected answers)
    """
    
    def __init__(self, retriever, generator=None):
        """
        Initialize RAG evaluator.
        
        Args:
            retriever: Retriever instance (with retrieve() method)
            generator: Optional LLM/generator for RAG evaluation (with generate() method)
        """
        self.retriever = retriever
        self.generator = generator
    
    def evaluate_retrieval(
        self,
        query: str,
        expected_doc_ids: Set[str],
        k: int = 5,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for a single query.
        
        Args:
            query: Query text
            expected_doc_ids: Set of relevant document IDs
            k: Number of results to retrieve
            k_values: K values for @k metrics
        
        Returns:
            Dict with retrieval metrics and results
        """
        # Retrieve
        retrieved_results = self.retriever.retrieve(query, k=k)
        retrieved_ids = [r["doc_id"] for r in retrieved_results]
        
        # Compute metrics
        metrics = evaluate_single_query(retrieved_ids, expected_doc_ids, k_values=k_values)
        
        return {
            "query": query,
            "expected_doc_ids": list(expected_doc_ids),
            "retrieved_results": retrieved_results,
            "metrics": metrics,
            "num_expected": len(expected_doc_ids),
            "num_retrieved": len(retrieved_ids)
        }
    
    def evaluate_rag(
        self,
        query: str,
        expected_answer: str,
        expected_doc_ids: Set[str],
        k: int = 5,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG (retrieval + generation).
        
        Args:
            query: Query text
            expected_answer: Expected/reference answer
            expected_doc_ids: Set of relevant document IDs
            k: Number of results to retrieve
            k_values: K values for @k metrics
        
        Returns:
            Dict with retrieval + generation metrics
        """
        if self.generator is None:
            raise ValueError("Generator required for RAG evaluation. Provide in __init__.")
        
        # Retrieve
        retrieval_eval = self.evaluate_retrieval(query, expected_doc_ids, k=k, k_values=k_values)
        
        # Build context from retrieved docs
        context = "\n\n".join([
            f"Document {i+1}: {r['text']}"
            for i, r in enumerate(retrieval_eval["retrieved_results"])
        ])
        
        # Generate
        generated_answer = self.generator.generate(query, context)
        
        # Compute RAG-specific metrics
        rag_metrics = self._compute_rag_metrics(generated_answer, context, expected_answer)
        
        return {
            "query": query,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "context": context,
            "retrieval_metrics": retrieval_eval["metrics"],
            "rag_metrics": rag_metrics,
            "retrieved_results": retrieval_eval["retrieved_results"]
        }
    
    def batch_evaluate(
        self,
        test_cases: List[Dict],
        eval_type: str = "retrieval",
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple test cases.
        
        Args:
            test_cases: List of test case dicts with keys:
                - query (required)
                - expected_doc_ids (required)
                - expected_answer (required for RAG eval)
            eval_type: "retrieval" or "rag"
            k: Number of results to retrieve
        
        Returns:
            Dict with per-query results and aggregated metrics
        """
        results = []
        
        for tc in test_cases:
            query = tc["query"]
            expected_doc_ids = set(tc.get("expected_doc_ids", []))
            
            if eval_type == "retrieval":
                result = self.evaluate_retrieval(query, expected_doc_ids, k=k)
            elif eval_type == "rag":
                expected_answer = tc.get("expected_answer", "")
                result = self.evaluate_rag(query, expected_answer, expected_doc_ids, k=k)
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")
            
            results.append(result)
        
        # Aggregate metrics
        if eval_type == "retrieval":
            all_metrics = [r["metrics"] for r in results]
        else:
            all_metrics = [r["retrieval_metrics"] for r in results]
        
        aggregated = aggregate_metrics(all_metrics)
        
        return {
            "eval_type": eval_type,
            "num_queries": len(test_cases),
            "per_query_results": results,
            "aggregated_metrics": aggregated
        }
    
    def _compute_rag_metrics(
        self,
        generated_answer: str,
        context: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """
        Compute RAG-specific quality metrics.
        
        Args:
            generated_answer: Generated answer text
            context: Retrieved context
            expected_answer: Expected reference answer
        
        Returns:
            Dict with RAG metrics (to be expanded with LLM evaluation)
        """
        metrics = {}
        
        # 1. Context Relevance (TODO: implement with LLM)
        # For now, simple heuristic: overlap with expected answer
        gen_words = set(generated_answer.lower().split())
        exp_words = set(expected_answer.lower().split())
        overlap = gen_words & exp_words
        metrics["word_overlap_ratio"] = len(overlap) / len(exp_words) if exp_words else 0.0
        
        # 2. Answer Relevance (TODO: implement with LLM)
        # Simple check: does generated answer contain key terms from expected?
        key_terms = {w for w in exp_words if len(w) > 4}  # Rough heuristic
        found_terms = {t for t in key_terms if t in generated_answer.lower()}
        metrics["key_term_coverage"] = len(found_terms) / len(key_terms) if key_terms else 1.0
        
        # 3. Faithfulness (TODO: implement with LLM)
        # Simple check: does answer only use words from context?
        context_words = set(context.lower().split())
        unfaithful_words = gen_words - context_words
        metrics["faithfulness_score"] = 1.0 - (len(unfaithful_words) / len(gen_words)) if gen_words else 1.0
        
        # 4. Length metrics
        metrics["answer_length"] = len(generated_answer.split())
        
        return metrics
    
    def compare_retrievers(
        self,
        query: str,
        expected_doc_ids: Set[str],
        retrievers: Dict[str, Any],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare multiple retrievers on the same query.
        
        Args:
            query: Query text
            expected_doc_ids: Set of relevant document IDs
            retrievers: Dict {name: retriever_instance}
            k: Number of results
        
        Returns:
            Dict with comparison results
        """
        comparison = {
            "query": query,
            "expected_doc_ids": list(expected_doc_ids),
            "k": k,
            "retriever_results": {}
        }
        
        for name, retriever in retrievers.items():
            # Temporarily swap retrievers
            original_retriever = self.retriever
            self.retriever = retriever
            
            eval_result = self.evaluate_retrieval(query, expected_doc_ids, k=k)
            comparison["retriever_results"][name] = {
                "metrics": eval_result["metrics"],
                "top_results": eval_result["retrieved_results"][:3]
            }
            
            self.retriever = original_retriever
        
        return comparison
    
    def export_results(
        self,
        results: Dict,
        output_json: Optional[str] = None,
        output_markdown: Optional[str] = None
    ) -> None:
        """
        Export evaluation results to JSON and optional markdown report.
        
        Args:
            results: Evaluation results dict
            output_json: Path to save JSON results
            output_markdown: Path to save markdown report
        """
        # Save JSON
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved JSON results to {output_json}")
        
        # Save Markdown report
        if output_markdown:
            self._generate_markdown_report(results, output_markdown)
    
    def _generate_markdown_report(self, results: Dict, output_path: str) -> None:
        """Generate markdown report from evaluation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(f"# RAG Evaluation Report\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Evaluation Type:** {results.get('eval_type', 'N/A')}\n")
            f.write(f"- **Number of Queries:** {results.get('num_queries', 'N/A')}\n\n")
            
            # Aggregated Metrics
            f.write("## Aggregated Metrics\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            for metric, score in sorted(results.get("aggregated_metrics", {}).items()):
                f.write(f"| {metric} | {score:.4f} |\n")
            
            # Per-query Results
            f.write("\n## Per-Query Results\n\n")
            for i, qr in enumerate(results.get("per_query_results", []), 1):
                f.write(f"### Query {i}: {qr.get('query', 'N/A')}\n\n")
                
                f.write("**Metrics:**\n")
                f.write("| Metric | Score |\n")
                f.write("|--------|-------|\n")
                for metric, score in sorted(qr.get("metrics", {}).items()):
                    f.write(f"| {metric} | {score:.4f} |\n")
                
                f.write("\n**Retrieved Documents:**\n")
                for j, doc in enumerate(qr.get("retrieved_results", [])[:3], 1):
                    f.write(f"- [{j}] {doc.get('doc_id', 'N/A')}: {doc.get('text', 'N/A')[:80]}...\n")
                
                f.write("\n")
        
        print(f"Saved markdown report to {output_path}")


# Example usage
if __name__ == "__main__":
    print("RAG Evaluator module loaded. Use with retriever instances.")
