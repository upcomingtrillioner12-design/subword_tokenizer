#!/usr/bin/env python3
"""
Phase 4 Task 4: End-to-end RAG generation evaluation with neural reranking.

Pipeline:
1) Hybrid retrieval (BM25 + dense)
2) Optional neural reranking (cross-encoder)
3) LoRA-augmented TinyLM generation with retrieved context
4) Generation quality metrics and report export
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None

from hybrid_retrieval import HybridRetriever
from inference_lora import LoRAInferenceEngine, load_tokenizer
from neural_reranker import CrossEncoderReranker
from semantic_metrics import SemanticMetricsEvaluator
from tool_executor import ToolExecutor


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    txt = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML config. Install with: pip install pyyaml")
        return yaml.safe_load(txt)
    return json.loads(txt)


class TinyLMRAGGenerator:
    def __init__(self, cfg: Dict[str, Any]):
        model_cfg = cfg["model"]
        gen_cfg = cfg["generation"]
        tools_cfg = cfg.get("tools", {})

        self.engine = LoRAInferenceEngine(
            base_checkpoint=Path(model_cfg["base_checkpoint"]),
            lora_checkpoint=Path(model_cfg["lora_checkpoint"]),
            device=model_cfg.get("device", "auto"),
            dtype=model_cfg.get("dtype", "float32"),
            verbose=False,
        )
        self.tokenizer = load_tokenizer(Path(model_cfg["tokenizer_model"]))

        self.max_tokens = int(gen_cfg.get("max_tokens", 48))
        self.temperature = float(gen_cfg.get("temperature", 1.2))
        self.top_k = gen_cfg.get("top_k", 80)
        self.top_p = gen_cfg.get("top_p", None)
        self.prompt_mode = str(gen_cfg.get("prompt_mode", "strict")).lower()
        self.enforce_context_overlap = bool(gen_cfg.get("enforce_context_overlap", True))
        self.faithfulness_floor = float(gen_cfg.get("faithfulness_floor", 0.25))
        self.tool_executor = ToolExecutor(enabled=bool(tools_cfg.get("enabled", False)))

    def _build_prompt(self, query: str, context: str, tool_hints: List[str]) -> str:
        hints_text = "\n".join(f"- {h}" for h in tool_hints)
        hints_block = f"\n\nTool Hints:\n{hints_text}" if tool_hints else ""

        if self.prompt_mode == "strict":
            return (
                "You are a physics research assistant. Use ONLY the provided context. "
                "If the answer is not explicitly in context, reply: 'insufficient context'.\n\n"
                f"Context:\n{context}{hints_block}\n\n"
                f"Question: {query}\n"
                "Answer (short, context-grounded):"
            )

        return (
            "You are a physics research assistant. Answer concisely using the provided context.\n\n"
            f"Context:\n{context}{hints_block}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate(self, query: str, context: str) -> str:
        tools = self.tool_executor.run(query)
        tool_hints = tools.get("hints", [])

        prompt = self._build_prompt(query, context, tool_hints)
        text, _metrics = self.engine.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        answer = text.strip()
        if self.enforce_context_overlap:
            faith = context_faithfulness(answer, context)
            if faith < self.faithfulness_floor:
                answer = extractive_fallback_answer(query, context)

        return answer.strip()

    def score_option(self, query: str, context: str, option_text: str) -> float:
        tools = self.tool_executor.run(query)
        tool_hints = tools.get("hints", [])
        prompt = self._build_prompt(query, context, tool_hints)
        continuation = " " + option_text.strip()
        return self.engine.score_continuation(prompt=prompt, continuation=continuation, tokenizer=self.tokenizer)

def extractive_fallback_answer(query: str, context: str) -> str:
    import re

    query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    chunks = [c.strip() for c in re.split(r"[\n\.]+", context) if c.strip()]
    if not chunks:
        return "insufficient context"

    best = ""
    best_score = -1
    for c in chunks:
        c_terms = set(re.findall(r"[a-zA-Z0-9_]+", c.lower()))
        score = len(query_terms & c_terms)
        if score > best_score:
            best_score = score
            best = c

    if not best:
        return "insufficient context"
    return best[:220]

def token_f1(pred: str, gold: str) -> float:
    import re

    p = re.findall(r"[a-zA-Z0-9_]+", pred.lower())
    g = re.findall(r"[a-zA-Z0-9_]+", gold.lower())
    if not p or not g:
        return 0.0

    p_counts = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    g_counts = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1

    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def context_faithfulness(pred: str, context: str) -> float:
    import re

    p = set(re.findall(r"[a-zA-Z0-9_]+", pred.lower()))
    c = set(re.findall(r"[a-zA-Z0-9_]+", context.lower()))
    if not p:
        return 0.0
    return len(p & c) / len(p)


def build_context(docs: List[Dict[str, Any]], max_docs: int) -> str:
    rows = []
    for i, d in enumerate(docs[:max_docs], start=1):
        src = d.get("source", "unknown")
        txt = d.get("text", "")
        rows.append(f"[{i}] source={src}\n{txt}")
    return "\n\n".join(rows)


def evaluate(cfg: Dict[str, Any], limit: int | None = None) -> Dict[str, Any]:
    retrieval_cfg = cfg["retrieval"]
    reranker_cfg = cfg["reranker"]
    eval_cfg = cfg["evaluation"]

    retriever = HybridRetriever(
        bm25_index_path=retrieval_cfg["bm25_index"],
        dense_index_path=retrieval_cfg["dense_index"],
        dense_metadata_path=retrieval_cfg["dense_metadata"],
        embedding_model=retrieval_cfg.get("embedding_model", "all-mpnet-base-v2"),
        alpha=float(retrieval_cfg.get("alpha", 0.5)),
        fusion_method=retrieval_cfg.get("fusion_method", "rrf"),
    )

    reranker = None
    if bool(reranker_cfg.get("enabled", True)):
        reranker = CrossEncoderReranker(
            model_name=reranker_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            strategy=reranker_cfg.get("strategy", "hybrid"),
            cross_weight=float(reranker_cfg.get("cross_weight", 0.55)),
            semantic_weight=float(reranker_cfg.get("semantic_weight", 0.30)),
            lexical_weight=float(reranker_cfg.get("lexical_weight", 0.15)),
            verbose=True,
        )

    semantic_eval = SemanticMetricsEvaluator(
        config=cfg.get("semantic_metrics", {}),
        enabled=bool(eval_cfg.get("enable_semantic_metrics", False)),
    )

    generator = TinyLMRAGGenerator(cfg)

    qa_path = Path(eval_cfg["dataset"])
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    questions = data.get("questions", data)

    max_questions = int(eval_cfg.get("max_questions", len(questions)))
    if limit is not None:
        max_questions = min(max_questions, limit)
    questions = questions[:max_questions]

    k_retrieve = int(retrieval_cfg.get("k_retrieve", 8))
    context_docs = int(retrieval_cfg.get("context_docs", 3))
    rerank_top_n = int(reranker_cfg.get("top_n", 5))

    started = time.perf_counter()
    per_q: List[Dict[str, Any]] = []

    f1_scores: List[float] = []
    contain_scores: List[float] = []
    faithful_scores: List[float] = []
    mc_exact_scores: List[float] = []
    mc_semantic_scores: List[float] = []
    semantic_similarity_scores: List[float] = []
    bertscore_f1_scores: List[float] = []
    entailment_scores: List[float] = []
    factual_consistency_scores: List[float] = []
    numeric_unit_scores: List[float] = []
    uncertainty_scores: List[float] = []

    eval_mode = str(eval_cfg.get("mode", "generation")).lower()

    for i, q in enumerate(questions, start=1):
        query = q["question"]
        expected = q.get("expected_answer", "")

        retrieved = retriever.retrieve(query, k=k_retrieve)
        final_docs = retrieved
        if reranker is not None:
            final_docs = reranker.rerank(
                query,
                retrieved,
                top_n=rerank_top_n,
                strategy=reranker_cfg.get("strategy", "hybrid"),
            )

        context = build_context(final_docs, max_docs=context_docs)
        if eval_mode == "mc_likelihood" and q.get("distractors"):
            options = [expected] + list(q.get("distractors", []))
            scored = [(opt, generator.score_option(query, context, opt)) for opt in options]
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            answer = scored_sorted[0][0]

            rank_expected = next((idx + 1 for idx, (opt, _) in enumerate(scored_sorted) if opt == expected), 999)
            mc_exact = 1.0 if rank_expected == 1 else 0.0
            mc_semantic = 1.0 if rank_expected <= 2 else 0.0
            mc_exact_scores.append(mc_exact)
            mc_semantic_scores.append(mc_semantic)
        else:
            answer = generator.generate(query, context)
            scored_sorted = None
            rank_expected = None
            mc_exact = None
            mc_semantic = None

        f1 = token_f1(answer, expected)
        contains_expected = 1.0 if expected.lower() in answer.lower() else 0.0
        faith = context_faithfulness(answer, context)

        sem = semantic_eval.evaluate(answer, expected, context)

        f1_scores.append(f1)
        contain_scores.append(contains_expected)
        faithful_scores.append(faith)
        semantic_similarity_scores.append(sem["semantic_similarity"])
        bertscore_f1_scores.append(sem["bertscore_f1"])
        entailment_scores.append(sem["entailment_score"])
        factual_consistency_scores.append(sem["factual_consistency"])
        numeric_unit_scores.append(sem["numeric_unit_consistency"])
        uncertainty_scores.append(sem["uncertainty_score"])

        row = {
            "id": q.get("id", f"q_{i:03d}"),
            "category": q.get("category", "unknown"),
            "query": query,
            "expected_answer": expected,
            "generated_answer": answer,
            "metrics": {
                "token_f1": f1,
                "contains_expected": contains_expected,
                "faithfulness": faith,
                "semantic_similarity": sem["semantic_similarity"],
                "bertscore_f1": sem["bertscore_f1"],
                "entailment_score": sem["entailment_score"],
                "factual_consistency": sem["factual_consistency"],
                "numeric_unit_consistency": sem["numeric_unit_consistency"],
                "uncertainty_score": sem["uncertainty_score"],
                "mc_exact": mc_exact,
                "mc_semantic_or_better": mc_semantic,
                "expected_rank": rank_expected,
            },
            "option_scores": [
                {"option": opt, "avg_logprob": score} for opt, score in (scored_sorted or [])
            ],
            "context": context,
            "retrieved_topk": [
                {
                    "doc_id": d.get("doc_id"),
                    "rank": d.get("rank"),
                    "score": d.get("score"),
                    "rerank_score": d.get("rerank_score"),
                    "rerank_components": d.get("rerank_components"),
                    "source": d.get("source"),
                    "text": d.get("text", ""),
                }
                for d in final_docs[:context_docs]
            ],
        }
        per_q.append(row)
        if eval_mode == "mc_likelihood":
            print(
                f"[{i}/{len(questions)}] {row['id']} rank={rank_expected} "
                f"mc_exact={0.0 if mc_exact is None else mc_exact:.1f}"
            )
        else:
            print(f"[{i}/{len(questions)}] {row['id']} f1={f1:.3f} contains={contains_expected:.1f} faith={faith:.3f}")

    elapsed = time.perf_counter() - started

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    report = {
        "metadata": {
            "task": "phase4_task4_rag_generation_eval",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_questions": len(questions),
            "elapsed_seconds": elapsed,
            "retrieval": retrieval_cfg,
            "reranker": {
                **reranker_cfg,
                "active": bool(reranker is not None),
                "fallback": bool(reranker.fallback) if reranker is not None else None,
            },
            "generation": cfg["generation"],
            "semantic_metrics": {
                **cfg.get("semantic_metrics", {}),
                "active": semantic_eval.active,
                "init_errors": semantic_eval.init_errors,
            },
            "model": {
                "base_checkpoint": cfg["model"]["base_checkpoint"],
                "lora_checkpoint": cfg["model"]["lora_checkpoint"],
                "tokenizer_model": cfg["model"]["tokenizer_model"],
            },
        },
        "summary": {
            "avg_token_f1": avg(f1_scores),
            "avg_contains_expected": avg(contain_scores),
            "avg_faithfulness": avg(faithful_scores),
            "avg_semantic_similarity": avg(semantic_similarity_scores),
            "avg_bertscore_f1": avg(bertscore_f1_scores),
            "avg_entailment_score": avg(entailment_scores),
            "avg_factual_consistency": avg(factual_consistency_scores),
            "avg_numeric_unit_consistency": avg(numeric_unit_scores),
            "avg_uncertainty_score": avg(uncertainty_scores),
            "mc_exact_rate": avg(mc_exact_scores),
            "mc_semantic_or_better_rate": avg(mc_semantic_scores),
        },
        "results": per_q,
    }

    # Apply calibrated uncertainty post-processing if enabled
    if cfg.get("calibrated_uncertainty", {}).get("enabled", False):
        try:
            from calibrated_uncertainty import CalibratedUncertaintyEvaluator, CalibratedUncertaintyConfig
            cal_cfg_dict = cfg.get("calibrated_uncertainty", {})
            cal_cfg = CalibratedUncertaintyConfig(
                logprob_spread_weight=float(cal_cfg_dict.get("logprob_spread_weight", 0.30)),
                context_weight=float(cal_cfg_dict.get("context_weight", 0.25)),
                entailment_weight=float(cal_cfg_dict.get("entailment_weight", 0.25)),
                faithfulness_weight=float(cal_cfg_dict.get("faithfulness_weight", 0.20)),
                calibration_slope=float(cal_cfg_dict.get("calibration_slope", 0.9)),
                calibration_offset=float(cal_cfg_dict.get("calibration_offset", 0.1)),
            )
            evaluator = CalibratedUncertaintyEvaluator(cal_cfg)
            
            calibrated_uncertainty_scores = []
            for result in report["results"]:
                metrics = result["metrics"]
                option_scores = result.get("option_scores", [])
                calibration = evaluator.calibrate(metrics, option_scores)
                result["calibration"] = calibration
                result["metrics"]["calibrated_uncertainty"] = calibration["calibrated_uncertainty"]
                calibrated_uncertainty_scores.append(calibration["calibrated_uncertainty"])
            
            # Update summary with calibrated metrics
            if calibrated_uncertainty_scores:
                report["summary"]["avg_calibrated_uncertainty"] = avg(calibrated_uncertainty_scores)
                report["summary"]["calibrated_uncertainty_range"] = [
                    min(calibrated_uncertainty_scores),
                    max(calibrated_uncertainty_scores),
                ]
                report["metadata"]["calibrated_uncertainty"] = {
                    "enabled": True,
                    "num_calibrated": len(calibrated_uncertainty_scores),
                }
        except Exception as e:
            print(f"Warning: Calibrated uncertainty processing failed: {e}")
    
    return report


def save_report(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"rag_generation_eval_{ts}.json"
    md_path = out_dir / f"rag_generation_eval_{ts}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Phase 4 Task 4: RAG Generation Evaluation")
    lines.append("")
    lines.append(f"- Timestamp: {report['metadata']['timestamp']}")
    lines.append(f"- Questions: {report['metadata']['num_questions']}")
    lines.append(f"- Elapsed: {report['metadata']['elapsed_seconds']:.2f}s")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| avg_token_f1 | {report['summary']['avg_token_f1']:.4f} |")
    lines.append(f"| avg_contains_expected | {report['summary']['avg_contains_expected']:.4f} |")
    lines.append(f"| avg_faithfulness | {report['summary']['avg_faithfulness']:.4f} |")
    lines.append(f"| avg_semantic_similarity | {report['summary'].get('avg_semantic_similarity', 0.0):.4f} |")
    lines.append(f"| avg_bertscore_f1 | {report['summary'].get('avg_bertscore_f1', 0.0):.4f} |")
    lines.append(f"| avg_entailment_score | {report['summary'].get('avg_entailment_score', 0.0):.4f} |")
    lines.append(f"| avg_factual_consistency | {report['summary'].get('avg_factual_consistency', 0.0):.4f} |")
    lines.append(f"| avg_numeric_unit_consistency | {report['summary'].get('avg_numeric_unit_consistency', 0.0):.4f} |")
    lines.append(f"| avg_uncertainty_score | {report['summary'].get('avg_uncertainty_score', 0.0):.4f} |")
    lines.append(f"| mc_exact_rate | {report['summary'].get('mc_exact_rate', 0.0):.4f} |")
    lines.append(
        f"| mc_semantic_or_better_rate | {report['summary'].get('mc_semantic_or_better_rate', 0.0):.4f} |"
    )
    lines.append("")
    lines.append("## Per-question")
    lines.append("")

    for row in report["results"]:
        lines.append(f"### {row['id']} ({row['category']})")
        lines.append("")
        lines.append(f"- Query: {row['query']}")
        lines.append(f"- Expected: {row['expected_answer']}")
        lines.append(f"- Generated: {row['generated_answer']}")
        lines.append(
            f"- Metrics: f1={row['metrics']['token_f1']:.3f}, contains={row['metrics']['contains_expected']:.1f}, faith={row['metrics']['faithfulness']:.3f}, sem={row['metrics'].get('semantic_similarity', 0.0):.3f}, entail={row['metrics'].get('entailment_score', 0.0):.3f}, unc={row['metrics'].get('uncertainty_score', 0.0):.3f}"
        )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 Task 4 RAG generation evaluation")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "phase4_task4_rag_eval.yaml"),
        help="Path to YAML/JSON config",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional max questions override")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    report = evaluate(cfg, limit=args.limit)
    out_dir = Path(cfg["evaluation"]["output_dir"])
    save_report(report, out_dir)


if __name__ == "__main__":
    main()
