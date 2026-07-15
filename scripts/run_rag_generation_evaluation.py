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

    def generate(self, query: str, context: str) -> str:
        prompt = (
            "You are a physics research assistant. Use ONLY the provided context to answer concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        text, _metrics = self.engine.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return text.strip()

    def score_option(self, query: str, context: str, option_text: str) -> float:
        prompt = (
            "You are a physics research assistant. Use ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        continuation = " " + option_text.strip()
        return self.engine.score_continuation(prompt=prompt, continuation=continuation, tokenizer=self.tokenizer)


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
            verbose=True,
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

    eval_mode = str(eval_cfg.get("mode", "generation")).lower()

    for i, q in enumerate(questions, start=1):
        query = q["question"]
        expected = q.get("expected_answer", "")

        retrieved = retriever.retrieve(query, k=k_retrieve)
        final_docs = retrieved
        if reranker is not None:
            final_docs = reranker.rerank(query, retrieved, top_n=rerank_top_n)

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

        f1_scores.append(f1)
        contain_scores.append(contains_expected)
        faithful_scores.append(faith)

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
                "mc_exact": mc_exact,
                "mc_semantic_or_better": mc_semantic,
                "expected_rank": rank_expected,
            },
            "option_scores": [
                {"option": opt, "avg_logprob": score} for opt, score in (scored_sorted or [])
            ],
            "retrieved_topk": [
                {
                    "doc_id": d.get("doc_id"),
                    "rank": d.get("rank"),
                    "score": d.get("score"),
                    "rerank_score": d.get("rerank_score"),
                    "source": d.get("source"),
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
            "mc_exact_rate": avg(mc_exact_scores),
            "mc_semantic_or_better_rate": avg(mc_semantic_scores),
        },
        "results": per_q,
    }
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
            f"- Metrics: f1={row['metrics']['token_f1']:.3f}, contains={row['metrics']['contains_expected']:.1f}, faith={row['metrics']['faithfulness']:.3f}"
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
