#!/usr/bin/env python3
"""Phase 3 Task 7: Physics domain knowledge QA evaluation.

Implements a robust multiple-choice QA scoring by likelihood:
- For each question, score expected answer and distractors by average token log-prob
- Pick best option per model
- Score rubric:
    exact match: 1.0 (expected answer ranked #1)
    semantic match: 0.5 (expected answer ranked #2)
    no match: 0.0 (ranked #3/#4)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "slm_v0") not in sys.path:
    sys.path.insert(0, str(ROOT / "slm_v0"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

phase2_path = ROOT / "scripts" / "phase2_lora_finetune.py"
spec = importlib.util.spec_from_file_location("phase2_lora_finetune_local", phase2_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to import {phase2_path}")
phase2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase2
spec.loader.exec_module(phase2)

build_model = phase2.build_model
inject_lora = phase2.inject_lora
load_config = phase2.load_config
resolve_device = phase2.resolve_device

import stream_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics QA evaluation (Task 7)")
    p.add_argument("--config", default=str(ROOT / "config" / "phase2_lora_config.yaml"))
    p.add_argument("--adapter", default=str(ROOT / "checkpoints" / "phase2_lora" / "best_lora_adapter.pt"))
    p.add_argument("--qa-dataset", default=str(ROOT / "data" / "physics_qa_dataset.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "results" / "physics_qa_results.json"))
    return p.parse_args()


def resolve_artifact(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        if p.exists():
            return p
        try:
            rel = p.relative_to(ROOT)
            alt = ROOT.parent / rel
            if alt.exists():
                return alt
        except ValueError:
            pass
    local = ROOT / p
    if local.exists():
        return local
    parent = ROOT.parent / p
    if parent.exists():
        return parent
    return local


def normalize_cfg_paths(cfg: Dict) -> Dict:
    model_cfg = cfg.get("model", {})
    if "base_summary" in model_cfg:
        model_cfg["base_summary"] = str(resolve_artifact(model_cfg["base_summary"]))
    if "base_checkpoint" in model_cfg:
        model_cfg["base_checkpoint"] = str(resolve_artifact(model_cfg["base_checkpoint"]))
    if "tokenizer_model" in model_cfg:
        model_cfg["tokenizer_model"] = str(resolve_artifact(model_cfg["tokenizer_model"]))
    cfg["model"] = model_cfg
    return cfg


def load_adapter_into_model(model: torch.nn.Module, adapter_path: Path, lora_cfg: Dict) -> None:
    inject_lora(model, lora_cfg)
    payload = torch.load(adapter_path, map_location="cpu")
    lora_state = payload.get("lora_state", payload)
    _, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in adapter checkpoint: {unexpected}")


def avg_logprob_continuation(model: torch.nn.Module, prefix_ids: List[int], cont_ids: List[int], device: str) -> float:
    """Average log-prob of continuation tokens given prefix (teacher-forced)."""
    if not cont_ids:
        return float("-inf")

    seq = prefix_ids + cont_ids
    x = torch.tensor([seq], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)  # [1, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

    start = len(prefix_ids) - 1
    total_lp = 0.0
    count = 0
    for i, tok in enumerate(cont_ids):
        lp = float(log_probs[0, start + i, tok].item())
        total_lp += lp
        count += 1
    return total_lp / max(count, 1)


def score_question(model: torch.nn.Module, question: str, expected: str, distractors: List[str], device: str) -> Tuple[float, Dict]:
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = stream_train._tokenize_with_our_model(prompt)

    options = [expected] + distractors
    option_scores = []
    for opt in options:
        opt_ids = stream_train._tokenize_with_our_model(" " + opt)
        s = avg_logprob_continuation(model, prompt_ids, opt_ids, device)
        option_scores.append((opt, s))

    ranked = sorted(option_scores, key=lambda x: x[1], reverse=True)
    best = ranked[0][0]

    # rubric-compatible score
    rank_expected = [i for i, (opt, _) in enumerate(ranked) if opt == expected][0] + 1
    if rank_expected == 1:
        score = 1.0
    elif rank_expected == 2:
        score = 0.5
    else:
        score = 0.0

    return score, {
        "best_answer": best,
        "expected_rank": rank_expected,
        "ranked_options": [{"answer": a, "avg_logprob": s} for a, s in ranked],
    }


def aggregate(results: List[Dict]) -> Dict:
    if not results:
        return {}
    total = len(results)
    phase1_scores = [r["phase1_score"] for r in results]
    phase2_scores = [r["phase2_score"] for r in results]

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    by_category = {}
    for c, rows in by_cat.items():
        p1 = [x["phase1_score"] for x in rows]
        p2 = [x["phase2_score"] for x in rows]
        by_category[c] = {
            "count": len(rows),
            "phase1_avg_score": float(sum(p1) / len(p1)),
            "phase2_avg_score": float(sum(p2) / len(p2)),
            "phase1_exact_rate": float(sum(1 for s in p1 if s == 1.0) / len(p1)),
            "phase2_exact_rate": float(sum(1 for s in p2 if s == 1.0) / len(p2)),
        }

    return {
        "num_questions": total,
        "phase1_avg_score": float(sum(phase1_scores) / total),
        "phase2_avg_score": float(sum(phase2_scores) / total),
        "phase1_exact_rate": float(sum(1 for s in phase1_scores if s == 1.0) / total),
        "phase2_exact_rate": float(sum(1 for s in phase2_scores if s == 1.0) / total),
        "phase1_semantic_or_better": float(sum(1 for s in phase1_scores if s >= 0.5) / total),
        "phase2_semantic_or_better": float(sum(1 for s in phase2_scores if s >= 0.5) / total),
        "delta_avg_score": float((sum(phase2_scores) - sum(phase1_scores)) / total),
        "by_category": by_category,
    }


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = normalize_cfg_paths(load_config(Path(args.config)))
    device = resolve_device(cfg["training"].get("device", "auto"))

    tokenizer_model = resolve_artifact(cfg["model"]["tokenizer_model"])
    stream_train._activate_tokenizer_model(tokenizer_model)

    phase1 = build_model(cfg).to(device)

    phase2 = build_model(cfg)
    adapter_path = resolve_artifact(args.adapter)
    load_adapter_into_model(phase2, adapter_path, cfg["lora"])
    phase2 = phase2.to(device)

    qa_path = resolve_artifact(args.qa_dataset)
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = qa_data.get("questions", qa_data)

    started = time.perf_counter()
    rows = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        category = q["category"]
        question = q["question"]
        expected = q["expected_answer"]
        distractors = q["distractors"]

        p1_score, p1_meta = score_question(phase1, question, expected, distractors, device)
        p2_score, p2_meta = score_question(phase2, question, expected, distractors, device)

        rows.append(
            {
                "id": qid,
                "category": category,
                "question": question,
                "expected_answer": expected,
                "phase1_score": p1_score,
                "phase2_score": p2_score,
                "phase1": p1_meta,
                "phase2": p2_meta,
            }
        )

        print(f"[{i}/{len(questions)}] {qid}: P1={p1_score:.1f}, P2={p2_score:.1f}")

    summary = aggregate(rows)
    elapsed = time.perf_counter() - started

    report = {
        "metadata": {
            "task": "phase3_task7_physics_qa_eval",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "config": str(args.config),
            "adapter": str(adapter_path),
            "qa_dataset": str(qa_path),
            "seed": int(args.seed),
            "elapsed_seconds": elapsed,
            "scoring": {
                "exact_match": 1.0,
                "semantic_match": 0.5,
                "no_match": 0.0,
                "definition": "rank of expected answer among 1 correct + 3 distractors by continuation likelihood",
            },
        },
        "summary": summary,
        "results": rows,
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== PHYSICS QA RESULTS (TASK 7) ===")
    print(f"Questions:          {summary['num_questions']}")
    print(f"Phase 1 avg score:  {summary['phase1_avg_score']:.3f}")
    print(f"Phase 2 avg score:  {summary['phase2_avg_score']:.3f}")
    print(f"Phase 1 exact:      {summary['phase1_exact_rate']*100:.1f}%")
    print(f"Phase 2 exact:      {summary['phase2_exact_rate']*100:.1f}%")
    print(f"Delta avg score:    {summary['delta_avg_score']:.3f}")
    print(f"Saved:              {out}")


if __name__ == "__main__":
    main()
