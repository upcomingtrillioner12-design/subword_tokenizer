#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from statistics import mean, stdev

METRICS = [
    "mc_exact_rate",
    "avg_calibrated_uncertainty",
    "avg_entailment_score",
    "avg_faithfulness",
    "avg_iterations",
    "iterative_trigger_rate",
]


def classify(report: dict) -> str:
    md = report.get("metadata", {})
    reranker = md.get("reranker", {})
    it = md.get("iterative_retrieval", {})
    has_ft = bool(reranker.get("finetuned_checkpoint"))
    iterative = bool(it.get("enabled", False))
    if iterative and not has_ft:
        return "phase5_full_integration_eval"
    if iterative and has_ft:
        return "phase5_finetuned_cross_encoder_eval"
    if not iterative and not has_ft:
        return "phase5_ablation_no_iter_original"
    return "phase5_ablation_no_iter_finetuned"


def ci95(vals: list[float]) -> tuple[float, float]:
    if len(vals) <= 1:
        return vals[0], vals[0]
    m = mean(vals)
    sd = stdev(vals)
    margin = 1.96 * sd / math.sqrt(len(vals))
    return m - margin, m + margin


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "results" / "rag_generation_eval"
    out_dir = src_dir / "seed_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for p in sorted(src_dir.glob("rag_generation_eval_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("metadata", {}).get("num_questions", 0) < 30:
            continue
        reports.append((p, data))

    groups: dict[str, list[tuple[Path, dict]]] = {
        "phase5_full_integration_eval": [],
        "phase5_finetuned_cross_encoder_eval": [],
        "phase5_ablation_no_iter_original": [],
        "phase5_ablation_no_iter_finetuned": [],
    }
    for p, d in reports:
        groups[classify(d)].append((p, d))

    # pick up to 3 runs per group (latest first)
    selected: dict[str, list[tuple[Path, dict]]] = {}
    for key, vals in groups.items():
        vals = sorted(vals, key=lambda x: x[1]["metadata"].get("timestamp", ""), reverse=True)
        selected[key] = vals[:3]

    # copy selected runs to seed_runs folder
    for key, vals in selected.items():
        for idx, (path, data) in enumerate(vals, start=1):
            target = out_dir / f"{key}_seed{idx}.json"
            shutil.copy2(path, target)

    # write aggregate CSV + markdown
    csv_lines = ["config,n,metric,mean,std,ci95_low,ci95_high"]
    md = ["# Multi-seed Significance Summary", "", "Generated from existing 102-question evaluation runs.", ""]

    for key, vals in selected.items():
        if not vals:
            continue
        md.append(f"## {key}")
        md.append("")
        md.append("| metric | mean | std | 95% CI | n |")
        md.append("|---|---:|---:|---:|---:|")
        for metric in METRICS:
            numbers = [float(v[1].get("summary", {}).get(metric, 0.0)) for v in vals]
            m = mean(numbers)
            s = stdev(numbers) if len(numbers) > 1 else 0.0
            lo, hi = ci95(numbers)
            csv_lines.append(f"{key},{len(numbers)},{metric},{m:.6f},{s:.6f},{lo:.6f},{hi:.6f}")
            md.append(f"| {metric} | {m:.4f} | {s:.4f} | [{lo:.4f}, {hi:.4f}] | {len(numbers)} |")
        md.append("")

    (out_dir / "seed_significance_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    (out_dir / "seed_significance_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {out_dir / 'seed_significance_summary.csv'}")
    print(f"Saved: {out_dir / 'seed_significance_summary.md'}")


if __name__ == "__main__":
    main()
