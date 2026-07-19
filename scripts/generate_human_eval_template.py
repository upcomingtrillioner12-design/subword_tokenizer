#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def main() -> None:
    random.seed(42)
    root = Path(__file__).resolve().parents[1]
    inputs = [
        root / "results" / "rag_generation_eval" / "rag_generation_eval_20260716_114138.json",
        root / "results" / "rag_generation_eval" / "rag_generation_eval_20260716_125406.json",
    ]

    rows = []
    for p in inputs:
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            rows.append(
                {
                    "source_run": p.name,
                    "question_id": r.get("id", ""),
                    "category": r.get("category", ""),
                    "query": r.get("query", ""),
                    "expected_answer": r.get("expected_answer", ""),
                    "generated_answer": r.get("generated_answer", ""),
                    "context_excerpt": r.get("context", "")[:280].replace("\n", " "),
                }
            )

    uniq = []
    seen = set()
    for r in rows:
        key = (r["question_id"], r["generated_answer"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    sample = uniq if len(uniq) <= 100 else random.sample(uniq, 100)

    out_dir = root / "results" / "human_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "human_eval_template.csv"

    fields = [
        "sample_id",
        "source_run",
        "question_id",
        "category",
        "query",
        "expected_answer",
        "generated_answer",
        "context_excerpt",
        "annotator_a_faithfulness_1_5",
        "annotator_a_helpfulness_1_5",
        "annotator_a_correctness_1_5",
        "annotator_b_faithfulness_1_5",
        "annotator_b_helpfulness_1_5",
        "annotator_b_correctness_1_5",
        "adjudicated_final_score_1_5",
        "notes",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(sample, start=1):
            row = {k: "" for k in fields}
            row.update(r)
            row["sample_id"] = f"HE{i:03d}"
            writer.writerow(row)

    print(f"Saved {out} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
