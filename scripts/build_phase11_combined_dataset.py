#!/usr/bin/env python3
"""Build combined 100Q benchmark for next phase (STEM 60 + Adversarial 40)."""

from __future__ import annotations

import json
from pathlib import Path


def _load_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data)


def _normalize(q: dict, default_category: str) -> dict:
    return {
        "id": q.get("id", "unknown_id"),
        "category": q.get("category", default_category),
        "question": q.get("question", q.get("query", "")),
        "expected_answer": q.get("expected_answer", ""),
        "distractors": q.get("distractors", []),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stem_path = root / "data/phase4_task6_expanded_stem_qa_dataset.json"
    adv_path = root / "data/phase4_task7_adversarial_qa_dataset.json"
    out_path = root / "data/phase5_combined_100qa.json"

    stem_qs = [_normalize(q, "stem") for q in _load_questions(stem_path)]
    adv_qs = [_normalize(q, "adversarial") for q in _load_questions(adv_path)]

    combined = []
    seen = set()
    for q in stem_qs + adv_qs:
        if q["id"] in seen:
            q = dict(q)
            q["id"] = f"{q['id']}_dup"
        seen.add(q["id"])
        combined.append(q)

    payload = {
        "name": "phase5_combined_100qa",
        "description": "Combined benchmark: 60 STEM + 40 adversarial",
        "counts": {
            "total": len(combined),
            "stem": len(stem_qs),
            "adversarial": len(adv_qs),
        },
        "questions": combined,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(payload["counts"])


if __name__ == "__main__":
    main()
