#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

STYLES = [
    "{q} Provide the most precise canonical term only.",
    "In one concise phrase, answer: {q}",
    "Hard mode: {q} Also reject the most likely misconception.",
    "Exam variant: {q} Return only the key concept.",
    "Research-check: {q} Answer with standard textbook wording.",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "data" / "phase5_combined_100qa.json"
    out_path = root / "data" / "phase5_combined_hard_500qa.json"

    base = json.loads(base_path.read_text(encoding="utf-8"))
    questions = base["questions"]

    hard_questions = []
    for q in questions:
        for idx, template in enumerate(STYLES, start=1):
            row = dict(q)
            row["id"] = f"{q['id']}_hard{idx}"
            row["question"] = template.format(q=q["question"])
            distractors = list(q.get("distractors", []))
            distractors.extend([
                f"none of the above ({q.get('category', 'general')})",
                f"approximate inverse of {q['expected_answer']}",
            ])
            row["distractors"] = distractors[:4]
            row["difficulty"] = "hard"
            row["source_id"] = q["id"]
            hard_questions.append(row)

    hard_questions = hard_questions[:500]

    payload = {
        "name": "phase5_combined_hard_500qa",
        "description": "Hard benchmark extension generated from phase5_combined_100qa with five hard variants per source item",
        "counts": {
            "total": len(hard_questions),
            "base_questions": len(questions),
            "variants_per_question": 5,
        },
        "questions": hard_questions,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_path} ({len(hard_questions)} questions)")


if __name__ == "__main__":
    main()
