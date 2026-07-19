#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    seed_csv = root / "results" / "rag_generation_eval" / "seed_runs" / "seed_significance_summary.csv"
    pub_tbl = root / "results" / "publication" / "tables"
    pub_tbl.mkdir(parents=True, exist_ok=True)

    rows = []
    with seed_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    key_metrics = {"mc_exact_rate", "avg_calibrated_uncertainty", "avg_entailment_score", "avg_iterations"}
    out_table = pub_tbl / "phase5_seed_summary_table.csv"
    with out_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "n", "metric", "mean", "std", "ci95_low", "ci95_high"])
        writer.writeheader()
        for row in rows:
            if row.get("metric") in key_metrics:
                writer.writerow(row)

    print(f"Saved {out_table}")


if __name__ == "__main__":
    main()
