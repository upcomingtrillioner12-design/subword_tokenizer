#!/usr/bin/env python3
"""Phase 3 Task 4: Qualitative evaluation runner.

Generates side-by-side outputs for Phase 1 (base) and Phase 2 (LoRA)
on a curated prompt subset, then writes JSON + markdown for manual scoring.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from inference_lora import LoRAInferenceEngine, load_tokenizer
from sampling_profiles import resolve_sampling_config


def load_prompt_subset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported prompt file format: {path}")


def run_best_of_n(
    engine: LoRAInferenceEngine,
    tokenizer,
    prompt: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Dict[str, Any]:
    trials = []
    for _ in range(n):
        text, metrics = engine.generate(
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        trials.append(
            {
                "text": text,
                "generated_tokens": metrics.generated_tokens,
                "elapsed_seconds": metrics.elapsed_seconds,
                "tokens_per_second": metrics.tokens_per_second,
            }
        )

    # pick the longest completion for review
    best = max(trials, key=lambda t: t["generated_tokens"])
    return {"best": best, "trials": trials}


def write_markdown_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Phase 3 Task 4 - Qualitative Evaluation")
    lines.append("")
    lines.append("Manual rubric (score each 1-5):")
    lines.append("- Coherence")
    lines.append("- Physics correctness")
    lines.append("- Relevance to prompt")
    lines.append("- Depth/clarity")
    lines.append("")

    for i, r in enumerate(rows, 1):
        lines.append(f"## Prompt {i}: {r['prompt_id']}")
        lines.append(f"- Category: {r['category']}")
        lines.append(f"- Difficulty: {r['difficulty']}")
        lines.append("")
        lines.append(f"**Prompt**: {r['prompt_text']}")
        lines.append("")

        lines.append("### Phase 1 (base)")
        lines.append(f"- Tokens: {r['phase1_best']['generated_tokens']}")
        lines.append(f"- Time: {r['phase1_best']['elapsed_seconds']:.4f}s")
        lines.append(f"- Output: {r['phase1_best']['text']}")
        lines.append("")

        lines.append("### Phase 2 (LoRA)")
        lines.append(f"- Tokens: {r['phase2_best']['generated_tokens']}")
        lines.append(f"- Time: {r['phase2_best']['elapsed_seconds']:.4f}s")
        lines.append(f"- Output: {r['phase2_best']['text']}")
        lines.append("")

        lines.append("### Manual assessment")
        lines.append("- Winner: [phase1 | phase2 | tie]")
        lines.append("- Notes:")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run qualitative side-by-side evaluation")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("data/qualitative_eval_subset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sampling-profile", choices=["production", "canonical"], default="production")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=3, help="best-of-N per model/prompt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    generation_config = resolve_sampling_config(
        profile=args.sampling_profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompt_subset(args.prompts)
    tokenizer = load_tokenizer(args.tokenizer_model)

    phase1 = LoRAInferenceEngine(
        base_checkpoint=args.base_checkpoint,
        lora_checkpoint=None,
        device=args.device,
        verbose=args.verbose,
    )
    phase2 = LoRAInferenceEngine(
        base_checkpoint=args.base_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        verbose=args.verbose,
    )

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, Any]] = []

    for idx, p in enumerate(prompts, 1):
        prompt_id = p.get("id", f"p{idx:03d}")
        category = p.get("category", "unknown")
        difficulty = p.get("difficulty", "unknown")
        text = p.get("text", "")

        print(f"[{idx}/{len(prompts)}] {prompt_id} ({difficulty})")

        p1 = run_best_of_n(
            engine=phase1,
            tokenizer=tokenizer,
            prompt=text,
            n=args.num_samples,
            max_tokens=generation_config["max_tokens"],
            temperature=generation_config["temperature"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
        )
        p2 = run_best_of_n(
            engine=phase2,
            tokenizer=tokenizer,
            prompt=text,
            n=args.num_samples,
            max_tokens=generation_config["max_tokens"],
            temperature=generation_config["temperature"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
        )

        rows.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "difficulty": difficulty,
                "prompt_text": text,
                "phase1_best": p1["best"],
                "phase1_trials": p1["trials"],
                "phase2_best": p2["best"],
                "phase2_trials": p2["trials"],
                "manual_winner": "pending",
                "manual_notes": "",
            }
        )

    out_json = args.output_dir / "phase3_qualitative_outputs.json"
    payload = {
        "metadata": {
            "task": "phase3_task4_qualitative_evaluation",
            "started": started,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "num_prompts": len(rows),
            "num_samples_per_prompt": args.num_samples,
            "sampling_profile": args.sampling_profile,
            "generation": {
                "max_tokens": generation_config["max_tokens"],
                "temperature": generation_config["temperature"],
                "top_k": generation_config["top_k"],
                "top_p": generation_config["top_p"],
            },
        },
        "results": rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_md = args.output_dir / "phase3_qualitative_assessment.md"
    write_markdown_report(out_md, rows)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
