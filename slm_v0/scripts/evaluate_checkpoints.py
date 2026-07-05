#!/usr/bin/env python3
"""
Evaluate multiple checkpoints and select the best one by validation loss.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train


def _infer_prefix(ckpt_path: Path) -> str:
    m = re.match(r"(.+)_epoch\d+\.pt$", ckpt_path.name)
    if not m:
        raise ValueError(f"Checkpoint name should look like '<prefix>_epochN.pt': {ckpt_path.name}")
    return m.group(1)


def _load_summary_for_checkpoint(ckpt_path: Path) -> Dict:
    prefix = _infer_prefix(ckpt_path)
    summary_path = ckpt_path.parent / f"{prefix}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for checkpoint: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_eval_batches(
    tokenizer_model: Path,
    category: str,
    max_papers: int,
    batch_size: int,
    seq_len: int,
    eval_steps: int,
    min_tokens: int,
    delay_seconds: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    stream_train._activate_tokenizer_model(tokenizer_model)

    token_stream = stream_train.stream_arxiv(
        category=category,
        max_results=max_papers,
        delay_seconds=delay_seconds,
        min_tokens=min_tokens,
    )

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    buffer: List[int] = []

    for tokens in token_stream:
        buffer.extend(tokens)
        while len(buffer) >= batch_size * seq_len + 1:
            batch = buffer[: batch_size * seq_len + 1]
            buffer = buffer[batch_size * seq_len :]
            x = torch.tensor(batch[:-1], dtype=torch.long).view(batch_size, seq_len)
            y = torch.tensor(batch[1:], dtype=torch.long).view(batch_size, seq_len)
            batches.append((x, y))
            if len(batches) >= eval_steps:
                return batches

    return batches


def _evaluate_checkpoint(
    checkpoint_path: Path,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    d_model: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
) -> float:
    device = stream_train.DEVICE
    model = stream_train.TinyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    losses: List[float] = []
    with torch.no_grad():
        for x_cpu, y_cpu in batches:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(float(loss.item()))

    if not losses:
        raise RuntimeError("No evaluation batches available.")
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and pick the best one automatically.")
    parser.add_argument(
        "--checkpoints-glob",
        default="prototype_laptop_epoch*.pt",
        help="Glob pattern inside checkpoints/ to select checkpoint files.",
    )
    parser.add_argument("--category", default="physics")
    parser.add_argument("--max-papers", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=12)
    parser.add_argument("--delay-seconds", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(CHECKPOINTS_DIR / "best_checkpoint.json"),
        help="Output JSON file with ranking and best checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_paths = sorted(CHECKPOINTS_DIR.glob(args.checkpoints_glob))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {args.checkpoints_glob}")

    summaries = {ckpt: _load_summary_for_checkpoint(ckpt) for ckpt in checkpoint_paths}

    # Use tokenizer/vocab from first checkpoint summary for a fair comparison.
    first_summary = summaries[checkpoint_paths[0]]
    tokenizer_model = Path(first_summary["tokenizer_model"])
    vocab_size = stream_train._resolve_vocab_size(tokenizer_model)

    batches = _build_eval_batches(
        tokenizer_model=tokenizer_model,
        category=args.category,
        max_papers=args.max_papers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_steps=args.eval_steps,
        min_tokens=args.min_tokens,
        delay_seconds=args.delay_seconds,
    )
    if not batches:
        raise RuntimeError("Could not build evaluation batches. Increase --max-papers or lower --seq-len.")

    results = []
    for ckpt in checkpoint_paths:
        summary = summaries[ckpt]
        loss = _evaluate_checkpoint(
            checkpoint_path=ckpt,
            batches=batches,
            d_model=int(summary.get("d_model", 128)),
            n_layers=int(summary.get("n_layers", 2)),
            n_heads=int(summary.get("n_heads", 4)),
            vocab_size=vocab_size,
        )
        results.append(
            {
                "checkpoint": str(ckpt),
                "avg_eval_loss": loss,
                "d_model": summary.get("d_model"),
                "n_layers": summary.get("n_layers"),
                "n_heads": summary.get("n_heads"),
            }
        )
        print(f"{ckpt.name}: avg_eval_loss={loss:.4f}")

    ranked = sorted(results, key=lambda r: r["avg_eval_loss"])
    best = ranked[0]

    output = {
        "device": stream_train.DEVICE,
        "category": args.category,
        "eval_steps": len(batches),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "min_tokens": args.min_tokens,
        "tokenizer_model": str(tokenizer_model),
        "best_checkpoint": best,
        "ranking": ranked,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nBest checkpoint selected automatically:")
    print(f"  {Path(best['checkpoint']).name} (avg_eval_loss={best['avg_eval_loss']:.4f})")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
