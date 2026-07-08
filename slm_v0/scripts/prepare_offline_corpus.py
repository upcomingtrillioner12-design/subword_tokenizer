#!/usr/bin/env python3
"""
Prepare an offline physics corpus for Phase 2 LoRA fine-tuning.

Outputs:
- raw_papers.jsonl
- train.bin / val.bin / test.bin (uint16 token ids)
- corpus_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare offline tokenized corpus for Phase 2.")
    p.add_argument("--category", default="physics")
    p.add_argument("--max-papers", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--min-tokens", type=int, default=12)
    p.add_argument("--delay-seconds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--tokenizer-model", default=str(ROOT / "subword_tokenizer" / "model_32k.json"))
    p.add_argument("--output-dir", default=str(ROOT / "data" / "offline_physics"))
    p.add_argument("--max-sequences", type=int, default=0, help="Optional hard cap for generated sequences per split.")
    return p.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")


def _collect_docs(
    category: str,
    max_papers: int,
    delay_seconds: int,
    min_tokens: int,
) -> List[List[int]]:
    docs: List[List[int]] = []
    stream = stream_train.stream_arxiv(
        category=category,
        max_results=max_papers,
        delay_seconds=delay_seconds,
        min_tokens=min_tokens,
        max_retries=100,
        retry_backoff_seconds=5,
    )

    for tokens in stream:
        if tokens:
            docs.append(tokens)

    return docs


def _write_raw_jsonl(output_path: Path, docs: List[List[int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, tks in enumerate(docs):
            rec = {
                "doc_id": i,
                "token_count": len(tks),
                "tokens": tks,
            }
            f.write(json.dumps(rec) + "\n")


def _pack_sequences(docs: List[List[int]], seq_len: int) -> np.ndarray:
    packed: List[int] = []
    buffer: List[int] = []

    for doc in docs:
        buffer.extend(doc)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            packed.extend(chunk[:-1])
            buffer = buffer[seq_len:]

    return np.asarray(packed, dtype=np.uint16)


def _split_docs(
    docs: List[List[int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_docs = shuffled[:train_end]
    val_docs = shuffled[train_end:val_end]
    test_docs = shuffled[val_end:]
    return train_docs, val_docs, test_docs


def _truncate_sequences(arr: np.ndarray, max_sequences: int, seq_len: int) -> np.ndarray:
    if max_sequences <= 0:
        return arr
    max_tokens = max_sequences * seq_len
    return arr[:max_tokens]


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_train._activate_tokenizer_model(Path(args.tokenizer_model))

    print(f"[phase2] Collecting docs from category={args.category} max_papers={args.max_papers}...")
    docs = _collect_docs(
        category=args.category,
        max_papers=args.max_papers,
        delay_seconds=args.delay_seconds,
        min_tokens=args.min_tokens,
    )
    if not docs:
        raise RuntimeError("No documents collected from arXiv stream.")

    raw_path = output_dir / "raw_papers.jsonl"
    _write_raw_jsonl(raw_path, docs)

    train_docs, val_docs, test_docs = _split_docs(
        docs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_arr = _pack_sequences(train_docs, seq_len=args.seq_len)
    val_arr = _pack_sequences(val_docs, seq_len=args.seq_len)
    test_arr = _pack_sequences(test_docs, seq_len=args.seq_len)

    train_arr = _truncate_sequences(train_arr, args.max_sequences, args.seq_len)
    val_arr = _truncate_sequences(val_arr, max(1, args.max_sequences // 8) if args.max_sequences > 0 else 0, args.seq_len)
    test_arr = _truncate_sequences(test_arr, max(1, args.max_sequences // 8) if args.max_sequences > 0 else 0, args.seq_len)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    test_path = output_dir / "test.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    test_arr.tofile(test_path)

    stats: Dict[str, object] = {
        "category": args.category,
        "max_papers": args.max_papers,
        "documents_collected": len(docs),
        "seq_len": args.seq_len,
        "tokenizer_model": args.tokenizer_model,
        "splits": {
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "train_tokens": int(train_arr.size),
            "val_tokens": int(val_arr.size),
            "test_tokens": int(test_arr.size),
        },
        "files": {
            "raw": str(raw_path),
            "train_bin": str(train_path),
            "val_bin": str(val_path),
            "test_bin": str(test_path),
        },
    }

    stats_path = output_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[phase2] Offline corpus preparation complete.")
    print(f"[phase2] Stats: {stats_path}")


if __name__ == "__main__":
    main()
