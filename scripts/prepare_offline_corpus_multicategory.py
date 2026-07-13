#!/usr/bin/env python3
"""
Prepare an offline physics corpus for Phase 2 LoRA fine-tuning.
Uses multiple arXiv categories to avoid deep-offset 500s.

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
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train

# Split collection across multiple categories to avoid deep-offset API failures
PHYSICS_CATEGORIES = [
    "all:physics",
    "cat:physics.quant-ph",
    "cat:physics.optics",
    "cat:hep-th",
    "cat:gr-qc",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare offline tokenized corpus for Phase 2 (multi-category).")
    p.add_argument("--max-papers", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--min-tokens", type=int, default=12)
    p.add_argument("--delay-seconds", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--request-timeout", type=int, default=90)
    p.add_argument("--max-retries", type=int, default=60)
    p.add_argument("--retry-backoff-seconds", type=int, default=5)
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


def _collect_docs_multicategory(
    max_papers: int,
    delay_seconds: int,
    min_tokens: int,
    chunk_size: int,
    request_timeout: int,
    max_retries: int,
    retry_backoff_seconds: int,
    progress_path: Path,
) -> List[List[int]]:
    """Collect docs from multiple categories to avoid deep-offset API failures."""
    docs: List[List[int]] = []
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    seen_arxiv_ids: set = set()
    
    papers_per_category = max(100, max_papers // len(PHYSICS_CATEGORIES))
    
    for cat_idx, category in enumerate(PHYSICS_CATEGORIES):
        print(f"\n[phase2] Category {cat_idx+1}/{len(PHYSICS_CATEGORIES)}: {category}")
        print(f"[phase2] Target papers per category: {papers_per_category}")
        
        for start in range(0, papers_per_category, chunk_size):
            if len(docs) >= max_papers:
                print(f"[phase2] Reached target {max_papers} papers, stopping.")
                break
            
            want = min(chunk_size, papers_per_category - start, max_papers - len(docs))
            
            progress = {
                "category": category,
                "category_idx": cat_idx,
                "total_categories": len(PHYSICS_CATEGORIES),
                "target_max_papers": max_papers,
                "chunk_size": chunk_size,
                "state": "requesting_chunk",
                "last_start": start,
                "last_requested": want,
                "documents_collected": len(docs),
                "timestamp": int(time.time()),
            }
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            
            entries = None
            for attempt in range(max_retries + 1):
                try:
                    params = {
                        "search_query": category,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "start": start,
                        "max_results": want,
                    }
                    resp = requests.get("https://export.arxiv.org/api/query", params=params, timeout=request_timeout)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.text)
                    entries = root.findall("atom:entry", atom_ns)
                    break
                except Exception as e:
                    if attempt >= max_retries:
                        print(f"[phase2] Category {category} chunk start={start} failed after {max_retries} retries: {e}")
                        break
                    sleep_s = retry_backoff_seconds * min(attempt + 1, 12)
                    progress["state"] = "retrying_chunk"
                    progress["retry_attempt"] = attempt + 1
                    progress["retry_max"] = max_retries
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(f"[phase2] cat {category} chunk start={start} failed ({e}); retry {attempt+1}/{max_retries} in {sleep_s}s")
                    time.sleep(sleep_s)
            
            if entries is None:
                print(f"[phase2] Category {category} chunk start={start} exhausted retries, moving to next category.")
                break
            
            used_in_chunk = 0
            for entry in entries:
                arxiv_id = entry.findtext("atom:id", default="", namespaces=atom_ns).strip()
                if arxiv_id in seen_arxiv_ids:
                    continue
                seen_arxiv_ids.add(arxiv_id)
                
                title = entry.findtext("atom:title", default="", namespaces=atom_ns).strip()
                summary = entry.findtext("atom:summary", default="", namespaces=atom_ns).strip()
                text = f"Title: {title}\nAbstract: {summary}\n\n"
                tokens = stream_train._tokenize_with_our_model(text)
                if len(tokens) >= min_tokens:
                    docs.append(tokens)
                    used_in_chunk += 1
            
            progress["state"] = "chunk_complete"
            progress["last_entries_returned"] = len(entries)
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(
                f"[phase2] cat {category} chunk done start={start} returned={len(entries)} "
                f"usable={used_in_chunk} total_docs={len(docs)}"
            )
            
            if len(entries) < want:
                print(f"[phase2] Category {category} returned fewer entries than requested; moving to next category.")
                break
            
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        
        if len(docs) >= max_papers:
            break
    
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

    print(f"[phase2] Collecting {args.max_papers} docs from {len(PHYSICS_CATEGORIES)} physics categories...")
    progress_path = output_dir / "download_progress.json"
    docs = _collect_docs_multicategory(
        max_papers=args.max_papers,
        delay_seconds=args.delay_seconds,
        min_tokens=args.min_tokens,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        progress_path=progress_path,
    )
    if not docs:
        raise RuntimeError("No documents collected from arXiv.")
    
    print(f"[phase2] Collected {len(docs)} unique documents. Writing raw_papers.jsonl...")
    raw_path = output_dir / "raw_papers.jsonl"
    _write_raw_jsonl(raw_path, docs)

    print(f"[phase2] Splitting into train/val/test...")
    train_docs, val_docs, test_docs = _split_docs(
        docs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[phase2] Packing sequences...")
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
        "collection_method": "multi_category",
        "categories": PHYSICS_CATEGORIES,
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
