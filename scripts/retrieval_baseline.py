#!/usr/bin/env python3
"""
Phase 4 Task 2: Retrieval Baseline (dependency-light).

Builds a BM25-style retrieval index from a JSONL corpus and supports query-time
Top-K retrieval for RAG scaffolding.

Usage:
  python scripts/retrieval_baseline.py build \
    --input /Users/jdsingh/slm_v0/data/offline_physics/raw_papers.jsonl \
    --index /Users/jdsingh/slm_v0/subword_tokenizer/data/retrieval/bm25_index.json \
    --chunk-size 220 --chunk-overlap 40

  python scripts/retrieval_baseline.py query \
    --index /Users/jdsingh/slm_v0/subword_tokenizer/data/retrieval/bm25_index.json \
    --q "What is the Higgs mechanism?" --k 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def extract_text(record: Dict) -> str:
    # Robust extraction from multiple possible schemas.
    candidates = [
        record.get("text"),
        record.get("content"),
        record.get("abstract"),
        record.get("summary"),
        record.get("title"),
    ]
    parts = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    return "\n\n".join(parts)


def chunk_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    if chunk_size <= 0:
        return []
    stride = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), stride):
        chunk = words[start : start + chunk_size]
        if len(chunk) < 20:
            continue
        chunks.append(chunk)
    return chunks


def build_index(input_path: Path, out_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    docs: List[Dict] = []
    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    doc_lens: List[int] = []
    doc_freq: Dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = extract_text(rec)
            if not text:
                continue

            words = tokenize(text)
            for chunk in chunk_words(words, chunk_size, chunk_overlap):
                tf = Counter(chunk)
                doc_id = len(docs)
                title = rec.get("title", "") if isinstance(rec.get("title"), str) else ""
                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": " ".join(chunk),
                        "source_line": line_num,
                    }
                )
                doc_lens.append(len(chunk))
                for term, freq in tf.items():
                    postings[term].append((doc_id, freq))
                for term in tf.keys():
                    doc_freq[term] += 1

    if not docs:
        raise RuntimeError("No retrievable chunks were built from the input corpus.")

    avgdl = sum(doc_lens) / len(doc_lens)

    # Store compact JSON index.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "num_docs": len(docs),
            "avgdl": avgdl,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "source": str(input_path),
        },
        "docs": docs,
        "doc_lens": doc_lens,
        "doc_freq": dict(doc_freq),
        "postings": {k: v for k, v in postings.items()},
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[OK] Index built: {out_path}")
    print(f"[OK] Chunks: {len(docs)} | Vocabulary: {len(doc_freq)} | AvgLen: {avgdl:.2f}")


def bm25_score(
    query_terms: List[str],
    index: Dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    N = index["meta"]["num_docs"]
    avgdl = index["meta"]["avgdl"]
    doc_lens = index["doc_lens"]
    doc_freq = index["doc_freq"]
    postings = index["postings"]

    qtf = Counter(query_terms)
    for term, q_count in qtf.items():
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings.get(term, []):
            dl = doc_lens[doc_id]
            denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            term_score = idf * (tf * (k1 + 1)) / max(denom, 1e-9)
            scores[doc_id] += q_count * term_score
    return scores


def query_index(index_path: Path, query: str, k: int) -> None:
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    q_terms = tokenize(query)
    scores = bm25_score(q_terms, index)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    if not ranked:
        print("No matching results found.")
        return

    docs = index["docs"]
    for i, (doc_id, score) in enumerate(ranked, start=1):
        d = docs[doc_id]
        snippet = d["text"][:260].replace("\n", " ")
        print(f"\n[{i}] score={score:.4f} doc_id={doc_id} source_line={d['source_line']}")
        if d.get("title"):
            print(f"Title: {d['title']}")
        print(f"Snippet: {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval baseline for Phase 4.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build index from JSONL corpus")
    build.add_argument("--input", required=True, help="Path to input JSONL corpus")
    build.add_argument("--index", required=True, help="Path to output index JSON")
    build.add_argument("--chunk-size", type=int, default=220)
    build.add_argument("--chunk-overlap", type=int, default=40)

    query = sub.add_parser("query", help="Run retrieval query")
    query.add_argument("--index", required=True, help="Path to index JSON")
    query.add_argument("--q", required=True, help="Query text")
    query.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "build":
        build_index(
            input_path=Path(args.input),
            out_path=Path(args.index),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        query_index(index_path=Path(args.index), query=args.q, k=args.k)


if __name__ == "__main__":
    main()
