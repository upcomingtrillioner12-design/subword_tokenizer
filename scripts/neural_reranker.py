#!/usr/bin/env python3
"""
Phase 4 Task 4: Neural reranking for RAG.

Uses a cross-encoder to rerank retrieved passages for better query-passage relevance.
Includes a fallback lexical reranker when model loading fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.fallback = False

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            if self.verbose:
                print(f"[reranker] Loaded cross-encoder: {model_name}")
        except Exception as exc:
            self.fallback = True
            if self.verbose:
                print(f"[reranker] Failed to load {model_name}, using fallback lexical reranker: {exc}")

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
        if not candidates:
            return []

        top_n = max(1, min(top_n, len(candidates)))

        if self.fallback or self.model is None:
            return self._fallback_rerank(query, candidates, top_n)

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)

        rescored: List[Dict] = []
        for c, s in zip(candidates, scores):
            row = dict(c)
            row["rerank_score"] = float(s)
            rescored.append(row)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, row in enumerate(rescored[:top_n], start=1):
            row["rank"] = i
        return rescored[:top_n]

    def _fallback_rerank(self, query: str, candidates: List[Dict], top_n: int) -> List[Dict]:
        import re

        q_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        rescored: List[Dict] = []
        for c in candidates:
            text = c.get("text", "").lower()
            t_terms = set(re.findall(r"[a-zA-Z0-9_]+", text))
            overlap = len(q_terms & t_terms)
            row = dict(c)
            row["rerank_score"] = float(overlap)
            rescored.append(row)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, row in enumerate(rescored[:top_n], start=1):
            row["rank"] = i
        return rescored[:top_n]
