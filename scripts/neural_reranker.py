#!/usr/bin/env python3
"""
Phase 4 Task 4: Neural reranking for RAG.

Uses a cross-encoder to rerank retrieved passages for better query-passage relevance.
Includes a fallback lexical reranker when model loading fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import re


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    strategy: str = "hybrid"
    cross_weight: float = 0.55
    semantic_weight: float = 0.30
    lexical_weight: float = 0.15


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        strategy: str = "hybrid",
        cross_weight: float = 0.55,
        semantic_weight: float = 0.30,
        lexical_weight: float = 0.15,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.strategy = strategy.lower()
        self.cross_weight = float(cross_weight)
        self.semantic_weight = float(semantic_weight)
        self.lexical_weight = float(lexical_weight)
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

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5, strategy: Optional[str] = None) -> List[Dict]:
        if not candidates:
            return []

        top_n = max(1, min(top_n, len(candidates)))
        selected_strategy = (strategy or self.strategy).lower()

        lexical_scores = self._lexical_scores(query, candidates)
        semantic_scores = self._semantic_scores(query, candidates)
        cross_scores = self._cross_scores(query, candidates)

        if selected_strategy == "lexical":
            final_scores = lexical_scores
        elif selected_strategy == "semantic":
            final_scores = semantic_scores
        elif selected_strategy == "cross_encoder":
            final_scores = cross_scores if cross_scores is not None else lexical_scores
        elif selected_strategy == "cascade":
            return self._cascade_rerank(query, candidates, top_n, lexical_scores, semantic_scores)
        else:
            final_scores = self._hybrid_scores(lexical_scores, semantic_scores, cross_scores)

        rescored: List[Dict] = []
        for idx, c in enumerate(candidates):
            row = dict(c)
            row["rerank_score"] = float(final_scores[idx])
            row["rerank_components"] = {
                "lexical": float(lexical_scores[idx]),
                "semantic": float(semantic_scores[idx]),
                "cross_encoder": float(cross_scores[idx]) if cross_scores is not None else None,
                "strategy": selected_strategy,
            }
            rescored.append(row)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, row in enumerate(rescored[:top_n], start=1):
            row["rank"] = i
        return rescored[:top_n]

    def _cross_scores(self, query: str, candidates: List[Dict]) -> Optional[List[float]]:
        if self.fallback or self.model is None:
            return None

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def _lexical_scores(self, query: str, candidates: List[Dict]) -> List[float]:
        q_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        scores: List[float] = []
        for c in candidates:
            text = c.get("text", "").lower()
            t_terms = set(re.findall(r"[a-zA-Z0-9_]+", text))
            overlap = len(q_terms & t_terms)
            scores.append(float(overlap))
        return scores

    def _semantic_scores(self, query: str, candidates: List[Dict]) -> List[float]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            docs = [query] + [c.get("text", "") for c in candidates]
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
            matrix = tfidf.fit_transform(docs)
            sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            return [float(s) for s in sims]
        except Exception:
            # robust fallback when sklearn isn't available
            return self._lexical_scores(query, candidates)

    def _hybrid_scores(
        self,
        lexical_scores: List[float],
        semantic_scores: List[float],
        cross_scores: Optional[List[float]],
    ) -> List[float]:
        lex_n = self._normalize(lexical_scores)
        sem_n = self._normalize(semantic_scores)

        if cross_scores is not None:
            cross_n = self._normalize(cross_scores)
            weights = [self.cross_weight, self.semantic_weight, self.lexical_weight]
            total_w = sum(max(w, 0.0) for w in weights) or 1.0
            cw, sw, lw = [max(w, 0.0) / total_w for w in weights]
            return [
                (cw * c) + (sw * s) + (lw * l)
                for c, s, l in zip(cross_n, sem_n, lex_n)
            ]

        # no cross-encoder: redistribute onto semantic + lexical
        total_w = max(self.semantic_weight, 0.0) + max(self.lexical_weight + self.cross_weight, 0.0)
        if total_w == 0:
            return sem_n
        sw = max(self.semantic_weight, 0.0) / total_w
        lw = max(self.lexical_weight + self.cross_weight, 0.0) / total_w
        return [(sw * s) + (lw * l) for s, l in zip(sem_n, lex_n)]

    def _cascade_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int,
        lexical_scores: List[float],
        semantic_scores: List[float],
    ) -> List[Dict]:
        pre_scores = self._hybrid_scores(lexical_scores, semantic_scores, cross_scores=None)
        pre_rank = sorted(range(len(candidates)), key=lambda i: pre_scores[i], reverse=True)
        stage_top = max(top_n * 2, 8)
        stage_idx = pre_rank[: min(stage_top, len(candidates))]

        reduced = [candidates[i] for i in stage_idx]
        cross_scores = self._cross_scores(query, reduced)
        if cross_scores is None:
            cross_scores = [pre_scores[i] for i in stage_idx]

        rescored: List[Dict] = []
        for j, i in enumerate(stage_idx):
            row = dict(candidates[i])
            row["rerank_score"] = float(cross_scores[j])
            row["rerank_components"] = {
                "lexical": float(lexical_scores[i]),
                "semantic": float(semantic_scores[i]),
                "cross_encoder": float(cross_scores[j]),
                "strategy": "cascade",
            }
            rescored.append(row)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, row in enumerate(rescored[:top_n], start=1):
            row["rank"] = i
        return rescored[:top_n]

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max <= v_min:
            return [0.0 for _ in values]
        span = v_max - v_min
        return [(v - v_min) / span for v in values]
