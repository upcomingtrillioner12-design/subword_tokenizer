#!/usr/bin/env python3
"""
Phase 4 Task 9: Semantic metrics and confidence signals for RAG evaluation.

Provides:
- semantic similarity (embedding cosine)
- BERTScore-like token alignment proxy
- entailment score (NLI, with lexical fallback)
- factual consistency score (answer entailed by context)
- numeric/unit consistency score
- uncertainty score for unanswerable handling
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\.\-/%]+")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall(_norm_text(s))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _token_f1(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0

    ca: Dict[str, int] = {}
    cb: Dict[str, int] = {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    for t in tb:
        cb[t] = cb.get(t, 0) + 1

    overlap = 0
    for t, n in ca.items():
        overlap += min(n, cb.get(t, 0))

    p = _safe_div(overlap, len(ta))
    r = _safe_div(overlap, len(tb))
    return _safe_div(2 * p * r, p + r) if (p + r) else 0.0


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(u, v) / denom)


@dataclass
class SemanticMetricsConfig:
    enabled: bool = False
    embedding_model: str = "all-mpnet-base-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    max_tokens_for_bertscore: int = 80


class SemanticMetricsEvaluator:
    """Lazy-loaded semantic metrics evaluator with robust fallbacks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, enabled: bool = False):
        cfg = config or {}
        self.cfg = SemanticMetricsConfig(
            enabled=bool(enabled or cfg.get("enabled", False)),
            embedding_model=str(cfg.get("embedding_model", "all-mpnet-base-v2")),
            nli_model=str(cfg.get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")),
            max_tokens_for_bertscore=int(cfg.get("max_tokens_for_bertscore", 80)),
        )

        self._embedder = None
        self._nli_pipeline = None
        self._init_errors: List[str] = []

        if self.cfg.enabled:
            self._lazy_init_embedding()
            self._lazy_init_nli()

    @property
    def active(self) -> bool:
        return self.cfg.enabled

    @property
    def init_errors(self) -> List[str]:
        return list(self._init_errors)

    def _lazy_init_embedding(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer(self.cfg.embedding_model)
        except Exception as e:
            self._init_errors.append(f"embedding_init_failed: {e}")
            self._embedder = None

    def _lazy_init_nli(self) -> None:
        if self._nli_pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore

            self._nli_pipeline = pipeline(
                "text-classification",
                model=self.cfg.nli_model,
                tokenizer=self.cfg.nli_model,
                truncation=True,
            )
        except Exception as e:
            self._init_errors.append(f"nli_init_failed: {e}")
            self._nli_pipeline = None

    def semantic_similarity(self, prediction: str, reference: str) -> float:
        if not prediction.strip() or not reference.strip():
            return 0.0

        if self._embedder is None:
            return _token_f1(prediction, reference)

        try:
            emb = self._embedder.encode([prediction, reference], normalize_embeddings=True)
            return _cosine(np.array(emb[0]), np.array(emb[1]))
        except Exception:
            return _token_f1(prediction, reference)

    def bertscore_proxy(self, prediction: str, reference: str) -> float:
        """
        BERTScore-like proxy:
        token-level max-sim alignment using embedding model.
        """
        p_toks = _tokens(prediction)[: self.cfg.max_tokens_for_bertscore]
        r_toks = _tokens(reference)[: self.cfg.max_tokens_for_bertscore]
        if not p_toks or not r_toks:
            return 0.0

        if self._embedder is None:
            return _token_f1(prediction, reference)

        try:
            p_emb = np.array(self._embedder.encode(p_toks, normalize_embeddings=True))
            r_emb = np.array(self._embedder.encode(r_toks, normalize_embeddings=True))

            sim = np.matmul(p_emb, r_emb.T)
            p_max = sim.max(axis=1)
            r_max = sim.max(axis=0)

            precision = float(np.mean(p_max)) if len(p_max) else 0.0
            recall = float(np.mean(r_max)) if len(r_max) else 0.0
            return _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        except Exception:
            return _token_f1(prediction, reference)

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Returns probability-like entailment score in [0,1].
        Fallback: lexical overlap proxy.
        """
        premise = premise.strip()
        hypothesis = hypothesis.strip()
        if not premise or not hypothesis:
            return 0.0

        if self._nli_pipeline is None:
            return _token_f1(premise, hypothesis)

        try:
            # NLI models often take: "premise </s></s> hypothesis"
            text = f"{premise} </s></s> {hypothesis}"
            out = self._nli_pipeline(text, top_k=None)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                # Some pipelines return list[dict], others list[list[dict]]
                labels = out
            elif isinstance(out, list) and out and isinstance(out[0], list):
                labels = out[0]
            else:
                return _token_f1(premise, hypothesis)

            ent = 0.0
            neu = 0.0
            con = 0.0
            for row in labels:
                label = str(row.get("label", "")).lower()
                score = float(row.get("score", 0.0))
                if "entail" in label:
                    ent = max(ent, score)
                elif "neutral" in label:
                    neu = max(neu, score)
                elif "contrad" in label:
                    con = max(con, score)

            # Penalize contradiction, keep normalized positive signal
            return max(0.0, min(1.0, ent - 0.5 * con + 0.1 * neu))
        except Exception:
            return _token_f1(premise, hypothesis)

    def numeric_unit_consistency(self, answer: str, context: str) -> float:
        """
        Numeric grounding score:
        - if answer has no numbers, returns 1.0 (non-numeric answer)
        - otherwise fraction of answer numbers also present in context (exact string)
        plus small bonus if answer units appear in context.
        """
        ans_nums = _NUM_RE.findall(answer or "")
        if not ans_nums:
            return 1.0

        ctx = _norm_text(context)
        hits = sum(1 for n in ans_nums if n.lower() in ctx)
        base = _safe_div(hits, len(ans_nums))

        # lightweight unit check
        units = ["m/s", "kg", "j", "ev", "k", "hz", "nm", "cm", "mm", "mol", "pa", "%"]
        ans_l = _norm_text(answer)
        unit_mentions = [u for u in units if u in ans_l]
        if not unit_mentions:
            return base

        unit_hits = sum(1 for u in unit_mentions if u in ctx)
        bonus = 0.2 * _safe_div(unit_hits, len(unit_mentions))
        return max(0.0, min(1.0, base + bonus))

    def uncertainty_score(self, answer: str, context: str, expected_answer: Optional[str] = None) -> float:
        """
        Confidence/uncertainty proxy in [0,1], higher means more uncertain.

        Signals:
        - low lexical grounding -> higher uncertainty
        - hedging language -> higher uncertainty
        - explicit insufficient-context -> very high uncertainty
        """
        a = _norm_text(answer)
        c = _norm_text(context)

        if not a:
            return 1.0

        if "insufficient context" in a or "cannot be determined" in a:
            return 0.95

        toks = _tokens(a)
        if not toks:
            return 1.0

        grounded = sum(1 for t in toks if t in c)
        grounding_ratio = _safe_div(grounded, len(toks))

        hedges = ["maybe", "likely", "possibly", "unclear", "unknown", "might", "perhaps"]
        hedge_count = sum(1 for h in hedges if h in a)
        hedge_score = min(1.0, hedge_count / 2.0)

        # lower grounding => higher uncertainty
        base_unc = 1.0 - grounding_ratio
        score = 0.8 * base_unc + 0.2 * hedge_score
        return max(0.0, min(1.0, score))

    def evaluate(self, prediction: str, reference: str, context: str) -> Dict[str, float]:
        if not self.cfg.enabled:
            return {
                "semantic_similarity": 0.0,
                "bertscore_f1": 0.0,
                "entailment_score": 0.0,
                "factual_consistency": 0.0,
                "numeric_unit_consistency": 0.0,
                "uncertainty_score": 0.0,
            }

        sim = self.semantic_similarity(prediction, reference)
        bsf1 = self.bertscore_proxy(prediction, reference)
        ent = self.entailment_score(context, prediction)
        fact = ent
        num = self.numeric_unit_consistency(prediction, context)
        unc = self.uncertainty_score(prediction, context, expected_answer=reference)

        return {
            "semantic_similarity": float(sim),
            "bertscore_f1": float(bsf1),
            "entailment_score": float(ent),
            "factual_consistency": float(fact),
            "numeric_unit_consistency": float(num),
            "uncertainty_score": float(unc),
        }
