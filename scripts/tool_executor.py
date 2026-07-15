#!/usr/bin/env python3
"""
Lightweight tool integration for Phase 4 Task 5.

Implements safe utility tools that can be called before generation:
- calculator: evaluates simple arithmetic expressions in query
- keyword_lookup: finds glossary hints for common physics terms
"""

from __future__ import annotations

import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Dict, List


_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval_expr(expr: str) -> float:
    """Safely evaluate a numeric expression using AST whitelist."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            return float(_ALLOWED_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return float(_ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand)))
        raise ValueError("Unsafe or unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


@dataclass
class ToolExecutor:
    enabled: bool = True

    def __post_init__(self) -> None:
        self._glossary = {
            "de broglie": "de Broglie relation: lambda = h/p.",
            "born rule": "Born rule: probability density is |psi|^2.",
            "heisenberg": "Heisenberg uncertainty: Delta x Delta p >= hbar/2.",
            "gauss": "Gauss law: div(E) = rho/epsilon0.",
            "faraday": "Faraday law: curl(E) = -dB/dt.",
            "lorentz": "Lorentz force: F = q(E + v x B).",
            "partition function": "Canonical partition function: Z = sum_i exp(-E_i/kT).",
            "entropy": "Second law: entropy of isolated system does not decrease.",
            "higgs": "Higgs mechanism gives mass to W and Z bosons.",
        }

    def run(self, query: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"used": False, "hints": []}

        hints: List[str] = []

        calc = self._maybe_calculate(query)
        if calc is not None:
            hints.append(f"calculator: {calc}")

        hints.extend(self._keyword_lookup(query))

        return {
            "used": bool(hints),
            "hints": hints,
        }

    def _maybe_calculate(self, query: str) -> str | None:
        # expression-like spans with digits and operators only
        candidates = re.findall(r"([0-9][0-9\s\+\-\*/\(\)\.%\^]{2,})", query)
        for raw in candidates:
            expr = raw.replace("^", "**").strip()
            try:
                value = _safe_eval_expr(expr)
                return f"{raw.strip()} = {value:g}"
            except Exception:
                continue
        return None

    def _keyword_lookup(self, query: str) -> List[str]:
        q = query.lower()
        out: List[str] = []
        for k, v in self._glossary.items():
            if k in q:
                out.append(f"lookup[{k}]: {v}")
        return out
