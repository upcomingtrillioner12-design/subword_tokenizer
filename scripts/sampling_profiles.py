#!/usr/bin/env python3
"""Shared sampling profiles for production and canonical evaluation."""

from __future__ import annotations

from typing import Dict, Any

SAMPLING_PROFILES: Dict[str, Dict[str, Any]] = {
    "production": {
        "max_tokens": 64,
        "temperature": 2.0,
        "top_k": 100,
        "top_p": None,
    },
    "canonical": {
        "max_tokens": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": None,
    },
}


def resolve_sampling_config(
    profile: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Dict[str, Any]:
    """Resolve generation config with precedence CLI override > profile default."""
    if profile not in SAMPLING_PROFILES:
        raise ValueError(f"Unknown sampling profile: {profile}")

    base = SAMPLING_PROFILES[profile].copy()
    if max_tokens is not None:
        base["max_tokens"] = int(max_tokens)
    if temperature is not None:
        base["temperature"] = float(temperature)
    if top_k is not None:
        base["top_k"] = int(top_k)
    if top_p is not None:
        base["top_p"] = float(top_p)

    return base
