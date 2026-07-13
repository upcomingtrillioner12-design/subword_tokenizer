#!/usr/bin/env python3
"""Phase 3 Task 5: Evaluate Phase 1 vs Phase 2 on held-out test split."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "slm_v0") not in sys.path:
    sys.path.insert(0, str(ROOT / "slm_v0"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

phase2_path = ROOT / "scripts" / "phase2_lora_finetune.py"
spec = importlib.util.spec_from_file_location("phase2_lora_finetune_local", phase2_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to import {phase2_path}")
phase2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase2
spec.loader.exec_module(phase2)

BatchConfig = phase2.BatchConfig
build_model = phase2.build_model
evaluate = phase2.evaluate
inject_lora = phase2.inject_lora
load_config = phase2.load_config
resolve_device = phase2.resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Phase 1 vs Phase 2 on test split")
    p.add_argument("--config", default=str(ROOT / "config" / "phase2_lora_config.yaml"))
    p.add_argument("--adapter", default=str(ROOT / "checkpoints" / "phase2_lora" / "best_lora_adapter.pt"))
    p.add_argument("--test-bin", default=None, help="Optional explicit path to test.bin")
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "results" / "phase3_test_set_evaluation.json"))
    return p.parse_args()


def resolve_test_bin(cfg: Dict, override: str | None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p

    training = cfg.get("training", {})
    if "test_bin" in training:
        p = Path(training["test_bin"])
        return p if p.is_absolute() else ROOT / p

    # Fallback: derive from val_bin path
    val_bin = Path(training["val_bin"])
    candidate = Path(str(val_bin).replace("val.bin", "test.bin"))
    candidate = candidate if candidate.is_absolute() else ROOT / candidate
    if candidate.exists():
        return candidate

    # Final fallbacks (support nested and parent workspace layouts)
    fallback_local = ROOT / "data" / "offline_physics" / "test.bin"
    if fallback_local.exists():
        return fallback_local
    fallback_parent = ROOT.parent / "data" / "offline_physics" / "test.bin"
    return fallback_parent


def resolve_artifact(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        if p.exists():
            return p
        # If absolute path points to nested repo, remap to parent workspace mirror
        try:
            rel = p.relative_to(ROOT)
            alt = ROOT.parent / rel
            if alt.exists():
                return alt
        except ValueError:
            pass
    local = ROOT / p
    if local.exists():
        return local
    parent = ROOT.parent / p
    if parent.exists():
        return parent
    return local if p.is_absolute() else parent


def normalize_cfg_paths(cfg: Dict) -> Dict:
    model_cfg = cfg.get("model", {})
    if "base_summary" in model_cfg:
        model_cfg["base_summary"] = str(resolve_artifact(model_cfg["base_summary"]))
    if "base_checkpoint" in model_cfg:
        model_cfg["base_checkpoint"] = str(resolve_artifact(model_cfg["base_checkpoint"]))
    if "tokenizer_model" in model_cfg:
        model_cfg["tokenizer_model"] = str(resolve_artifact(model_cfg["tokenizer_model"]))
    cfg["model"] = model_cfg
    return cfg


def load_adapter_into_model(model: torch.nn.Module, adapter_path: Path, lora_cfg: Dict) -> None:
    inject_lora(model, lora_cfg)
    payload = torch.load(adapter_path, map_location="cpu")
    lora_state = payload.get("lora_state", payload)
    _, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in adapter checkpoint: {unexpected}")


def main() -> None:
    args = parse_args()

    cfg = load_config(Path(args.config))
    cfg = normalize_cfg_paths(cfg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(cfg["training"].get("device", "auto"))
    test_bin = resolve_test_bin(cfg, args.test_bin)
    if not test_bin.exists():
        test_bin = resolve_artifact(test_bin)
    if not test_bin.exists():
        raise FileNotFoundError(f"Test split not found: {test_bin}")

    tokens = np.fromfile(test_bin, dtype=np.uint16)
    batch_cfg = BatchConfig(
        seq_len=int(cfg["training"]["seq_len"]),
        batch_size=int(cfg["training"]["batch_size"]),
    )

    t0 = time.perf_counter()

    # Phase 1 baseline
    model_phase1 = build_model(cfg).to(device)
    loss_phase1 = float(evaluate(model_phase1, tokens, batch_cfg, device, args.eval_steps))

    # Phase 2 best adapter
    model_phase2 = build_model(cfg)
    adapter_path = resolve_artifact(args.adapter)
    load_adapter_into_model(model_phase2, adapter_path, cfg["lora"])
    model_phase2 = model_phase2.to(device)
    loss_phase2 = float(evaluate(model_phase2, tokens, batch_cfg, device, args.eval_steps))

    dt = time.perf_counter() - t0

    ppl_phase1 = math.exp(loss_phase1)
    ppl_phase2 = math.exp(loss_phase2)
    loss_delta = loss_phase1 - loss_phase2
    relative_gain = (loss_delta / loss_phase1) * 100.0 if loss_phase1 > 0 else 0.0

    report = {
        "metadata": {
            "task": "phase3_task5_test_set_evaluation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "config": str(args.config),
            "adapter": str(adapter_path),
            "test_bin": str(test_bin),
            "eval_steps": args.eval_steps,
            "seed": args.seed,
            "elapsed_seconds": dt,
            "num_test_tokens": int(tokens.shape[0]),
        },
        "results": {
            "phase1_test_loss": loss_phase1,
            "phase2_test_loss": loss_phase2,
            "phase1_perplexity": ppl_phase1,
            "phase2_perplexity": ppl_phase2,
            "loss_delta": loss_delta,
            "relative_gain_percent": relative_gain,
        },
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== TEST SET EVALUATION (TASK 5) ===")
    print(f"Device:            {device}")
    print(f"Test split:        {test_bin}")
    print(f"Eval steps:        {args.eval_steps}")
    print(f"Phase 1 loss:      {loss_phase1:.6f}")
    print(f"Phase 2 loss:      {loss_phase2:.6f}")
    print(f"Loss improvement:  {loss_delta:.6f} ({relative_gain:.2f}%)")
    print(f"Phase 1 ppl:       {ppl_phase1:.3f}")
    print(f"Phase 2 ppl:       {ppl_phase2:.3f}")
    print(f"Saved:             {out}")


if __name__ == "__main__":
    main()
