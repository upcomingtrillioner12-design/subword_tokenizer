#!/usr/bin/env python3
"""
Evaluate Phase 2 LoRA adapter checkpoints and rank by evaluation loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train
from scripts.phase2_lora_finetune import BatchConfig, evaluate, inject_lora, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and rank LoRA adapters")
    p.add_argument("--config", default=str(ROOT / "config" / "phase2_lora_config.yaml"))
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints" / "phase2_lora"))
    p.add_argument("--pattern", default="lora_adapter_*.pt")
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--output", default=str(ROOT / "checkpoints" / "phase2_lora" / "phase2_evaluation_report.json"))
    return p.parse_args()


def build_base_model(cfg: Dict):
    model_cfg = cfg["model"]
    summary = json.loads((ROOT / model_cfg["base_summary"]).read_text(encoding="utf-8"))
    vocab_size = stream_train._resolve_vocab_size(ROOT / model_cfg["tokenizer_model"])
    model = stream_train.TinyLM(
        vocab_size=vocab_size,
        d_model=int(summary["d_model"]),
        n_layers=int(summary["n_layers"]),
        n_heads=int(summary["n_heads"]),
    )
    state = torch.load(ROOT / model_cfg["base_checkpoint"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def evaluate_adapter(ckpt_path: Path, cfg: Dict, tokens: np.ndarray, device: str, eval_steps: int) -> float:
    model = build_base_model(cfg)
    inject_lora(model, cfg["lora"])

    payload = torch.load(ckpt_path, map_location="cpu")
    lora_state = payload.get("lora_state", {})
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in adapter state: {unexpected}")

    model.to(device)
    batch_cfg = BatchConfig(
        seq_len=int(cfg["training"]["seq_len"]),
        batch_size=int(cfg["training"]["batch_size"]),
    )
    loss = evaluate(model, tokens, batch_cfg, device, eval_steps)
    return float(loss)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    device = resolve_device(cfg["training"].get("device", "auto"))

    split_bin = cfg["training"]["val_bin"] if args.eval_split == "val" else cfg["training"].get("test_bin", cfg["training"]["val_bin"])
    split_tokens = np.fromfile(ROOT / split_bin, dtype=np.uint16)

    ckpt_dir = Path(args.checkpoints_dir)
    checkpoints = sorted(ckpt_dir.glob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No adapter checkpoints found at {ckpt_dir} with pattern {args.pattern}")

    results: List[Dict] = []
    for ckpt in checkpoints:
        loss = evaluate_adapter(ckpt, cfg, split_tokens, device, args.eval_steps)
        print(f"{ckpt.name}: eval_loss={loss:.4f}")
        results.append({"checkpoint": str(ckpt), "eval_loss": loss})

    ranked = sorted(results, key=lambda x: x["eval_loss"])
    report = {
        "device": device,
        "eval_split": args.eval_split,
        "eval_steps": args.eval_steps,
        "best": ranked[0],
        "ranking": ranked,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nBest adapter: {Path(ranked[0]['checkpoint']).name} | loss={ranked[0]['eval_loss']:.4f}")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
