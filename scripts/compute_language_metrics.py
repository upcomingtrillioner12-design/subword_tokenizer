#!/usr/bin/env python3
"""Phase 3 Task 6: Compute Perplexity + BLEU-4 for Phase 1 vs Phase 2."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

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

import stream_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute language metrics (PPL + BLEU-4)")
    p.add_argument("--config", default=str(ROOT / "config" / "phase2_lora_config.yaml"))
    p.add_argument("--adapter", default=str(ROOT / "checkpoints" / "phase2_lora" / "best_lora_adapter.pt"))
    p.add_argument("--references", default=str(ROOT / "data" / "bleu_references.json"))
    p.add_argument("--test-bin", default=None)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--max-gen-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "results" / "language_metrics.json"))
    return p.parse_args()


def resolve_artifact(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        if p.exists():
            return p
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
    return local


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


def resolve_test_bin(cfg: Dict, override: str | None) -> Path:
    if override:
        return resolve_artifact(override)
    training = cfg.get("training", {})
    if "test_bin" in training:
        return resolve_artifact(training["test_bin"])
    val_bin = Path(training["val_bin"])
    candidate = Path(str(val_bin).replace("val.bin", "test.bin"))
    candidate = resolve_artifact(candidate)
    if candidate.exists():
        return candidate
    return resolve_artifact("data/offline_physics/test.bin")


def load_adapter_into_model(model: torch.nn.Module, adapter_path: Path, lora_cfg: Dict) -> None:
    inject_lora(model, lora_cfg)
    payload = torch.load(adapter_path, map_location="cpu")
    lora_state = payload.get("lora_state", payload)
    _, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in adapter checkpoint: {unexpected}")


def ngrams(tokens: List[int], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu4_token_level(reference: List[int], hypothesis: List[int]) -> float:
    if len(hypothesis) == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        hyp_ngrams = ngrams(hypothesis, n)
        ref_ngrams = ngrams(reference, n)
        if not hyp_ngrams:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams[g]) for g, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        # add-1 smoothing
        precisions.append((overlap + 1.0) / (total + 1.0))

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / hyp_len))

    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return float(score)


def generate_ids(
    model: torch.nn.Module,
    prompt_tokens: List[int],
    device: str,
    max_gen_tokens: int,
    temperature: float,
) -> List[int]:
    model.eval()
    current = torch.tensor([prompt_tokens[:256]], dtype=torch.long, device=device)
    out: List[int] = []

    with torch.no_grad():
        for _ in range(max_gen_tokens):
            logits = model(current)
            next_logits = logits[0, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id in (0, 2):
                break
            out.append(next_id)
            nxt = torch.tensor([[next_id]], dtype=torch.long, device=device)
            current = torch.cat([current, nxt], dim=1)
            if current.shape[1] >= 512:
                break

    return out


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = normalize_cfg_paths(load_config(Path(args.config)))
    device = resolve_device(cfg["training"].get("device", "auto"))

    # Perplexity on held-out test split
    test_bin = resolve_test_bin(cfg, args.test_bin)
    if not test_bin.exists():
        raise FileNotFoundError(f"Test split not found: {test_bin}")

    tokens = np.fromfile(test_bin, dtype=np.uint16)
    batch_cfg = BatchConfig(
        seq_len=int(cfg["training"]["seq_len"]),
        batch_size=int(cfg["training"]["batch_size"]),
    )

    t0 = time.perf_counter()

    phase1 = build_model(cfg).to(device)
    loss_phase1 = float(evaluate(phase1, tokens, batch_cfg, device, int(args.eval_steps)))

    phase2 = build_model(cfg)
    adapter_path = resolve_artifact(args.adapter)
    load_adapter_into_model(phase2, adapter_path, cfg["lora"])
    phase2 = phase2.to(device)
    loss_phase2 = float(evaluate(phase2, tokens, batch_cfg, device, int(args.eval_steps)))

    ppl_phase1 = float(math.exp(loss_phase1))
    ppl_phase2 = float(math.exp(loss_phase2))

    # BLEU-4 on reference prompt set
    ref_path = resolve_artifact(args.references)
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    items = ref_data.get("items", ref_data)

    tokenizer_model = resolve_artifact(cfg["model"]["tokenizer_model"])
    stream_train._activate_tokenizer_model(tokenizer_model)

    bleu_rows = []
    bleu_p1 = []
    bleu_p2 = []

    for item in items:
        prompt = item["prompt"]
        reference = item["reference"]

        prompt_ids = stream_train._tokenize_with_our_model(prompt)
        ref_ids = stream_train._tokenize_with_our_model(reference)

        hyp1 = generate_ids(phase1, prompt_ids, device, args.max_gen_tokens, args.temperature)
        hyp2 = generate_ids(phase2, prompt_ids, device, args.max_gen_tokens, args.temperature)

        s1 = bleu4_token_level(ref_ids, hyp1)
        s2 = bleu4_token_level(ref_ids, hyp2)
        bleu_p1.append(s1)
        bleu_p2.append(s2)

        bleu_rows.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "reference": reference,
                "phase1_tokens": len(hyp1),
                "phase2_tokens": len(hyp2),
                "phase1_bleu4": s1,
                "phase2_bleu4": s2,
            }
        )

    avg_bleu1 = float(sum(bleu_p1) / len(bleu_p1)) if bleu_p1 else 0.0
    avg_bleu2 = float(sum(bleu_p2) / len(bleu_p2)) if bleu_p2 else 0.0

    elapsed = time.perf_counter() - t0

    report = {
        "metadata": {
            "task": "phase3_task6_language_metrics",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "config": str(args.config),
            "adapter": str(adapter_path),
            "test_bin": str(test_bin),
            "references": str(ref_path),
            "eval_steps": int(args.eval_steps),
            "max_gen_tokens": int(args.max_gen_tokens),
            "temperature": float(args.temperature),
            "seed": int(args.seed),
            "elapsed_seconds": elapsed,
        },
        "perplexity": {
            "phase1_loss": loss_phase1,
            "phase2_loss": loss_phase2,
            "phase1_ppl": ppl_phase1,
            "phase2_ppl": ppl_phase2,
            "loss_delta": loss_phase1 - loss_phase2,
            "relative_gain_percent": ((loss_phase1 - loss_phase2) / loss_phase1 * 100.0) if loss_phase1 > 0 else 0.0,
        },
        "bleu4": {
            "phase1_avg_bleu4": avg_bleu1,
            "phase2_avg_bleu4": avg_bleu2,
            "delta_bleu4": avg_bleu2 - avg_bleu1,
            "num_samples": len(bleu_rows),
            "details": bleu_rows,
        },
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== LANGUAGE METRICS (TASK 6) ===")
    print(f"Perplexity P1:    {ppl_phase1:.6f}")
    print(f"Perplexity P2:    {ppl_phase2:.6f}")
    print(f"BLEU-4 P1 avg:    {avg_bleu1:.6f}")
    print(f"BLEU-4 P2 avg:    {avg_bleu2:.6f}")
    print(f"Saved:            {out}")


if __name__ == "__main__":
    main()
