#!/usr/bin/env python3
"""
Phase 2 LoRA fine-tuning for TinyLM.

- Loads base checkpoint from Phase 1
- Injects LoRA modules into selected linear layers
- Trains only LoRA params on offline tokenized corpus (.bin)
- Saves adapter-only checkpoints
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class BatchConfig:
    seq_len: int
    batch_size: int


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.r = r
        self.scale = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scale

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 LoRA fine-tuning for TinyLM")
    p.add_argument("--config", default=str(ROOT / "config" / "phase2_lora_config.yaml"))
    return p.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(cfg: Dict) -> nn.Module:
    model_cfg = cfg["model"]
    summary_path = ROOT / model_cfg["base_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    tokenizer_model = ROOT / model_cfg["tokenizer_model"]
    vocab_size = stream_train._resolve_vocab_size(tokenizer_model)

    model = stream_train.TinyLM(
        vocab_size=vocab_size,
        d_model=int(summary["d_model"]),
        n_layers=int(summary["n_layers"]),
        n_heads=int(summary["n_heads"]),
    )

    base_ckpt = ROOT / model_cfg["base_checkpoint"]
    state = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def _set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora(model: nn.Module, lora_cfg: Dict) -> List[str]:
    target_modules: List[str] = lora_cfg["target_modules"]
    r = int(lora_cfg["r"])
    alpha = float(lora_cfg["alpha"])
    dropout = float(lora_cfg.get("dropout", 0.0))

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(t in name for t in target_modules):
            _set_module_by_name(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)

    if not replaced:
        raise RuntimeError("No linear modules matched LoRA target_modules")
    return replaced


def freeze_non_lora_params(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n or "lora_B" in n)


def get_trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_bin(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing token bin: {path}")
    return np.fromfile(path, dtype=np.uint16)


def sample_batch(tokens: np.ndarray, cfg: BatchConfig, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - (cfg.seq_len + 1)
    if max_start <= 0:
        raise RuntimeError("Not enough tokens in dataset for requested seq_len")

    starts = np.random.randint(0, max_start, size=(cfg.batch_size,))
    x_list = []
    y_list = []
    for s in starts:
        chunk = tokens[s : s + cfg.seq_len + 1].astype(np.int64)
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)
    return x, y


def evaluate(model: nn.Module, tokens: np.ndarray, batch_cfg: BatchConfig, device: str, steps: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = sample_batch(tokens, batch_cfg, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for n, p in model.state_dict().items():
        if "lora_A" in n or "lora_B" in n:
            state[n] = p.detach().cpu()
    return state


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg["training"].get("device", "auto"))

    model = build_model(cfg)
    replaced = inject_lora(model, cfg["lora"])
    freeze_non_lora_params(model)
    model.to(device)

    train_bin = ROOT / cfg["training"]["train_bin"]
    val_bin = ROOT / cfg["training"]["val_bin"]
    train_tokens = load_bin(train_bin)
    val_tokens = load_bin(val_bin)

    batch_cfg = BatchConfig(
        seq_len=int(cfg["training"]["seq_len"]),
        batch_size=int(cfg["training"]["batch_size"]),
    )

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    max_steps = int(cfg["training"]["max_steps"])
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    eval_every = int(cfg["training"]["eval_every"])
    eval_steps = int(cfg["training"]["eval_steps"])
    save_every = int(cfg["training"]["save_every"])

    out_dir = ROOT / cfg["training"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(list(get_trainable_params(model)), lr=lr, weight_decay=wd)

    best_eval = float("inf")
    best_path = out_dir / "best_lora_adapter.pt"

    log = {
        "device": device,
        "replaced_modules": replaced,
        "train_tokens": int(train_tokens.size),
        "val_tokens": int(val_tokens.size),
        "max_steps": max_steps,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[phase2] Device: {device}")
    print(f"[phase2] LoRA modules replaced: {len(replaced)}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = sample_batch(train_tokens, batch_cfg, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            loss.backward()
            step_loss += float(loss.item())

        current_lr = cosine_lr(step, max_steps, warmup_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(f"[phase2] step={step} train_loss={step_loss:.4f} lr={current_lr:.6f}")

        if step % eval_every == 0:
            val_loss = evaluate(model, val_tokens, batch_cfg, device, eval_steps)
            print(f"[phase2] step={step} val_loss={val_loss:.4f}")

            if val_loss < best_eval:
                best_eval = val_loss
                payload = {
                    "step": step,
                    "val_loss": val_loss,
                    "lora_state": extract_lora_state(model),
                    "config": cfg,
                }
                torch.save(payload, best_path)
                print(f"[phase2] New best adapter saved: {best_path}")

        if step % save_every == 0:
            ckpt_path = out_dir / f"lora_adapter_step{step}.pt"
            payload = {
                "step": step,
                "lora_state": extract_lora_state(model),
                "config": cfg,
            }
            torch.save(payload, ckpt_path)
            print(f"[phase2] Saved checkpoint: {ckpt_path}")

    final_path = out_dir / "lora_adapter_final.pt"
    torch.save(
        {
            "step": max_steps,
            "best_val_loss": best_eval,
            "lora_state": extract_lora_state(model),
            "config": cfg,
        },
        final_path,
    )

    summary = {
        "best_val_loss": best_eval,
        "best_adapter": str(best_path),
        "final_adapter": str(final_path),
        "max_steps": max_steps,
        "output_dir": str(out_dir),
    }
    (out_dir / "phase2_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase2] Training complete. Summary: {out_dir / 'phase2_train_summary.json'}")


if __name__ == "__main__":
    main()
