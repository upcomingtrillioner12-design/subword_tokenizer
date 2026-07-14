#!/usr/bin/env python3
"""
Phase 4 Task 1c: Fast Generation-Aware Phase 1 Retraining

Uses pre-tokenized Phase 2 corpus (binary) to train Phase 1 base model
with generation-aware loss (EOS penalty + generation monitoring).

Much faster than streaming arXiv (~30-60 min vs 4-6 hours).
"""

import json
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Path setup
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
if not ROOT.name.endswith("slm_v0"):
    ROOT = ROOT.parent

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_DIR = ROOT / "checkpoints"


# ============================================================================
# Model Architecture (TinyLM from stream_train.py)
# ============================================================================

class TinyLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True),
            n_layers
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.head(self.transformer(self.embed(x) + self.pos_embed(pos)))


# ============================================================================
# Generation Evaluation
# ============================================================================

@torch.no_grad()
def estimate_generation_tokens(
    model: nn.Module,
    prompt_ids: list,
    max_new_tokens: int = 32,
    eos_token_id: int = 2,
) -> int:
    """Estimate how many tokens model generates before EOS."""
    if not prompt_ids:
        prompt_ids = [1]
    
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = x
    count = 0
    
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
        if next_token == eos_token_id:
            break
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        generated = torch.cat([generated, next_token_t], dim=1)
        count += 1
    
    return count


@torch.no_grad()
def evaluate_generation_health(
    model: nn.Module,
    prompts: list,
    max_new_tokens: int = 32,
    eos_token_id: int = 2,
) -> dict:
    """Evaluate generation quality on sample prompts."""
    model_was_training = model.training
    model.eval()
    
    token_counts = []
    eos_probs = []
    
    for prompt_ids in prompts:
        if isinstance(prompt_ids, str):
            # Skip string prompts for offline training
            continue
        
        x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        eos_probs.append(float(probs[eos_token_id].item()))
        
        token_counts.append(
            estimate_generation_tokens(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        )
    
    if model_was_training:
        model.train()
    
    avg_gen_tokens = float(sum(token_counts) / max(len(token_counts), 1)) if token_counts else 0
    max_gen_tokens = int(max(token_counts) if token_counts else 0)
    avg_eos_prob = float(sum(eos_probs) / max(len(eos_probs), 1)) if eos_probs else 0
    
    return {
        "avg_gen_tokens": avg_gen_tokens,
        "max_gen_tokens": max_gen_tokens,
        "avg_eos_prob": avg_eos_prob,
    }


# ============================================================================
# Data Loading from Phase 2 Binary Corpus
# ============================================================================

def load_corpus_file(path: Path, max_seqs: Optional[int] = None) -> torch.Tensor:
    """Load tokenized corpus from binary file (uint16 format)."""
    import numpy as np
    
    print(f"Loading {path}...")
    # Load as numpy uint16 binary file
    data_np = np.fromfile(path, dtype=np.uint16)
    data = torch.from_numpy(data_np).to(torch.long)
    
    if max_seqs and len(data) > max_seqs * 256:
        data = data[:max_seqs * 256]
        print(f"  Truncated to {max_seqs} sequences")
    
    print(f"  Shape: {data.shape}")
    print(f"  Min token: {data.min()}, Max token: {data.max()}")
    return data


def make_batches_from_data(data: torch.Tensor, batch_size: int = 4, seq_len: int = 256):
    """Create training batches from pre-tokenized data."""
    num_batches = len(data) // (batch_size * seq_len)
    for i in range(num_batches):
        start = i * batch_size * seq_len
        end = start + batch_size * seq_len
        batch = data[start:end].reshape(batch_size, seq_len)
        
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        yield x.to(DEVICE), y.to(DEVICE)


# ============================================================================
# Training with Generation-Aware Loss
# ============================================================================

def train(
    corpus_path: Path,
    output_dir: Path,
    checkpoint_prefix: str = "prototype_genfix_v2",
    batch_size: int = 4,
    seq_len: int = 256,
    max_steps: int = 6000,
    lr: float = 0.0003,
    eos_token_id: int = 2,
    non_eos_threshold: float = 0.25,
    eos_penalty_weight: float = 2.0,
    generation_eval_every_steps: int = 100,
    early_stop_avg_gen_tokens: float = 10.0,
    early_stop_patience: int = 2,
):
    """Train Phase 1 model with generation-aware loss."""
    
    print(f"\n{'='*80}")
    print("PHASE 4 TASK 1c: Fast Generation-Aware Phase 1 Retraining")
    print(f"{'='*80}\n")
    
    # Load corpus
    print(f"Loading corpus from {corpus_path}")
    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        return
    
    train_data = load_corpus_file(corpus_path)
    
    # Create model
    vocab_size = 32000
    d_model = 384
    n_layers = 6
    n_heads = 4
    
    model = TinyLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    model.to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"\nModel: TinyLM")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  params≈35.2M, device={DEVICE}")
    
    print(f"\nTraining config:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  max_steps={max_steps}, lr={lr}")
    print(f"  eos_penalty_weight={eos_penalty_weight}, non_eos_threshold={non_eos_threshold}")
    print(f"  generation_eval_every_steps={generation_eval_every_steps}")
    print(f"  early_stop: patience={early_stop_patience}, target_gen_tokens={early_stop_avg_gen_tokens}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    generation_history = []
    consecutive_hits = 0
    total_loss = 0
    
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    step = 0
    for x, y in make_batches_from_data(train_data, batch_size=batch_size, seq_len=seq_len):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # EOS penalty: encourage P(non-EOS) >= threshold
        probs = F.softmax(logits, dim=-1)
        eos_probs = probs[..., eos_token_id]
        eos_max_prob = max(0.0, 1.0 - non_eos_threshold)
        eos_penalty = torch.relu(eos_probs - eos_max_prob).pow(2).mean()
        
        # Combined loss
        loss = ce_loss + (eos_penalty_weight * eos_penalty)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        if step % 10 == 0:
            avg_loss = total_loss / step
            print(
                f"Step {step:5d} | Loss: {loss.item():.4f} (avg={avg_loss:.4f}) | "
                f"ce={ce_loss.item():.4f}, eos_pen={eos_penalty.item():.6f}, eos_prob={eos_probs.mean().item():.4f}"
            )
        
        # Generation evaluation
        if generation_eval_every_steps > 0 and step % generation_eval_every_steps == 0:
            # Extract some prompt_ids from batch for evaluation
            sample_prompts = [x[i, :32].tolist() for i in range(min(2, x.shape[0]))]
            
            health = evaluate_generation_health(
                model=model,
                prompts=sample_prompts,
                max_new_tokens=32,
                eos_token_id=eos_token_id,
            )
            generation_history.append({"step": step, **health})
            
            print(
                f"  [gen-eval] step={step} avg_gen_tokens={health['avg_gen_tokens']:.2f} "
                f"max_gen_tokens={health['max_gen_tokens']} avg_eos_prob={health['avg_eos_prob']:.4f}"
            )
            
            if health["avg_gen_tokens"] >= early_stop_avg_gen_tokens:
                consecutive_hits += 1
                print(
                    f"  [gen-eval] ✓ improvement hit {consecutive_hits}/{early_stop_patience} "
                    f"(target avg_gen_tokens >= {early_stop_avg_gen_tokens})"
                )
                
                if consecutive_hits >= early_stop_patience:
                    print(
                        f"  [early-stop] generation target reached! "
                        f"Stopping at step {step}."
                    )
                    break
            else:
                consecutive_hits = 0
        
        # Save checkpoint every 500 steps
        if step % 500 == 0:
            ckpt_path = output_dir / f"{checkpoint_prefix}_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
        
        if step >= max_steps:
            print(f"Reached max_steps={max_steps}")
            break
    
    # Final checkpoint
    final_ckpt_path = output_dir / f"{checkpoint_prefix}_final.pt"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"\nSaved final checkpoint: {final_ckpt_path}")
    
    # Save summary
    summary = {
        "device": str(DEVICE),
        "model": "TinyLM",
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "corpus": str(corpus_path),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_steps": max_steps,
        "steps_completed": step,
        "lr": lr,
        "generation_aware": {
            "eos_token_id": eos_token_id,
            "non_eos_threshold": non_eos_threshold,
            "eos_penalty_weight": eos_penalty_weight,
            "generation_eval_every_steps": generation_eval_every_steps,
            "early_stop_avg_gen_tokens": early_stop_avg_gen_tokens,
            "early_stop_patience": early_stop_patience,
        },
        "final_avg_loss": total_loss / step if step > 0 else 0,
        "generation_history": generation_history,
    }
    
    summary_path = output_dir / f"{checkpoint_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Final checkpoint: {final_ckpt_path}")
    print(f"Steps completed: {step}")
    print(f"Final avg loss: {summary['final_avg_loss']:.4f}")


if __name__ == "__main__":
    # Use Phase 2 offline corpus (already tokenized and binary)
    corpus_path = ROOT / "data" / "offline_physics" / "train.bin"
    
    if not corpus_path.exists():
        print(f"✗ Corpus not found at {corpus_path}")
        print(f"Run Phase 2 corpus prep first: scripts/prepare_offline_corpus_multicategory.py")
        sys.exit(1)
    
    train(
        corpus_path=corpus_path,
        output_dir=CHECKPOINT_DIR,
        checkpoint_prefix="prototype_genfix_v2",
        batch_size=4,
        seq_len=256,
        max_steps=6000,
        lr=0.0003,
        eos_token_id=2,
        non_eos_threshold=0.25,
        eos_penalty_weight=2.0,
        generation_eval_every_steps=100,
        early_stop_avg_gen_tokens=10.0,
        early_stop_patience=2,
    )
