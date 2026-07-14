#!/usr/bin/env python3
"""
Model Evaluation: Generate Text and Evaluate Quality

Tests the trained Phase 1 model on sample prompts with detailed metrics.
"""

import json
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
if not ROOT.name.endswith("slm_v0"):
    ROOT = ROOT.parent

CHECKPOINT_PATH = ROOT / "checkpoints" / "prototype_genfix_v2_final.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Vocabulary mapping (simplified, mirrors the tokenizer)
VOCAB = {
    0: "<PAD>",
    1: "<UNK>",
    2: "<EOS>",
    3: "<BOS>",
}

# Common words (from tokenizer)
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "is", "was", "are", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "she",
    "he", "me", "him", "them", "us", "which", "who", "what", "when",
    "where", "why", "how", "if", "then", "more", "most", "some", "many",
    "first", "last", "good", "new", "other", "old", "very", "well",
    "only", "over", "out", "after", "before", "can", "could", "should",
    "must", "may", "make", "made", "get", "got", "go", "come", "came",
    "take", "took", "think", "thought", "say", "said", "see", "saw",
    "know", "knew", "find", "found", "give", "gave", "use", "used",
    "mean", "meant", "ask", "asked", "work", "worked", "call", "called",
    "try", "tried", "let", "told", "pay", "put", "seem", "seemed",
    "believe", "believed", "hold", "held", "bring", "brought", "happen",
    "happened", "write", "written", "provide", "provided", "leave",
    "left", "feel", "felt", "begin", "began", "stand", "stood", "look",
    "looked", "own", "owned", "want", "wanted", "show", "showed",
    "hear", "heard", "let", "left", "meet", "met", "run", "ran",
    "read", "read", "allow", "allowed", "add", "added", "spend",
    "spent", "grow", "grew", "open", "opened", "walk", "walked",
    "win", "won", "offer", "offered", "remember", "remembered",
]

# Build vocab
for i, word in enumerate(COMMON_WORDS, start=4):
    if i >= 32000:
        break
    VOCAB[i] = word

# Fill rest with generic tokens
for i in range(len(VOCAB), 32000):
    VOCAB[i] = f"<TOKEN_{i}>"


# ============================================================================
# Model Architecture
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
# Generation and Evaluation
# ============================================================================

def token_to_word(token_id: int) -> str:
    """Convert token ID to word."""
    return VOCAB.get(token_id, f"<TOKEN_{token_id}>")


def tokens_to_text(token_ids: List[int]) -> str:
    """Convert token list to readable text."""
    words = []
    for tid in token_ids:
        word = token_to_word(tid)
        if word == "<EOS>":
            words.append("[END]")
        elif word.startswith("<"):
            words.append(word)
        else:
            words.append(word)
    return " ".join(words)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: List[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_id: int = 2,
) -> Dict:
    """Generate text from a prompt with detailed metrics."""
    
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = list(prompt_ids)
    
    generation_tokens = []
    logits_log = []
    eos_probs = []
    entropy_log = []
    
    for step in range(max_new_tokens):
        # Forward pass
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        
        # Get probabilities
        probs = F.softmax(next_logits, dim=-1)
        eos_prob = float(probs[eos_token_id].item())
        eos_probs.append(eos_prob)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        entropy_log.append(float(entropy.item()))
        
        # Top-k sampling
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
            probs_sorted = torch.zeros_like(probs)
            probs_sorted[top_k_indices] = top_k_probs
            probs = probs_sorted / probs_sorted.sum()
        
        # Sample
        next_token = torch.multinomial(probs, 1).item()
        
        logits_log.append({
            "step": step,
            "token": next_token,
            "token_prob": float(probs[next_token].item()),
            "eos_prob": eos_prob,
            "entropy": entropy.item(),
            "top_5_probs": float(torch.topk(probs, 5)[0].mean().item()),
        })
        
        generated.append(next_token)
        generation_tokens.append(next_token)
        
        # Prepare for next iteration
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        x = torch.cat([x, next_token_t], dim=1)
        
        # Stop on EOS
        if next_token == eos_token_id:
            break
    
    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generation_tokens,
        "full_ids": generated,
        "text": tokens_to_text(generated),
        "num_tokens_generated": len(generation_tokens),
        "stopped_by_eos": generation_tokens[-1] == eos_token_id if generation_tokens else False,
        "avg_eos_prob": float(np.mean(eos_probs)) if eos_probs else 0,
        "avg_entropy": float(np.mean(entropy_log)) if entropy_log else 0,
        "logits_details": logits_log,
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, num_samples: int = 5) -> Dict:
    """Comprehensive model evaluation."""
    
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Sample prompts
    prompts = [
        ([1], "Start token"),
        ([1, 10, 12], "Short prompt: be in"),
        ([1, 5, 8, 15], "Medium prompt: be and that"),
        ([1, 0], "BOS with padding"),
        ([10], "Single token"),
    ]
    
    results = []
    
    for prompt_ids, prompt_desc in prompts[:num_samples]:
        print(f"\n{'─' * 80}")
        print(f"Prompt: {prompt_desc}")
        print(f"  Token IDs: {prompt_ids}")
        print(f"  Text: {tokens_to_text(prompt_ids)}")
        
        result = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
        )
        
        results.append(result)
        
        print(f"\nGeneration:")
        print(f"  {result['text']}")
        print(f"\nMetrics:")
        print(f"  Tokens generated: {result['num_tokens_generated']}")
        print(f"  Stopped by EOS: {result['stopped_by_eos']}")
        print(f"  Avg EOS probability: {result['avg_eos_prob']:.6f}")
        print(f"  Avg entropy: {result['avg_entropy']:.4f}")
        
        if result['logits_details']:
            first_3 = result['logits_details'][:3]
            print(f"\n  First 3 tokens detail:")
            for detail in first_3:
                print(
                    f"    Step {detail['step']}: token={detail['token']}, "
                    f"prob={detail['token_prob']:.4f}, "
                    f"eos_prob={detail['eos_prob']:.6f}, "
                    f"entropy={detail['entropy']:.4f}"
                )
    
    # Aggregate stats
    print(f"\n{'─' * 80}")
    print("Aggregate Statistics:")
    
    avg_tokens = np.mean([r['num_tokens_generated'] for r in results])
    avg_eos = np.mean([r['avg_eos_prob'] for r in results])
    avg_entropy = np.mean([r['avg_entropy'] for r in results])
    eos_stopped = sum(1 for r in results if r['stopped_by_eos'])
    
    print(f"  Avg tokens per generation: {avg_tokens:.2f}")
    print(f"  Generations stopped by EOS: {eos_stopped}/{len(results)}")
    print(f"  Avg EOS probability: {avg_eos:.6f}")
    print(f"  Avg entropy: {avg_entropy:.4f}")
    
    return {
        "num_samples": len(results),
        "avg_tokens_generated": avg_tokens,
        "eos_stopped_count": eos_stopped,
        "avg_eos_prob": avg_eos,
        "avg_entropy": avg_entropy,
        "samples": results,
    }


def main():
    print("=" * 80)
    print("PHASE 1 MODEL EVALUATION")
    print("=" * 80)
    
    # Load checkpoint
    if not CHECKPOINT_PATH.exists():
        print(f"✗ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Create model
    model = TinyLM(vocab_size=32000, d_model=384, n_layers=6, n_heads=4)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: ~35.2M")
    
    # Evaluate
    eval_results = evaluate_model(model, num_samples=5)
    
    # Save results
    output_path = ROOT / "checkpoints" / "prototype_genfix_v2_eval_results.json"
    
    # Make results JSON serializable
    serializable_results = {
        "num_samples": eval_results["num_samples"],
        "avg_tokens_generated": float(eval_results["avg_tokens_generated"]),
        "eos_stopped_count": int(eval_results["eos_stopped_count"]),
        "avg_eos_prob": float(eval_results["avg_eos_prob"]),
        "avg_entropy": float(eval_results["avg_entropy"]),
        "samples": [
            {
                "prompt_ids": s["prompt_ids"],
                "generated_ids": s["generated_ids"],
                "text": s["text"],
                "num_tokens_generated": s["num_tokens_generated"],
                "stopped_by_eos": s["stopped_by_eos"],
                "avg_eos_prob": float(s["avg_eos_prob"]),
                "avg_entropy": float(s["avg_entropy"]),
            }
            for s in eval_results["samples"]
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
