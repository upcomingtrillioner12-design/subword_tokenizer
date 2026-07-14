#!/usr/bin/env python3
"""
Hyperparameter Tuning: Temperature and Top-K Sampling

Tests different temperature and top-k values to find optimal generation parameters.
Evaluates quality metrics for each configuration.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
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

# Vocabulary mapping (mirrors the tokenizer)
VOCAB = {
    0: "<PAD>",
    1: "<UNK>",
    2: "<EOS>",
    3: "<BOS>",
}

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
# Generation Utilities
# ============================================================================

def token_to_word(token_id: int) -> str:
    """Convert token ID to word."""
    return VOCAB.get(token_id, f"<TOKEN_{token_id}>")


def tokens_to_text(token_ids: List[int], verbose: bool = False) -> str:
    """Convert token list to readable text."""
    words = []
    for tid in token_ids:
        word = token_to_word(tid)
        if word == "<EOS>":
            words.append("[END]")
        elif word.startswith("<") and verbose:
            words.append(word)
        elif not word.startswith("<"):
            words.append(word)
    return " ".join(words)


@torch.no_grad()
def generate_with_params(
    model: nn.Module,
    prompt_ids: List[int],
    temperature: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 40,
    eos_token_id: int = 2,
) -> Dict:
    """Generate text with specific temperature and top-k parameters."""
    
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = list(prompt_ids)
    
    metrics = {
        "token_probs": [],
        "eos_probs": [],
        "entropy": [],
        "top_k_usage": 0,  # How often top-k cutoff was hit
        "diversity_score": 0,  # Unique tokens / total tokens
    }
    
    unique_tokens = set()
    
    for step in range(max_new_tokens):
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        
        probs = F.softmax(next_logits, dim=-1)
        eos_prob = float(probs[eos_token_id].item())
        metrics["eos_probs"].append(eos_prob)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        metrics["entropy"].append(float(entropy.item()))
        
        # Top-k filtering
        if top_k > 0 and top_k < len(probs):
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
            
            # Check if probability mass is concentrated in top-k
            top_k_mass = top_k_probs.sum().item()
            if top_k_mass < 0.99:
                metrics["top_k_usage"] += 1
            
            probs_filtered = torch.zeros_like(probs)
            probs_filtered[top_k_indices] = top_k_probs
            probs = probs_filtered / probs_filtered.sum()
        
        # Sample
        next_token = torch.multinomial(probs, 1).item()
        metrics["token_probs"].append(float(probs[next_token].item()))
        
        generated.append(next_token)
        unique_tokens.add(next_token)
        
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        x = torch.cat([x, next_token_t], dim=1)
        
        if next_token == eos_token_id:
            break
    
    num_generated = len(generated) - len(prompt_ids)
    metrics["diversity_score"] = len(unique_tokens) / max(num_generated, 1)
    metrics["top_k_usage"] = metrics["top_k_usage"] / max(num_generated, 1)
    
    return {
        "generated_ids": generated[len(prompt_ids):],
        "full_text": tokens_to_text(generated),
        "num_tokens": len(generated) - len(prompt_ids),
        "avg_prob": np.mean(metrics["token_probs"]) if metrics["token_probs"] else 0,
        "avg_eos_prob": np.mean(metrics["eos_probs"]) if metrics["eos_probs"] else 0,
        "avg_entropy": np.mean(metrics["entropy"]) if metrics["entropy"] else 0,
        "diversity": len(unique_tokens) / max(num_generated, 1),
        "top_k_usage_rate": metrics["top_k_usage"],
    }


# ============================================================================
# Hyperparameter Grid Search
# ============================================================================

def run_hyperparameter_search(
    model: nn.Module,
    temperatures: List[float] = None,
    top_ks: List[int] = None,
    prompts: List[List[int]] = None,
    max_new_tokens: int = 40,
) -> Dict:
    """Run grid search over temperature and top-k parameters."""
    
    if temperatures is None:
        temperatures = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    if top_ks is None:
        top_ks = [5, 10, 30, 50, 100]
    if prompts is None:
        prompts = [[1], [1, 5, 8], [10, 12, 15]]
    
    results = {
        "grid": {},
        "best_configs": {},
        "summary_stats": {},
    }
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(f"\nTesting:")
    print(f"  Temperatures: {temperatures}")
    print(f"  Top-K values: {top_ks}")
    print(f"  Prompts: {len(prompts)} samples")
    print(f"  Max tokens per generation: {max_new_tokens}")
    
    total_configs = len(temperatures) * len(top_ks)
    config_idx = 0
    
    for temp in temperatures:
        results["grid"][temp] = {}
        
        for top_k in top_ks:
            config_idx += 1
            config_key = f"temp_{temp:.1f}_topk_{top_k}"
            
            print(f"\n[{config_idx}/{total_configs}] Temperature={temp:.1f}, Top-K={top_k}")
            
            config_results = {
                "temperature": temp,
                "top_k": top_k,
                "samples": [],
                "avg_metrics": {},
            }
            
            all_probs = []
            all_eos_probs = []
            all_entropy = []
            all_diversity = []
            all_tokens = []
            
            for prompt_ids in prompts:
                result = generate_with_params(
                    model=model,
                    prompt_ids=prompt_ids,
                    temperature=temp,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )
                
                config_results["samples"].append({
                    "prompt": prompt_ids,
                    "text": result["full_text"],
                    "num_tokens": result["num_tokens"],
                    "avg_prob": float(result["avg_prob"]),
                    "avg_eos_prob": float(result["avg_eos_prob"]),
                    "avg_entropy": float(result["avg_entropy"]),
                    "diversity": float(result["diversity"]),
                })
                
                all_probs.append(result["avg_prob"])
                all_eos_probs.append(result["avg_eos_prob"])
                all_entropy.append(result["avg_entropy"])
                all_diversity.append(result["diversity"])
                all_tokens.append(result["num_tokens"])
            
            # Aggregate metrics
            config_results["avg_metrics"] = {
                "avg_prob": float(np.mean(all_probs)),
                "avg_eos_prob": float(np.mean(all_eos_probs)),
                "avg_entropy": float(np.mean(all_entropy)),
                "avg_diversity": float(np.mean(all_diversity)),
                "avg_tokens": float(np.mean(all_tokens)),
            }
            
            results["grid"][temp][top_k] = config_results
            
            print(
                f"  → Entropy={config_results['avg_metrics']['avg_entropy']:.4f}, "
                f"Diversity={config_results['avg_metrics']['avg_diversity']:.4f}, "
                f"Tokens={config_results['avg_metrics']['avg_tokens']:.1f}"
            )
    
    # Find best configurations
    print(f"\n{'─' * 80}")
    print("Finding best configurations...")
    
    # Best for diversity
    best_diversity_score = 0
    best_diversity_config = None
    for temp in temperatures:
        for top_k in top_ks:
            cfg = results["grid"][temp][top_k]
            score = cfg["avg_metrics"]["avg_diversity"]
            if score > best_diversity_score:
                best_diversity_score = score
                best_diversity_config = (temp, top_k, cfg)
    
    # Best for entropy
    best_entropy_score = 0
    best_entropy_config = None
    for temp in temperatures:
        for top_k in top_ks:
            cfg = results["grid"][temp][top_k]
            score = cfg["avg_metrics"]["avg_entropy"]
            if score > best_entropy_score:
                best_entropy_score = score
                best_entropy_config = (temp, top_k, cfg)
    
    # Best balanced (entropy + diversity)
    best_balance_score = 0
    best_balance_config = None
    for temp in temperatures:
        for top_k in top_ks:
            cfg = results["grid"][temp][top_k]
            # Normalize both to 0-1 range and combine
            normalized_entropy = min(cfg["avg_metrics"]["avg_entropy"] / 3.0, 1.0)
            normalized_diversity = cfg["avg_metrics"]["avg_diversity"]
            score = 0.5 * normalized_entropy + 0.5 * normalized_diversity
            if score > best_balance_score:
                best_balance_score = score
                best_balance_config = (temp, top_k, cfg)
    
    results["best_configs"] = {
        "best_diversity": {
            "config": {"temperature": best_diversity_config[0], "top_k": best_diversity_config[1]},
            "metrics": best_diversity_config[2]["avg_metrics"],
            "score": best_diversity_score,
        },
        "best_entropy": {
            "config": {"temperature": best_entropy_config[0], "top_k": best_entropy_config[1]},
            "metrics": best_entropy_config[2]["avg_metrics"],
            "score": best_entropy_score,
        },
        "best_balanced": {
            "config": {"temperature": best_balance_config[0], "top_k": best_balance_config[1]},
            "metrics": best_balance_config[2]["avg_metrics"],
            "score": best_balance_score,
        },
    }
    
    # Print results
    print("\n📊 Best Configurations:")
    print(f"\nBest for Diversity:")
    print(f"  Config: temp={best_diversity_config[0]:.1f}, top_k={best_diversity_config[1]}")
    print(f"  Diversity: {best_diversity_score:.4f}")
    print(f"  Entropy: {best_diversity_config[2]['avg_metrics']['avg_entropy']:.4f}")
    
    print(f"\nBest for Entropy:")
    print(f"  Config: temp={best_entropy_config[0]:.1f}, top_k={best_entropy_config[1]}")
    print(f"  Entropy: {best_entropy_score:.4f}")
    print(f"  Diversity: {best_entropy_config[2]['avg_metrics']['avg_diversity']:.4f}")
    
    print(f"\nBest Balanced:")
    print(f"  Config: temp={best_balance_config[0]:.1f}, top_k={best_balance_config[1]}")
    print(f"  Score: {best_balance_score:.4f}")
    print(f"  Entropy: {best_balance_config[2]['avg_metrics']['avg_entropy']:.4f}")
    print(f"  Diversity: {best_balance_config[2]['avg_metrics']['avg_diversity']:.4f}")
    
    return results


def main():
    print("=" * 80)
    print("HYPERPARAMETER TUNING: Temperature & Top-K Sampling")
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
    
    # Define hyperparameter ranges
    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    top_ks = [5, 10, 30, 50, 100, 200]
    
    # Test prompts
    test_prompts = [
        [1],
        [1, 5, 8],
        [10, 12, 15],
        [1, 0],
        [5, 8, 15],
    ]
    
    # Run grid search
    results = run_hyperparameter_search(
        model=model,
        temperatures=temperatures,
        top_ks=top_ks,
        prompts=test_prompts,
        max_new_tokens=40,
    )
    
    # Save results
    output_path = ROOT / "checkpoints" / "hyperparameter_tuning_results.json"
    
    # Make serializable
    serializable_results = {
        "temperatures": temperatures,
        "top_ks": top_ks,
        "best_configs": results["best_configs"],
        "sample_outputs": {},
    }
    
    # Include sample outputs for best configs
    for config_name, config_data in results["best_configs"].items():
        temp = config_data["config"]["temperature"]
        top_k = config_data["config"]["top_k"]
        cfg = results["grid"][temp][top_k]
        serializable_results["sample_outputs"][config_name] = {
            "config": config_data["config"],
            "metrics": config_data["metrics"],
            "samples": cfg["samples"][:2],  # First 2 samples
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    print(f"\nRecommendations:")
    print(f"  • For creative/diverse output: temp={results['best_configs']['best_diversity']['config']['temperature']:.1f}, top_k={results['best_configs']['best_diversity']['config']['top_k']}")
    print(f"  • For high-quality output: temp={results['best_configs']['best_entropy']['config']['temperature']:.1f}, top_k={results['best_configs']['best_entropy']['config']['top_k']}")
    print(f"  • Balanced (recommended): temp={results['best_configs']['best_balanced']['config']['temperature']:.1f}, top_k={results['best_configs']['best_balanced']['config']['top_k']}")


if __name__ == "__main__":
    main()
