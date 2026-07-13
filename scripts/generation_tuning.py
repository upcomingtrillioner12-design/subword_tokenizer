#!/usr/bin/env python3
"""
Phase 4 Task 1: Generation-control tuning script

Purpose: Systematically explore decoding parameters to address early EOS behavior
and enable meaningful text generation for BLEU and qualitative evaluation.

Approach:
1. Load Phase 2 LoRA adapter + base model
2. Test different configurations: max_length, temperature, top_p, top_k, etc.
3. Evaluate on subset of prompts (qualitative eval set)
4. Report token counts, generation length distributions, and sample outputs
5. Recommend best parameters for Phase 4 evaluation tasks
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import numpy as np

# ============================================================================
# Path setup (handle nested repo layout)
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
SLMV0_ROOT = ROOT.parent
if not SLMV0_ROOT.name.endswith("slm_v0"):
    SLMV0_ROOT = ROOT

sys.path.insert(0, str(SLMV0_ROOT))
sys.path.insert(0, str(ROOT))

# Import inference engine
from scripts.inference_lora import LoRAInferenceEngine, InferenceMetrics

# Import stream_train for tokenization
import stream_train

# Import tokenizer loading from inference_lora
from scripts.inference_lora import load_tokenizer

# ============================================================================
# Configuration
# ============================================================================

# Decoding parameter configurations to test
# Note: min_length/length_penalty not directly supported by current model
# Focus on: max_tokens, temperature, top_k, top_p
PARAM_CONFIGS = {
    "baseline": {
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    },
    "high_temp": {
        "max_tokens": 100,
        "temperature": 1.5,
        "top_p": 1.0,
        "top_k": 50,
    },
    "low_temp": {
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 50,
    },
    "nucleus_0_9": {
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": None,
    },
    "nucleus_0_95": {
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": None,
    },
    "top_k_30": {
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 30,
    },
    "long_output": {
        "max_tokens": 200,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    },
    "creative": {
        "max_tokens": 150,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 40,
    },
}

# ============================================================================
# Utilities
# ============================================================================


def load_eval_prompts(num_samples: int = 5) -> List[str]:
    """Load a subset of qualitative eval prompts."""
    eval_data_path = ROOT / "data" / "qualitative_eval_subset.json"
    
    if eval_data_path.exists():
        with open(eval_data_path) as f:
            data = json.load(f)
        
        # Extract prompts from the correct structure
        if isinstance(data, dict) and "prompts" in data:
            prompts = [item.get("text", "") for item in data["prompts"][:num_samples]]
        elif isinstance(data, list):
            prompts = [item.get("text", item) if isinstance(item, dict) else str(item) for item in data[:num_samples]]
        else:
            # Fallback
            prompts = []
        
        if prompts:
            print(f"Loaded {len(prompts)} prompts from {eval_data_path}")
            return prompts
    
    # Fallback: use generic physics prompts
    print(f"⚠ {eval_data_path} not found or empty, using fallback prompts")
    return [
        "The quantum mechanical principle that governs the behavior of electrons in atoms is",
        "In special relativity, time dilation is explained by",
        "The relationship between energy and mass in Einstein's equation E=mc² demonstrates",
        "Entropy in thermodynamics describes the",
        "The fundamental force that binds quarks together is",
    ][:num_samples]


def evaluate_generation_config(
    engine: LoRAInferenceEngine,
    tokenizer,
    prompts: List[str],
    config_name: str,
    config: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Test a specific decoding configuration on a set of prompts.
    
    Returns:
        dict with:
        - config_name, config dict
        - token_counts, avg_tokens, max_tokens, min_tokens
        - generation_samples (first 2 prompts)
    """
    results = {
        "config_name": config_name,
        "config": config,
        "prompts_tested": len(prompts),
        "token_counts": [],
        "generation_samples": [],
    }
    
    for i, prompt in enumerate(prompts):
        try:
            # Generate with this config
            generated_text, metrics = engine.generate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_tokens=config.get("max_tokens", 100),
                temperature=config.get("temperature", 1.0),
                top_p=config.get("top_p", 1.0),
                top_k=config.get("top_k", 50),
            )
            
            # Count tokens in generation
            token_count = metrics.generated_tokens
            results["token_counts"].append(token_count)
            
            # Store first 2 samples for inspection
            if i < 2:
                results["generation_samples"].append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "generation": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                    "token_count": token_count,
                    "elapsed_sec": metrics.elapsed_seconds,
                })
            
            print(f"  Prompt {i+1}/{len(prompts)}: {token_count} tokens in {metrics.elapsed_seconds:.3f}s")
            
        except Exception as e:
            print(f"  Prompt {i+1}/{len(prompts)}: ERROR - {e}")
            results["token_counts"].append(0)
    
    # Compute statistics
    if results["token_counts"]:
        results["avg_tokens"] = float(np.mean(results["token_counts"]))
        results["max_tokens"] = int(np.max(results["token_counts"]))
        results["min_tokens"] = int(np.min(results["token_counts"]))
        results["std_tokens"] = float(np.std(results["token_counts"]))
    else:
        results["avg_tokens"] = 0.0
        results["max_tokens"] = 0
        results["min_tokens"] = 0
        results["std_tokens"] = 0.0
    
    return results


def main():
    """Main entry point."""
    print("=" * 80)
    print("Phase 4 Task 1: Generation-Control Tuning")
    print("=" * 80)
    
    # Device setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Locate model and adapter files
    base_model_path = SLMV0_ROOT / "checkpoints" / "prototype_long4h_epoch1.pt"
    adapter_path = SLMV0_ROOT / "checkpoints" / "phase2_lora" / "best_lora_adapter.pt"
    tokenizer_path = ROOT / "model_32k.json"
    
    # Fallback paths
    if not base_model_path.exists():
        base_model_path = SLMV0_ROOT / "checkpoints" / "production_sml_v1.pt"
    
    print(f"\nPaths:")
    print(f"  Base model: {base_model_path} (exists: {base_model_path.exists()})")
    print(f"  Adapter: {adapter_path} (exists: {adapter_path.exists()})")
    print(f"  Tokenizer: {tokenizer_path} (exists: {tokenizer_path.exists()})")
    
    if not all([base_model_path.exists(), adapter_path.exists(), tokenizer_path.exists()]):
        print("\n✗ ERROR: Not all required files found!")
        sys.exit(1)
    
    # Load model and tokenizer
    print("\nLoading Phase 2 model + LoRA adapter...")
    try:
        engine = LoRAInferenceEngine(
            base_checkpoint=base_model_path,
            lora_checkpoint=adapter_path,
            device=device,
            verbose=True,
        )
        print("✓ Model and adapter loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    try:
        tok = load_tokenizer(tokenizer_path)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Load evaluation prompts
    print("\nLoading evaluation prompts...")
    prompts = load_eval_prompts(num_samples=5)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Test each configuration
    print("\n" + "=" * 80)
    print("Testing decoding configurations...")
    print("=" * 80)
    
    all_results = []
    
    for config_name, config in PARAM_CONFIGS.items():
        print(f"\n[{config_name.upper()}]")
        print(f"  Config: {config}")
        
        results = evaluate_generation_config(
            engine, tok, prompts, config_name, config, device=device
        )
        all_results.append(results)
        
        # Print summary
        print(f"  ✓ Results:")
        print(f"    - Avg tokens: {results['avg_tokens']:.1f}")
        print(f"    - Max tokens: {results['max_tokens']}")
        print(f"    - Min tokens: {results['min_tokens']}")
        print(f"    - Std dev: {results['std_tokens']:.2f}")
    
    # Save results to JSON
    results_path = ROOT / "results" / "generation_tuning_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best config by avg tokens
    sorted_results = sorted(all_results, key=lambda x: x["avg_tokens"], reverse=True)
    
    print(f"\nResults saved to: {results_path}")
    print("\nRanked by average token generation:")
    for i, r in enumerate(sorted_results, 1):
        print(
            f"{i}. {r['config_name']:15} → avg={r['avg_tokens']:6.1f} "
            f"(max={r['max_tokens']:3}, min={r['min_tokens']:3}, std={r['std_tokens']:5.2f})"
        )
    
    best = sorted_results[0]
    if best["avg_tokens"] >= 10:
        print(f"\n✓ RECOMMENDATION: Use '{best['config_name']}' config")
        print(f"  Achieves {best['avg_tokens']:.1f} tokens/prompt on average")
        print(f"  Config: {best['config']}")
    else:
        print("\n⚠ WARNING: All configs still show low token generation")
        print("  Consider:")
        print("    1. Checking model weights (possible corruption)")
        print("    2. Examining tokenizer output (padding/truncation)")
        print("    3. Investigating early-EOS token IDs in model")
    
    # Print sample generations
    print("\nSample generations from best config:")
    for sample in best["generation_samples"]:
        print(f"\n  Prompt: {sample['prompt']}")
        print(f"  Generation ({sample['token_count']} tokens): {sample['generation']}")
    
    print("\n" + "=" * 80)
    print("Phase 4 Task 1 complete. Ready for Task 2 (RAG setup).")
    print("=" * 80)


if __name__ == "__main__":
    main()
