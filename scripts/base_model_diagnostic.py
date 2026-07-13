#!/usr/bin/env python3
"""
Base Model Diagnostic: Test Phase 1 vs Phase 2 generation

Purpose: Isolate whether early-EOS issue is from:
1. Phase 1 training (base model itself)
2. LoRA injection (adapter application)
3. Model architecture (TinyLM design)

Approach:
- Load Phase 1 model WITHOUT LoRA
- Load Phase 1 model WITH LoRA (Phase 2)
- Run identical prompts on both
- Compare token generation counts and patterns
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np

# ============================================================================
# Path setup
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
SLMV0_ROOT = ROOT.parent
if not SLMV0_ROOT.name.endswith("slm_v0"):
    SLMV0_ROOT = ROOT

sys.path.insert(0, str(SLMV0_ROOT))
sys.path.insert(0, str(ROOT))

from scripts.inference_lora import LoRAInferenceEngine, load_tokenizer


# ============================================================================
# Configuration
# ============================================================================

TEST_PROMPTS = [
    "In quantum mechanics, the wave function describes the",
    "The uncertainty principle states that there is a fundamental limit to the precision with which",
    "The Born rule provides a probabilistic interpretation:",
    "Entropy in thermodynamics represents the",
    "In special relativity, time dilation occurs because",
]


# ============================================================================
# Diagnostic Functions
# ============================================================================

def run_diagnostic_test(
    engine: LoRAInferenceEngine,
    tokenizer,
    prompts: List[str],
    model_label: str,
    device: str = "mps",
) -> Dict:
    """Test a model configuration and return diagnostics."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_label}")
    print(f"{'='*70}")
    
    results = {
        "model": model_label,
        "prompts_tested": len(prompts),
        "token_counts": [],
        "logits_analysis": [],
        "samples": [],
    }
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}]")
        print(f"  Text: {prompt[:60]}...")
        
        try:
            # Generate with verbose output
            generated_text, metrics = engine.generate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
            
            token_count = metrics.generated_tokens
            results["token_counts"].append(token_count)
            
            print(f"  Tokens generated: {token_count}")
            print(f"  Time: {metrics.elapsed_seconds:.3f}s")
            
            # Store first 2 samples
            if i < 2:
                results["samples"].append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "tokens": token_count,
                })
            
            # Analyze logits for EOS probability
            try:
                # Manually check next token logits
                import torch.nn.functional as F
                
                # Encode prompt
                input_ids = tokenizer.encode_text(prompt, max_length=256)
                if not input_ids:
                    input_ids = [1]
                
                input_ids_tensor = torch.tensor(
                    input_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
                
                # Forward pass
                with torch.no_grad():
                    logits = engine.model(input_ids_tensor)
                    next_logits = logits[0, -1, :]
                    probs = F.softmax(next_logits, dim=-1)
                
                # Top tokens and EOS probability
                top_values, top_indices = torch.topk(probs, k=5)
                eos_prob = probs[2].item()  # EOS token = 2
                
                logits_info = {
                    "eos_token_prob": eos_prob,
                    "top_5_tokens": [int(idx) for idx in top_indices],
                    "top_5_probs": [float(val) for val in top_values],
                }
                
                results["logits_analysis"].append(logits_info)
                
                print(f"  EOS (token 2) probability: {eos_prob:.4f}")
                print(f"  Top 5 tokens: {top_indices.tolist()} → {top_values.tolist()}")
                
            except Exception as e:
                print(f"  Logits analysis error: {e}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results["token_counts"].append(0)
    
    # Summary statistics
    if results["token_counts"]:
        results["avg_tokens"] = float(np.mean(results["token_counts"]))
        results["max_tokens"] = int(np.max(results["token_counts"]))
        results["min_tokens"] = int(np.min(results["token_counts"]))
    else:
        results["avg_tokens"] = 0.0
        results["max_tokens"] = 0
        results["min_tokens"] = 0
    
    return results


def main():
    """Main diagnostic entry point."""
    print("=" * 70)
    print("BASE MODEL DIAGNOSTIC: Phase 1 vs Phase 2 Generation")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Locate checkpoints
    base_model_path = SLMV0_ROOT / "checkpoints" / "prototype_long4h_epoch1.pt"
    lora_adapter_path = SLMV0_ROOT / "checkpoints" / "phase2_lora" / "best_lora_adapter.pt"
    tokenizer_path = ROOT / "model_32k.json"
    
    print(f"\nCheckpoints:")
    print(f"  Base model: {base_model_path} (exists: {base_model_path.exists()})")
    print(f"  LoRA adapter: {lora_adapter_path} (exists: {lora_adapter_path.exists()})")
    print(f"  Tokenizer: {tokenizer_path} (exists: {tokenizer_path.exists()})")
    
    if not all([base_model_path.exists(), tokenizer_path.exists()]):
        print("\n✗ ERROR: Required files not found!")
        sys.exit(1)
    
    # Load tokenizer once
    print(f"\nLoading tokenizer...")
    try:
        tokenizer = load_tokenizer(tokenizer_path)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    all_results = {}
    
    # ========================================================================
    # Test 1: Base Model Only (No LoRA)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: PHASE 1 BASE MODEL (No LoRA)")
    print("=" * 70)
    
    try:
        engine_base = LoRAInferenceEngine(
            base_checkpoint=base_model_path,
            lora_checkpoint=None,  # No LoRA
            device=device,
            verbose=True,
        )
        print("✓ Phase 1 model loaded (without LoRA)")
        
        results_base = run_diagnostic_test(
            engine_base,
            tokenizer,
            TEST_PROMPTS,
            "Phase 1 Base Model (No LoRA)",
            device=device,
        )
        all_results["phase1_base"] = results_base
        
        print(f"\n[SUMMARY] Phase 1 Base Model:")
        print(f"  Avg tokens: {results_base['avg_tokens']:.1f}")
        print(f"  Max tokens: {results_base['max_tokens']}")
        print(f"  Min tokens: {results_base['min_tokens']}")
        
    except Exception as e:
        print(f"✗ Failed to load Phase 1 model: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 2: Phase 2 With LoRA (if available)
    # ========================================================================
    if lora_adapter_path.exists():
        print("\n" + "=" * 70)
        print("TEST 2: PHASE 2 WITH LoRA ADAPTER")
        print("=" * 70)
        
        try:
            engine_lora = LoRAInferenceEngine(
                base_checkpoint=base_model_path,
                lora_checkpoint=lora_adapter_path,
                device=device,
                verbose=True,
            )
            print("✓ Phase 2 model loaded (with LoRA)")
            
            results_lora = run_diagnostic_test(
                engine_lora,
                tokenizer,
                TEST_PROMPTS,
                "Phase 2 With LoRA",
                device=device,
            )
            all_results["phase2_lora"] = results_lora
            
            print(f"\n[SUMMARY] Phase 2 With LoRA:")
            print(f"  Avg tokens: {results_lora['avg_tokens']:.1f}")
            print(f"  Max tokens: {results_lora['max_tokens']}")
            print(f"  Min tokens: {results_lora['min_tokens']}")
        
        except Exception as e:
            print(f"✗ Failed to load Phase 2 model: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Comparison & Diagnosis
    # ========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    if len(all_results) >= 1:
        base_result = all_results.get("phase1_base")
        lora_result = all_results.get("phase2_lora")
        
        print("\n[Results Comparison]")
        if base_result:
            print(f"Phase 1 Base:  avg={base_result['avg_tokens']:.1f} tokens, max={base_result['max_tokens']}")
        if lora_result:
            print(f"Phase 2 LoRA:  avg={lora_result['avg_tokens']:.1f} tokens, max={lora_result['max_tokens']}")
        
        print("\n[Diagnosis]")
        
        if base_result and base_result["avg_tokens"] > 0:
            print("✓ Phase 1 model GENERATES tokens successfully")
            if lora_result and lora_result["avg_tokens"] == 0:
                print("✗ Phase 2 LoRA BREAKS generation")
                print("\n→ ROOT CAUSE: LoRA injection is corrupting the model")
                print("  Action: Debug _load_lora_adapter() in inference_lora.py")
            elif lora_result and lora_result["avg_tokens"] < base_result["avg_tokens"]:
                print("⚠ Phase 2 LoRA REDUCES token generation")
                print("\n→ ROOT CAUSE: LoRA is interfering with generation (possibly improving on training task)")
                print("  Action: Check LoRA rank, alpha, and fine-tuning config")
            else:
                print("✓ Phase 2 LoRA maintains/improves generation")
        
        elif base_result and base_result["avg_tokens"] == 0:
            print("✗ Phase 1 model does NOT generate tokens")
            print("\n→ ROOT CAUSE: Phase 1 training or model architecture issue")
            print("  Action options:")
            print("    1. Check Phase 1 training logs for loss convergence")
            print("    2. Inspect TinyLM architecture for softmax issues")
            print("    3. Consider retraining Phase 1 with generation loss")
        
        # EOS probability analysis
        if base_result and base_result.get("logits_analysis"):
            print("\n[EOS Token Analysis]")
            eos_probs = [la["eos_token_prob"] for la in base_result["logits_analysis"] if la]
            if eos_probs:
                print(f"Phase 1 EOS probs: mean={np.mean(eos_probs):.4f}, max={np.max(eos_probs):.4f}")
                if np.mean(eos_probs) > 0.5:
                    print("  → EOS token has >50% probability (explains 0-token outputs)")
    
    # Save results
    results_path = ROOT / "results" / "base_model_diagnostic.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Diagnostic results saved to: {results_path}")
    print("\n" + "=" * 70)
    print("BASE MODEL DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
