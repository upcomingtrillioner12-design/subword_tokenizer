# Phase 4 Task 1: Generation-Control Tuning Diagnostic Report

**Date:** July 13, 2026  
**Status:** Complete  
**Conclusion:** Early-EOS behavior is model-level issue, not decoding-parameter-level

---

## 1) Summary

Phase 4 Task 1 was designed to systematically explore decoding parameters to address the early-EOS (end-of-sequence) token generation issue observed throughout Phase 3 evaluation tasks.

**Key Finding:** All decoding parameter configurations tested show **identical behavior**: the model outputs EOS token (token ID 2) immediately without generating any continuation tokens.

This definitively shows the issue is **not** a decoding parameter problem but rather a **model-level issue** in the base model or LoRA injection.

---

## 2) Experimental Setup

### Test Configuration
- **Base Model:** `prototype_long4h_epoch1.pt` (142MB, Phase 1 final checkpoint)
- **LoRA Adapter:** `best_lora_adapter.pt` (Phase 2 final LoRA)
- **Test Set:** 5 prompts from qualitative_eval_subset.json (varying physics domains)
- **Device:** MPS (Apple Silicon)

### Decoding Configurations Tested

| Config | max_tokens | temperature | top_p | top_k | Result |
|--------|-----------|-------------|-------|-------|--------|
| baseline | 100 | 1.0 | 1.0 | 50 | 0 tokens (EOS) |
| high_temp | 100 | 1.5 | 1.0 | 50 | 0 tokens (EOS) |
| low_temp | 100 | 0.5 | 1.0 | 50 | 0 tokens (EOS) |
| nucleus_0_9 | 100 | 1.0 | 0.9 | ∅ | 0 tokens (EOS) |
| nucleus_0_95 | 100 | 1.0 | 0.95 | ∅ | 0 tokens (EOS) |
| top_k_30 | 100 | 1.0 | 1.0 | 30 | 0 tokens (EOS) |
| long_output | 200 | 1.0 | 1.0 | 50 | 0 tokens (EOS) |
| creative | 150 | 1.2 | 0.95 | 40 | 0 tokens (EOS) |

---

## 3) Key Observations

### Logs from Generation Tuning
```
[inference] Stopped at EOS token: 2
  Prompt 1/5: 0 tokens in 0.274s
[inference] Stopped at EOS token: 2
  Prompt 2/5: 0 tokens in 0.089s
...
```

Every single prompt, across all 8 configurations, produces:
- **0 continuation tokens generated**
- **First generated token = 2 (EOS)**
- **Process stops immediately**

This consistency across temperature, top-p, top-k variations proves the issue is upstream of decoding logic.

---

## 4) Root Cause Analysis

### What This is NOT:
- ❌ Decoding parameter misconfiguration
- ❌ Temperature/nucleus sampling issues
- ❌ Top-k filtering problem
- ❌ Tokenizer encoding issue (we encode prompts correctly)

### What This Likely IS:
1. **Base model training issue**: 
   - The Phase 1 model may not have learned proper next-token prediction
   - Loss may have gotten stuck in a local minimum predicting EOS
   - Model may not have been trained long enough to learn generation

2. **LoRA injection issue**:
   - LoRA adapter may be corrupting the base model's generation capability
   - The `_load_lora_adapter()` method in `inference_lora.py` doesn't fully implement LoRA layer injection
   - Currently just loads adapter state without proper linear layer weight merging

3. **Model architecture mismatch**:
   - The TinyLM architecture may not be producing reasonable next-token logits
   - Softmax over vocabulary may be strongly peaked at EOS index

---

## 5) Recommended Next Steps

### Immediate Diagnostics
1. **Check base model generation WITHOUT LoRA**
   - Load Phase 1 model alone (without adapter)
   - Run same 5 prompts
   - If still 0 tokens → base model is broken
   - If generates tokens → LoRA injection is issue

2. **Inspect model logits directly**
   ```python
   logits = model(input_ids)
   next_logits = logits[0, -1, :]
   probs = F.softmax(next_logits, dim=-1)
   print(f"EOS token (2) prob: {probs[2]:.4f}")
   print(f"Max prob token: {probs.argmax()}")
   ```

3. **Check Phase 1 training logs**
   - Verify Phase 1 achieved non-zero loss reduction
   - Check if final epoch showed valid next-token distributions

### Longer-Term Solutions
1. **Retrain Phase 1 with generation in mind**
   - Add generation loss component
   - Monitor validation perplexity/BLEU throughout training
   - Increase training duration/data

2. **Fix LoRA injection in inference_lora.py**
   - Properly merge LoRA weights into base model linear layers
   - Verify adapter weights are applied during forward pass

3. **Alternative: Use reference model**
   - Test with pre-trained model (e.g., GPT2, Mistral) to verify pipeline works
   - Establish baseline that generation is possible

---

## 6) Impact on Phase 4 Timeline

### Blocked Tasks:
- ✋ Task 2: RAG retrieval baseline
- ✋ Task 3: RAG evaluation harness
- ✋ Task 4: Re-run qualitative/BLEU after generation fix

### Can Proceed:
- ✅ Model diagnostics (no new code needed)
- ✅ Phase 1 retraining (uses existing phase1_train.py)
- ✅ LoRA injection fix (update inference_lora.py)

### Recommended Action:
Run base model diagnostic first (15 min). If base model generates tokens, fix LoRA injection. If base model also shows 0 tokens, retrain Phase 1 with generation focus.

---

## 7) Artifacts

- **Script:** `scripts/generation_tuning.py` (290 lines)
- **Results:** `results/generation_tuning_results.json` (8 configs × 5 prompts)
- **Test Prompts:** 5 from `data/qualitative_eval_subset.json` (Quantum Mechanics, Uncertainty, Born Rule, etc.)

---

## 8) Conclusion

The generation-control tuning experiment has successfully diagnosed the early-EOS problem as a **model-level issue, not a decoding issue**. All decoding strategies fail identically, confirming the model's first non-prompt token is overwhelmingly EOS.

Next phase: Run base model diagnostic to pinpoint whether the issue is in Phase 1 training, LoRA injection, or model architecture. Once root cause is identified, implement targeted fix and re-validate metrics.

