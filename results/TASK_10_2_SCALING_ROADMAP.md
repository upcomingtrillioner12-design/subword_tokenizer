# Task 10.2: Model Scaling & Fine-Tuning Roadmap

**Date:** July 15, 2026  
**Phase:** Phase 4, Task 10 (Model Scaling experiments)  
**Status:** 🔄 IN PROGRESS - Planning phase complete, ready for implementation

---

## Overview

Task 10 has two sub-components:

1. **Task 10.1: Uncertainty Calibration** ✅ COMPLETE
   - Implemented 4-component calibration framework
   - 33.8% mean improvement (0.662 → 0.438)
   - 71.1% discrimination boost (std dev 0.073)
   - Committed to main branch (commit 2e09ef5)

2. **Task 10.2: Model Scaling & Fine-Tuning** 🔄 IN PROGRESS
   - Three parallel experiments designed
   - Configs created for 7B baseline
   - Ready for implementation

---

## Task 10.2: Three-Phase Scaling Strategy

### Phase 10.2a: Larger Base Models (7B, 13B)

**Hypothesis:** Current 1.3B model constrained by parameter budget. Larger models should:
- Better reasoning on complex multi-hop questions
- More nuanced entailment understanding
- Improved context grounding

**Experiments:**
1. **Mistral-7B-Instruct-v0.2** (7B parameters, quantized int8)
   - Expected accuracy: 40% (+5pp vs baseline 35%)
   - Estimated latency: +30-50% (larger model)
   - Config: `config/phase4_task10_2a_mistral7b.yaml`

2. **Llama-2-13B-chat** (13B parameters, quantized int8)
   - Expected accuracy: 45% (+10pp vs baseline 35%)
   - Estimated latency: +80-100% (much larger model)
   - Config: (to be created after 7B results)

**Evaluation Methodology:**
- Benchmark: 20-question adversarial subset (same as Task 9)
- Metrics: MC accuracy, calibrated uncertainty, entailment, faithfulness, inference time
- Success criteria: accuracy >= 40% for 7B, >= 45% for 13B
- If failed: Investigate calibration issues or routing problems

**Timeline:**
- Download & quantization: 30 minutes
- Benchmark run (20Q): 15 minutes
- Analysis & decision: 15 minutes
- **Total Phase 10.2a: ~1 hour**

### Phase 10.2b: Cross-Encoder Fine-Tuning

**Hypothesis:** Pre-trained MS Marco cross-encoder not optimized for STEM+adversarial domain. Fine-tuning on domain-specific preferences should:
- Improve ranking of relevant context
- Boost retrieval precision (top-1, top-3)
- Indirectly improve confidence calibration

**Data Collection:**
- Source: 60-question STEM benchmark
- Labels: Binary relevance (1.0 if contains answer info, 0.0 otherwise)
- Processing: Top-5 retrieved docs × 60 questions = 300 training pairs
- Train/val split: 240/60 (80/20)

**Fine-Tuning Approach:**
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (original)
- Layers: Fine-tune last 3 layers only (avoid catastrophic forgetting)
- Loss: Binary cross-entropy on relevance scores
- Learning rate: 2e-5 (conservative)
- Batch size: 8
- Epochs: 3-5 (stop if validation plateaus)
- Hardware: GPU (single-pass fine-tuning)

**Evaluation:**
- Before fine-tuning: Baseline retrieval precision
- After fine-tuning: Measure on held-out validation set
- Success criteria: +5-10pp improvement in precision@1
- Test on 20-question adversarial subset: expected +2-3pp accuracy

**Timeline:**
- Data collection: 30 minutes
- Fine-tuning: 20-30 minutes (small dataset)
- Evaluation: 15 minutes
- **Total Phase 10.2b: ~1.5 hours**

### Phase 10.2c: Iterative Retrieval Loop

**Hypothesis:** Many adversarial questions are multi-hop. Single-pass retrieval + generation insufficient. Iterative refinement should:
- Generate partial answer with confidence
- If uncertain (calibrated_uncertainty > 0.6), extract entities & re-retrieve
- Generate final answer with combined context

**Architecture:**
```
Query
  ↓
[Iteration 1]
  - Retrieve context (RRF fusion)
  - Rerank (cross-encoder)
  - Generate partial answer + confidence
  - If confidence OK (< 0.6): RETURN
  - Else: Extract entities
  ↓
[Iteration 2 (optional)]
  - Formulate follow-up query (e.g., "What is X?" → "Who invented X?")
  - Retrieve new context
  - Rerank
  - Generate final answer
  - RETURN
```

**Confidence-Based Loop Termination:**
- After iteration 1: If calibrated_uncertainty < 0.6 → confident enough → STOP
- After iteration 2: Always return (max 2 iterations to prevent infinite loops)
- Rationale: Calibrated uncertainty as decision signal for confidence

**Expected Impact:**
- Multi-hop questions: +15-20pp improvement
- Single-hop questions: No change (already handled in iteration 1)
- Overall: +7-10pp on mixed 40-question adversarial set

**Evaluation:**
- Test on 10 explicitly multi-hop adversarial questions
- Metrics: Accuracy, iteration count distribution, confidence trajectory
- Success criteria: Multi-hop accuracy >= 60% (vs single-pass ~30%)

**Timeline:**
- Implement loop: 45 minutes
- Test & debug: 30 minutes
- Benchmark: 20 minutes
- **Total Phase 10.2c: ~2 hours**

---

## Expected Cumulative Results

| Phase | Component | Accuracy | Calibrated Unc | Notes |
|-------|-----------|----------|---|-------|
| Baseline (Task 9) | 1.3B + MS Marco + single-pass | 35% | 0.438 ± 0.073 | Established |
| 10.2a | +7B model | 40% | ~0.43 | Model-independent calibration |
| 10.2b | +Fine-tuned cross-encoder | 42-43% | ~0.42 | Better context selection |
| 10.2c | +Iterative retrieval | 50% | ~0.45 | Multi-hop breakthrough |

**Target Achieved:** 50% accuracy on adversarial set (43% relative improvement over baseline)

---

## Risk Mitigation

### Memory Constraints
- **Risk:** 7B/13B quantization still too large for 16GB GPU
- **Mitigation:** Use 4-bit quantization (bitsandbytes) if int8 fails
- **Fallback:** Stick with 1.3B, investigate calibration improvements instead

### Fine-Tuning Risks
- **Risk:** Overfitting on small 300-pair dataset
- **Mitigation:** Layer freezing, early stopping on validation set, low learning rate
- **Fallback:** Skip Phase 10.2b if no improvement observed

### Iterative Loop Risks
- **Risk:** Error accumulation (wrong partial answer leads astray)
- **Mitigation:** High confidence threshold (0.6) for continuing, max 2 iterations
- **Fallback:** Disable loop, stay with single-pass if performance degraded

### Latency Risks
- **Risk:** Larger models + iterative loops make inference too slow
- **Mitigation:** Profile latency at each phase, track acceptable budget
- **Acceptable budget:** < 5 seconds per question (from single-pass ~1-2s)

---

## Implementation Checklist

### Phase 10.2a: 7B Baseline
- [ ] Download Mistral-7B-Instruct-v0.2 to local checkpoint dir
- [ ] Implement quantization wrapper in `inference_lora.py` (int8, device_map='auto')
- [ ] Create config: `config/phase4_task10_2a_mistral7b.yaml` ✓
- [ ] Run benchmark: 20-question adversarial set
- [ ] Analyze: Accuracy, calibrated uncertainty, latency
- [ ] Decision: Proceed to 13B or investigate?

### Phase 10.2b: Fine-Tuning
- [ ] Collect 300 preference pairs from STEM benchmark
- [ ] Create fine-tuning script: `scripts/finetune_cross_encoder.py`
- [ ] Fine-tune MS Marco model on preference data
- [ ] Evaluate: Validation precision improvement
- [ ] Integrate into eval pipeline
- [ ] Benchmark: 20-question set with fine-tuned reranker

### Phase 10.2c: Iterative Retrieval
- [ ] Design entity extraction module (simple regex + NER)
- [ ] Implement iterative loop in `hybrid_retrieval.py`
- [ ] Create loop termination logic (calibrated_uncertainty > 0.6)
- [ ] Test on 10 multi-hop adversarial questions
- [ ] Benchmark: 20-question set with iterative retrieval
- [ ] Commit final version

---

## Commits to Make

1. **Task 10.1 Complete** ✅
   - Commit: `2e09ef5` (uncertainty calibration framework)
   - Status: Pushed to main

2. **Task 10.2 Planning** (This session)
   - Files: `config/phase4_task10_2a_mistral7b.yaml`, scaling plan docs
   - Commit: (ready for push)

3. **Task 10.2a Results** (Next session)
   - Files: Config results JSON, analysis markdown
   - Commit: (after 7B benchmark)

4. **Task 10.2b Results** (Following session)
   - Files: Fine-tuned model checkpoint, validation results
   - Commit: (after fine-tuning complete)

5. **Task 10.2c Results** (Final session)
   - Files: Iterative loop implementation, full benchmark results
   - Commit: (when 50% target achieved)

---

## Next Immediate Action

1. Attempt to load and benchmark Mistral-7B-Instruct-v0.2
   - Command: `python scripts/run_rag_generation_evaluation.py --config config/phase4_task10_2a_mistral7b.yaml --limit 2`
   - Smoke test: 2 questions to verify setup
   - If successful: Full 20-question run

2. If 7B available:
   - Continue to Phase 10.2b (fine-tuning design)

3. If 7B unavailable (memory/download):
   - Fall back to Phase 10.2b immediately (fine-tuning can work with 1.3B)
   - Revisit 7B later with adjusted quantization

---

**Report Generated:** July 15, 2026 20:05 UTC  
**Status:** Ready for Phase 10.2a implementation
