# Phase 3: Inference & Evaluation Plan

**Status:** 🚧 **IN PROGRESS (Tasks 1-4 Complete)**  
**Date:** July 13, 2026  
**Duration:** 2 weeks (estimated)  
**Objective:** Evaluate Phase 2 LoRA adapter quality, generate completions, and benchmark against Phase 1 baseline

---

## Execution Progress (Live)

- [x] Task 1 complete: [scripts/inference_lora.py](scripts/inference_lora.py)
- [x] Task 2 complete: [data/eval_prompts.json](data/eval_prompts.json)
- [x] Task 3 complete: [scripts/benchmark_inference.py](scripts/benchmark_inference.py)
- [x] Task 4 complete: [scripts/qualitative_eval.py](scripts/qualitative_eval.py)
- [ ] Task 5 pending: test set evaluation
- [ ] Task 6 pending: perplexity / BLEU metrics
- [ ] Task 7 pending: physics QA evaluation

Current outputs:
- [results/phase3_benchmark_results.json](results/phase3_benchmark_results.json)
- [results/phase3_qualitative_outputs.json](results/phase3_qualitative_outputs.json)
- [results/phase3_qualitative_assessment.md](results/phase3_qualitative_assessment.md)

---

## Executive Summary

Phase 3 focuses on **production inference validation** of the best Phase 2 checkpoint (`lora_adapter_step9000.pt`). We will:

1. Build inference pipeline with streaming output
2. Generate physics completions on diverse prompts
3. Benchmark quality vs Phase 1 (quantitative + qualitative)
4. Assess domain-specific knowledge and reasoning
5. Document findings for Phase 4 (RAG integration)

**Success Criteria:**
- ✅ Inference pipeline runs at <200ms per 256-token generation
- ✅ Generated text is coherent and physics-relevant
- ✅ Domain knowledge assessment shows improvement over Phase 1
- ✅ Identified 5+ use cases for Phase 4 RAG integration
- ✅ All evaluation artifacts documented and committed to Git

---

## Part 1: Architecture & Components

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3 INFERENCE SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  │   Tokenizer  │      │  Base Model  │      │LoRA Adapter  │
│  │  (32K vocab) │─────▶│(35.2M params)│─────▶│(0.19% params)│
│  └──────────────┘      └──────────────┘      └──────────────┘
│         │                      │                      │
│         └──────────────────────┼──────────────────────┘
│                                │
│                    ┌───────────▼────────────┐
│                    │  Inference Engine      │
│                    │ (generate + beam search)
│                    └───────────┬────────────┘
│                                │
│         ┌──────────────────────┼──────────────────────┐
│         │                      │                      │
│    ┌────▼─────┐         ┌──────▼──────┐        ┌─────▼────┐
│    │ Prompt   │         │  Streaming  │        │Evaluation│
│    │Generator │         │Output Queue │        │Metrics   │
│    └──────────┘         └─────────────┘        └──────────┘
│
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Inference Engine** — Forward pass through model + LoRA adapter
2. **Generation Methods** — Greedy, beam search, nucleus sampling
3. **Prompt Manager** — Curated physics prompts (5 categories)
4. **Evaluation Harness** — Quantitative metrics + qualitative assessment
5. **Output Logger** — Store results for analysis

---

## Part 2: Detailed Task Breakdown

### Task 1: Build Inference Pipeline (Days 1-2)

**File:** `scripts/inference_lora.py`

**Functionality:**
```python
class LoRAInference:
    def __init__(self, base_model_path, adapter_path, tokenizer_path):
        # Load base model
        # Load LoRA adapter (PEFT)
        # Load tokenizer
        # Move to device (MPS)
        pass
    
    def generate(self, prompt, max_length=256, temperature=0.7, 
                 top_p=0.9, num_beams=1):
        # Tokenize prompt
        # Forward pass with LoRA
        # Decode output
        # Return text
        pass
    
    def generate_batch(self, prompts, **kwargs):
        # Parallel generation for multiple prompts
        pass
    
    def get_generation_stats(self):
        # Time per token, tokens/second
        pass
```

**Inputs:**
- Base model checkpoint: `checkpoints/production_sml_v1.pt`
- LoRA adapter: `checkpoints/phase2_lora/lora_adapter_step9000.pt`
- Tokenizer: `subword_tokenizer/model_32k.json`

**Outputs:**
- Inference log with timings
- Generated text samples (JSON)
- Generation statistics

**Success Criteria:**
- Generates 256 tokens in <200ms (M3 Pro)
- No NaN/overflow errors
- Output is valid tokenizable text

---

### Task 2: Create Prompt Suite (Days 1-3)

**File:** `data/eval_prompts.json`

**5 Physics Categories with 10 prompts each (50 total):**

```json
{
  "quantum_mechanics": [
    "The Schrödinger equation describes...",
    "Quantum entanglement is a phenomenon where...",
    "The uncertainty principle states that...",
    ...
  ],
  "relativity": [
    "Einstein's theory of general relativity explains...",
    "Spacetime curvature is caused by...",
    "Black holes form when...",
    ...
  ],
  "thermodynamics": [
    "The second law of thermodynamics states...",
    "Entropy increases because...",
    "Heat flows from hot to cold due to...",
    ...
  ],
  "electromagnetism": [
    "Maxwell's equations describe...",
    "Electric fields exert forces on...",
    "Electromagnetic waves propagate...",
    ...
  ],
  "particle_physics": [
    "The Standard Model includes...",
    "Quarks are fundamental particles that...",
    "The Higgs boson was discovered...",
    ...
  ]
}
```

**Design:**
- Partial sentences (150-200 tokens prompt length)
- Physics-relevant but open-ended
- Mix of conceptual, computational, and descriptive
- Difficulty: beginner → expert levels

---

### Task 3: Inference Benchmark Suite (Days 3-4)

**File:** `scripts/benchmark_inference.py`

**Benchmark:**
```
For each prompt:
├─ Phase 1 (baseline model only)
├─ Phase 2 (best LoRA adapter) 
└─ Comparison metrics

Metrics collected:
├─ Generation time (ms/token)
├─ Output length (tokens)
├─ Perplexity (on test set)
├─ BLEU-4 score (vs expected physics)
└─ Coherence score (manual 1-5)
```

**Output:** `results/inference_benchmark.json`

**Execution:**
```bash
python scripts/benchmark_inference.py \
    --phase1-model checkpoints/production_sml_v1.pt \
    --phase2-adapter checkpoints/phase2_lora/lora_adapter_step9000.pt \
    --prompts data/eval_prompts.json \
    --num-samples 50 \
    --output results/inference_benchmark.json
```

---

### Task 4: Qualitative Evaluation (Days 4-5)

**File:** `data/qualitative_assessment.md`

**Evaluation Framework:**
```
For each generated completion:

1. Correctness
   ├─ Physics accuracy (0-5 scale)
   ├─ Terminology usage (0-5)
   └─ Logical consistency (0-5)

2. Relevance
   ├─ Stays on topic (yes/no)
   ├─ Answers the prompt (partial/yes/no)
   └─ Depth of answer (shallow/medium/deep)

3. Quality
   ├─ Writing clarity (0-5)
   ├─ Structure & organization (0-5)
   └─ Grammar & spelling (0-5)

4. Domain Knowledge
   ├─ Uses correct formulas (count)
   ├─ Mentions relevant concepts (count)
   ├─ Cites laws/principles (count)
   └─ Shows understanding (yes/no)
```

**Sample Evaluation (5 generations per category):**
- Total samples: 25 (5 categories × 5 samples)
- Evaluator: Domain expert review
- Output: Scores + qualitative notes

---

### Task 5: Test Set Evaluation (Day 5)

**File:** `scripts/eval_test_set.py`

**Methodology:**
- Use Phase 2's held-out test split (3,456 docs)
- Compute test loss with LoRA adapter loaded
- Compare against Phase 1 baseline test loss

**Metrics:**
```
Test Loss Comparison:
├─ Phase 1 baseline: 0.0107 (from checkpoint)
└─ Phase 2 LoRA:     ? (to be measured)

Expected: <0.0065 (based on val_loss=0.0060 correlation)
```

**Output:** `results/test_set_evaluation.json`

---

### Task 6: Perplexity & BLEU Analysis (Day 6)

**File:** `scripts/compute_language_metrics.py`

**Metrics:**
```
1. Perplexity (on test set)
   - Lower = better language modeling
   - Expected improvement: 20-30%

2. BLEU-4 Score (vs reference physics text)
   - Measure n-gram overlap with high-quality physics papers
   - Reference: 500 papers from test corpus
   - Expected improvement: 15-25%

3. Token-level accuracy
   - Measure if model predicts correct next tokens
   - Compare Phase 1 vs Phase 2
```

**Output:** `results/language_metrics.json`

---

### Task 7: Domain Knowledge Quiz (Day 7)

**File:** `data/physics_qa_dataset.json`

**Purpose:** Test if model learned physics-specific knowledge

**Format:**
```json
[
  {
    "question": "What is the de Broglie wavelength formula?",
    "expected_answer": "λ = h/p",
    "context": "wave-particle duality"
  },
  {
    "question": "State the second law of thermodynamics",
    "expected_answer": "entropy of an isolated system increases",
    "context": "thermodynamics"
  },
  ...
]
```

**Dataset:** 20 QA pairs (4 per physics category)

**Scoring:**
- Exact match: 1.0
- Semantic match: 0.5
- No match: 0.0

**Expected Performance:**
- Phase 1: ~30% correct (random baseline ~25%)
- Phase 2: ~55-65% correct (physics-specialized LoRA)

---

## Part 3: Development Timeline

### Week 1 (Days 1-5)
```
Mon (Day 1-2): Task 1 - Inference pipeline
                Task 2 - Prompt suite (parallel)

Tue (Day 3):   Task 3 - Benchmark suite
                Task 2 - Prompts finalization

Wed (Day 4-5): Task 4 - Qualitative evaluation
                Task 5 - Test set evaluation
```

### Week 2 (Days 6-10)
```
Thu (Day 6):   Task 6 - Perplexity & BLEU
                Task 7 - Physics QA quiz

Fri (Day 7-8): Analysis & documentation
                Results compilation

Mon-Tue (D9-10): Phase 4 preparation
                 RAG system design
```

---

## Part 4: Success Criteria & Deliverables

### Quantitative Targets

| Metric | Phase 1 | Phase 2 Target | Success? |
|--------|---------|----------------|----------|
| Test Loss | 0.0107 | <0.0065 | If ✓ |
| Perplexity | ~85 | <70 | If ✓ |
| BLEU-4 | ~0.15 | >0.18 | If ✓ |
| Physics QA Accuracy | ~30% | >55% | If ✓ |
| Gen Speed | <200ms | <200ms | If ✓ |

### Qualitative Targets

- [ ] Generated text is grammatically correct (95%+ samples)
- [ ] Physics terminology used correctly (90%+ mentions)
- [ ] Logical consistency maintained (85%+ samples)
- [ ] Stays on-topic throughout (90%+ samples)
- [ ] Shows domain-specific knowledge (70%+ samples)

### Deliverables

1. **Inference Pipeline** — Production-ready generation engine
2. **Benchmark Report** — Quantitative comparison (Phase 1 vs 2)
3. **Qualitative Assessment** — Domain expert evaluation notes
4. **Results Dashboard** — JSON artifacts + visualization
5. **Analysis Document** — Key findings & insights
6. **Phase 4 Recommendations** — RAG design based on Phase 3 insights

---

## Part 5: Phase 3 Scripts to Create

### Priority 1 (Essential)
1. `scripts/inference_lora.py` — Main inference engine
2. `scripts/benchmark_inference.py` — Comparison benchmark
3. `scripts/eval_test_set.py` — Test set evaluation

### Priority 2 (Important)
4. `scripts/compute_language_metrics.py` — Perplexity/BLEU
5. `scripts/physics_qa_eval.py` — Domain knowledge quiz
6. `scripts/generate_completions.py` — Batch generation

### Priority 3 (Nice to have)
7. `scripts/visualize_results.py` — Charts & plots
8. `scripts/compare_phase1_phase2.py` — Side-by-side comparison

---

## Part 6: Data & Configuration

### New Directories
```
data/
├── eval_prompts.json              ← Prompt suite (50 prompts)
├── physics_qa_dataset.json        ← QA pairs (20 samples)
├── qualitative_assessment.md      ← Expert eval notes

results/
├── inference_benchmark.json       ← Timing & metrics
├── test_set_evaluation.json       ← Loss comparison
├── language_metrics.json          ← Perplexity/BLEU
├── physics_qa_results.json        ← QA scores
└── generation_samples.jsonl       ← Generated completions
```

### Configuration Updates
```yaml
# config/phase3_inference_config.yaml
inference:
  base_model: checkpoints/production_sml_v1.pt
  adapter_phase2: checkpoints/phase2_lora/lora_adapter_step9000.pt
  tokenizer: subword_tokenizer/model_32k.json
  
generation:
  max_length: 256
  temperature: 0.7
  top_p: 0.9
  num_beams: 1
  
evaluation:
  test_batch_size: 32
  num_inference_samples: 50
  qualitative_samples: 25
```

---

## Part 7: Expected Outcomes & Phase 4 Preparation

### Key Findings We Expect

1. **LoRA improves physics reasoning** by 30-40%
2. **Generation quality is high** (>0.18 BLEU-4)
3. **Domain knowledge retained** (~60% QA accuracy)
4. **Inference is fast enough** for real-time use (<200ms)

### Phase 4 Implications (RAG Integration)

Based on Phase 3 findings, Phase 4 will:
- Build vector retrieval system using best checkpoint embeddings
- Implement retrieval-augmented generation (RAG) pipeline
- Evaluate QA performance with retrieved context
- Expected QA accuracy improvement: 60% → 75-80% (with RAG)

### Knowledge Base for RAG

- Use Phase 2 corpus (34,464 papers) as retrieval source
- Generate embeddings via best adapter's hidden layers
- Build FAISS/Weaviate index for fast retrieval
- Benchmark: "Retrieve top-K papers, then generate answer"

---

## Part 8: Risk Mitigation

### Potential Issues & Solutions

| Risk | Mitigation |
|------|------------|
| Slow inference (>200ms) | Optimize batch size, use MPS quantization |
| Poor generation quality | Check tokenizer alignment, try sampling |
| OOM during batch eval | Reduce batch_size, eval in chunks |
| Evaluation metrics are low | Compare vs Phase 1 — may still be +improvement |
| Domain knowledge limited | This is expected; Phase 4 RAG will help |

---

## Getting Started Checklist

- [ ] Review this plan document
- [ ] Set up Phase 3 directory structure
- [ ] Create evaluation prompts dataset
- [ ] Code inference pipeline
- [ ] Run first end-to-end test
- [ ] Benchmark Phase 1 vs Phase 2
- [ ] Document findings

---

## Next Steps (Immediate Actions)

**If you approve this plan:**

1. ✅ **Confirm direction** — Any adjustments to the plan?
2. 🔜 **Task 1 start** — Build `inference_lora.py` (2 hours)
3. 🔜 **Task 2 start** — Create `eval_prompts.json` (1 hour)
4. 🔜 **First end-to-end test** — Run generation on 1 prompt (30 min)

---

## Questions for Review

1. Should we focus on speed (inference latency) or quality (better metrics)?
2. Any specific physics domains to prioritize in prompts?
3. Do you want human expert evaluation or automated metrics only?
4. Should Phase 3 include fine-tuning adjustments or just evaluation?

---

**Ready to start Phase 3? Let me know if you'd like adjustments to this plan, then we'll begin development!**
