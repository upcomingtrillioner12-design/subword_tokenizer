



---
# Subword Tokenizer & Prototype Training Reference
**Subtitle:** Current status and architecture alignment
**Date:** July 3, 2026
**Authors:** Jaydip Singh (jaydip.singh@gmail.com), Linkan Kumbhar (upcomingtrillioner12@gmail.com)
---

## ## 1. Executive Summary

The project currently has two aligned layers:

1. **Tokenizer layer (Rust)**
   - Production BPE tokenizer implementation in `subword_tokenizer/`.
   - CLI + reusable library API for train/encode/decode/model ops.
2. **Prototype training layer (Python, laptop-first)**
   - `stream_train.py` uses the Rust tokenizer model (`model_32k.json`) and trains a small Transformer on MPS.
   - Includes checkpointing, step checkpointing, and automated evaluation scripts.

The active path is **Rust tokenizer + Python prototype training**. Legacy Rust/C++ wording is obsolete.

## ## 2. What Changed Recently

- Tokenizer usage was enforced in training (removed GPT-2/tiktoken dependency path).
- Added prototype modes suitable for Mac GPU constraints.
- Added loop automation:
  - `scripts/run_prototype_3x_and_select_best.sh`
  - `scripts/evaluate_checkpoints.py`
- Added longer-run preset:
  - `scripts/run_prototype_long_4h.sh`
- Added stream resilience in training:
  - retry/backoff for transient arXiv network timeouts.

## ## 3. Current Architecture (Aligned)

Source architecture diagram is maintained in `doc/architecture.md`.

### High-level flow

1. **Data source**
   - Live arXiv stream (primary prototype path), with retry/backoff.
2. **Tokenization**
   - Rust CLI tokenizer (`bpe-tokenizer`) using model `subword_tokenizer/model_32k.json`.
3. **Model training (prototype)**
   - Tiny Transformer in `stream_train.py` (MPS/CUDA/CPU auto device).
4. **Checkpointing & evaluation**
   - Periodic step checkpoints + end checkpoint.
   - Multi-run ranking by eval loss (`best_checkpoint_*.json`).

## ## 4. Tokenizer CLI Commands (Current)

From `subword_tokenizer/`:

- `cargo run --bin bpe-tokenizer -- train <corpus.txt> <vocab_size>`
- `cargo run --bin bpe-tokenizer -- expand <corpus.txt> <vocab_size>`
- `cargo run --bin bpe-tokenizer -- tokenize <text>`
- `cargo run --bin bpe-tokenizer -- decode <comma_separated_ids>`
- `cargo run --bin bpe-tokenizer -- count`
- `cargo run --bin bpe-tokenizer -- prepare <input.txt> --train <train.bin> --val <val.bin> --test <test.bin>`

Notes:

- Active runtime model file for CLI is `model.json`.
- Training scripts activate the desired model by copying from `model_32k.json` to `model.json` before tokenization.

## ## 5. Prototype Training Commands (Current)

From repository root:

- Quick prototype: `./scripts/run_prototype.sh`
- 3-run compare and auto-select: `./scripts/run_prototype_3x_and_select_best.sh`
- Long run (~3–4h target on M3 Pro): `./scripts/run_prototype_long_4h.sh`
- Evaluate existing checkpoints: `./scripts/run_evaluation.sh`

## ## 6. Validation Status

- Rust tokenizer tests are passing (`src/tests.rs`).
- Prototype training on Mac MPS is validated.
- Best-checkpoint reports are being generated under `checkpoints/`.
- Long-run path now includes network timeout retry/backoff and step checkpoint safety.

## ## 7. Current Risks / Observations

1. **Live-stream dependency risk**: arXiv network timeouts can still pause long runs (now retried).
2. **Artifact growth**: checkpoints and corpora can grow quickly.
3. **Documentation drift risk**: ensure markdown-to-PDF sync is part of workflow.

## ## 8. Recommended Next Actions

1. Add offline dataset training mode (train from local predownloaded corpus).
2. Add resume-from-checkpoint support for interrupted long runs.
3. Add CI/doc task to regenerate this PDF from markdown automatically.
4. Track large artifacts through retention/LFS policy.

## 






---

## 16. Project Roadmap: Physics Research Assistant SLM

### 16.1 Final Goal

Build a Physics Research Assistant using a 3B-7B parameter SLM, fine-tuned with LoRA on physics papers, combined with RAG and tool-using agents.

### 16.2 Current Status

**Completed:**
- Subword tokenizer (32K vocab, BPE)
- CLI for tokenization/detokenization
- Tokenizer validation (32K vs 50K trade-off analysis)
- Prototype training pipeline on Mac MPS
- Checkpointing and evaluation automation

**Not Started:**
- SLM model training pipeline (3B-7B parameters)
- LoRA fine-tuning setup
- RAG (Retrieval-Augmented Generation) integration
- Tool-using agent framework
- Physics paper dataset preparation at scale

### 16.3 Phase 1: Environment & Data Pipeline (Week 1-2)

**Framework:** Python 3.10+, PyTorch (with MPS), Hugging Face Transformers, Datasets, Accelerate, Pandas, NumPy

**Steps:**
1. Activate venv: source /Users/jdsingh/slm_v0/venv/bin/activate
2. Install packages: pip install torch transformers datasets accelerate peft
3. Verify MPS support for Mac
4. Download 10K-50K physics papers from arXiv
5. Organize into: /Users/jdsingh/slm_v0/data/physics_papers/raw/
6. Extract abstracts, introductions, methodologies
7. Remove noise, duplicates, non-English text
8. Normalize whitespace, split into chunks (max 4K words)
9. Tokenize using Rust tokenizer (32K vocab, frozen)
10. Split into train/val/test (98%/1%/1%), pack into 512-token sequences

**Expected Outcome:** 300M-1B tokens ready for SLM training.

### 16.4 Phase 2: Base SLM Training (Week 3-5)

**Framework:** PyTorch, Hugging Face Transformers, Accelerate

| Parameter | 3B Model | 7B Model |
|-----------|----------|----------|
| Architecture | GPT2 or LLaMA-style decoder-only | GPT2 or LLaMA-style decoder-only |
| Hidden size | 2048 | 3072 |
| Layers | 24 | 32 |
| Attention heads | 16 | 24 |
| Max sequence length | 512 | 512 |
| Vocab size | 32,000 | 32,000 |
| Batch size | 32 | 32 |
| Gradient accumulation | 4 | 4 |
| Learning rate | 5e-4 with cosine schedule | 5e-4 with cosine schedule |
| Total steps | 100,000 | 100,000 |

**Success Criteria:**
- Training loss decreases smoothly
- Validation perplexity < 20
- Generated samples coherent
- Peak memory < 16GB (Mac)

**Expected Outcome:** Production-ready base SLM (3B-7B parameters).

### 16.5 Phase 3: LoRA Fine-tuning (Week 6-7)

**Framework:** PEFT (Parameter-Efficient Fine-Tuning)

| Parameter | Value |
|-----------|-------|
| Rank | 32 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Trainable parameters | ~1-2% of base model |

**Training:**
- Batch size: 16
- Learning rate: 2e-4
- Epochs: 3-5
- Output: LoRA weights (~50-100MB)

**Expected Outcome:** Fine-tuned SLM ready for RAG integration.

### 16.6 Phase 4: RAG Integration (Week 8-9)

**Framework:** FAISS, Sentence-Transformers, LangChain

**Steps:**
1. Generate embeddings for all physics paper chunks
2. Build FAISS vector index (384 or 768 dimensions)
3. Create retrieval function: query -> top-k relevant chunks
4. Augment prompt with retrieved context
5. Generate answer using fine-tuned SLM

**Expected Outcome:** RAG system with physics-grounded responses.

### 16.7 Phase 5: Tool-Using Agents (Week 10-11)

**Framework:** LangChain, LanGraph (ReAct framework)

| Tool | Purpose |
|------|---------|
| search_arxiv | Query arXiv for physics papers |
| solve_equation | Solve math equations (SymPy) |
| execute_python | Safe code execution for calculations |

**Agent Loop:**
1. Generate thought + action
2. Execute appropriate tool
3. Observe result
4. Iterate until final answer
5. Max steps: 10-15

**Expected Outcome:** Multi-step reasoning with tool integration.

### 16.8 Phase 6: Deployment (Week 12+)

**Framework:** FastAPI, Docker

| Endpoint | Method | Description |
|----------|--------|-------------|
| /query | POST | Submit physics question |
| /status | GET | Model health check |
| /upload_papers | POST | Add new physics papers to RAG |

**UI (Optional):** Streamlit web interface

**Deployment:** Docker container on cloud (AWS, GCP, Azure, Hugging Face Spaces)

**Expected Outcome:** Production-ready Physics Research Assistant.

### 16.9 Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Data Pipeline | 300M-1B tokenized tokens |
| 3-5 | Base SLM | 3B-7B model checkpoint |
| 6-7 | LoRA | 50MB LoRA weights |
| 8-9 | RAG | Vector index + retrieval |
| 10-11 | Agents | Multi-step reasoning demo |
| 12+ | Deployment | API + UI live |

### 16.10 Hardware Requirements

| Level | Specs |
|-------|-------|
| Minimum (M1/M2 Mac) | 16GB RAM, 100GB SSD |
| Recommended | GPU with 24GB+ VRAM, 32-64GB RAM, 500GB NVMe SSD |
| Training (AWS p4d.24xlarge) | 8x A100, 320GB total VRAM, spot instances |

### 16.11 Success Criteria

- Generates coherent physics-grounded answers
- Uses tools appropriately
- Cites retrieved papers
- Handles multi-step reasoning
- API response time < 5s
- Deployment-ready

---\n\n## 17. File-Level Reference

- `subword_tokenizer/src/lib.rs`: tokenizer core and model operations
- `subword_tokenizer/src/main.rs`: CLI dispatch (`bpe-tokenizer`)
- `stream_train.py`: prototype streaming training loop
- `scripts/evaluate_checkpoints.py`: checkpoint ranking
- `scripts/run_prototype_3x_and_select_best.sh`: train+select workflow
- `scripts/run_prototype_long_4h.sh`: long prototype preset

---


## Architecture Diagram

![System Architecture](architecture.png)

*Figure 1: Live arXiv Stream → Rust Tokenizer → TinyLM Training → Checkpoint Evaluation*

```
Live arXiv Stream → stream_train.py → Retry/Backoff → Rust Tokenizer CLI → Token IDs → TinyLM Training → Checkpoints → Evaluation → best_checkpoint_*.json
```

Prepared on July 3, 2026
Updated July 5, 2026 — Added references and architecture diagram for IIT Bombay server request for end-of-day status handoff.


## References

[1] P. Jhandi, O. Kazi, S. Subramanian, N. Sendas. "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning." arXiv preprint arXiv:2512.15943, 2025.

[2] Y. Kang, et al. "Fine-tuning Small Language Models as Efficient Enterprise Search Relevance Labelers." arXiv preprint arXiv:2601.03211, 2026.

[3] R. Sharma, M. Mehta. "Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs." arXiv preprint arXiv:2510.03847, 2025.

[4] Z. K. Chong, et al. "Compiling Deterministic Structure into SLM Harnesses." arXiv preprint arXiv:2604.17450, 2026.

[5] ServiceNow, SLB. "Scaling AI Development with DGX Cloud: ServiceNow and SLB Production Deployments." Nvidia DGX Cloud Case Study, ZenML LLMOps Database, 2025.

[6] N. Patience. "Leveraging Small Language Models for Enterprise AI: Benefits, Use Cases, and IBM's Approach." The Futurum Group, in partnership with IBM, 2025.

[7] M. Elfeki, R. Liu, C. Voegele. "Return of the Encoder: Maximizing Parameter Efficiency for Small Language Models." arXiv preprint arXiv:2501.16273, 2025.

[8] SACAI R. "Small Language Models for Enterprise Edge Deployment." Southern African Conference on Artificial Intelligence Research, 2025.

[9] "Data-Centric Fine-Tuning of Small Language Models for Industrial Applications." IEEE Transactions on Industrial Informatics, 2025.

[10] LinkedIn Engineering. "Production-Scale SLM Ranking: Optimizations for Inference." arXiv preprint arXiv:2510.22101, 2025.

[11] A. Karpathy. "NanoGPT: The simplest, fastest repository for training medium-sized GPTs." GitHub, 2023.

[12] R. Eldan, Y. Li. "TinyStories: How Small Can Language Models Be and Still Speak Coherently?" arXiv preprint arXiv:2305.07759, 2023.

