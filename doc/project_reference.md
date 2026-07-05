## Architecture Diagram

![System Architecture Flow](architecture.png)
*Figure 1: Live arXiv Stream → Rust Tokenizer → TinyLM Training → Checkpoint Evaluation*



---
title: Subword Tokenizer & Prototype Training Reference
subtitle: Current status and architecture alignment
date: July 3, 2026
---

## 1) Executive Summary

The project currently has two aligned layers:

1. **Tokenizer layer (Rust)**
   - Production BPE tokenizer implementation in `subword_tokenizer/`.
   - CLI + reusable library API for train/encode/decode/model ops.
2. **Prototype training layer (Python, laptop-first)**
   - `stream_train.py` uses the Rust tokenizer model (`model_32k.json`) and trains a small Transformer on MPS.
   - Includes checkpointing, step checkpointing, and automated evaluation scripts.

The active path is **Rust tokenizer + Python prototype training**. Legacy Rust/C++ wording is obsolete.

## 2) What Changed Recently

- Tokenizer usage was enforced in training (removed GPT-2/tiktoken dependency path).
- Added prototype modes suitable for Mac GPU constraints.
- Added loop automation:
  - `scripts/run_prototype_3x_and_select_best.sh`
  - `scripts/evaluate_checkpoints.py`
- Added longer-run preset:
  - `scripts/run_prototype_long_4h.sh`
- Added stream resilience in training:
  - retry/backoff for transient arXiv network timeouts.

## 3) Current Architecture (Aligned)

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

## 4) Tokenizer CLI Commands (Current)

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

## 5) Prototype Training Commands (Current)

From repository root:

- Quick prototype: `./scripts/run_prototype.sh`
- 3-run compare and auto-select: `./scripts/run_prototype_3x_and_select_best.sh`
- Long run (~3–4h target on M3 Pro): `./scripts/run_prototype_long_4h.sh`
- Evaluate existing checkpoints: `./scripts/run_evaluation.sh`

## 6) Validation Status

- Rust tokenizer tests are passing (`src/tests.rs`).
- Prototype training on Mac MPS is validated.
- Best-checkpoint reports are being generated under `checkpoints/`.
- Long-run path now includes network timeout retry/backoff and step checkpoint safety.

## 7) Current Risks / Observations

1. **Live-stream dependency risk**: arXiv network timeouts can still pause long runs (now retried).
2. **Artifact growth**: checkpoints and corpora can grow quickly.
3. **Documentation drift risk**: ensure markdown-to-PDF sync is part of workflow.

## 8) Recommended Next Actions

1. Add offline dataset training mode (train from local predownloaded corpus).
2. Add resume-from-checkpoint support for interrupted long runs.
3. Add CI/doc task to regenerate this PDF from markdown automatically.
4. Track large artifacts through retention/LFS policy.

## 

## Architecture Diagram

![System Architecture](architecture.png)

*Figure 1: Live arXiv Stream → Rust Tokenizer → TinyLM Training → Checkpoint Evaluation*

```
Live arXiv Stream → stream_train.py → Retry/Backoff → Rust Tokenizer CLI → Token IDs → TinyLM Training → Checkpoints → Evaluation → best_checkpoint_*.json
```



## References

[1] P. Jhandi, O. Kazi, S. Subramanian, N. Sendas. "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning." arXiv preprint arXiv:2512.15943, 2025.

[2] Y. Kang, et al. "Fine-tuning Small Language Models as Efficient Enterprise Search Relevance Labelers." arXiv preprint arXiv:2601.03211, 2026.

[3] R. Sharma, M. Mehta. "Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs." arXiv preprint arXiv:2510.03847, 2025.

[4] Z. K. Chong, et al. "Compiling Deterministic Structure into SLM Harnesses." arXiv preprint arXiv:2604.17450, 2026.

[5] ServiceNow, SLB. "Scaling AI Development with DGX Cloud: ServiceNow and SLB Production Deployments." Nvidia DGX Cloud Case Study, ZenML LLMOps Database, 2025.

[6] N. Patience. "Leveraging Small Language Models for Enterprise AI: Benefits, Use Cases, and IBM Approach." The Futurum Group, in partnership with IBM, 2025.

[7] M. Elfeki, R. Liu, C. Voegele. "Return of the Encoder: Maximizing Parameter Efficiency for Small Language Models." arXiv preprint arXiv:2501.16273, 2025.

[8] SACAI R. "Small Language Models for Enterprise Edge Deployment." Southern African Conference on Artificial Intelligence Research, 2025.

[9] "Data-Centric Fine-Tuning of Small Language Models for Industrial Applications." IEEE Transactions on Industrial Informatics, 2025.

[10] LinkedIn Engineering. "Production-Scale SLM Ranking: Optimizations for Inference." arXiv preprint arXiv:2510.22101, 2025.

[11] A. Karpathy. "NanoGPT: The simplest, fastest repository for training medium-sized GPTs." GitHub, 2023.

[12] R. Eldan, Y. Li. "TinyStories: How Small Can Language Models Be and Still Speak Coherently?" arXiv preprint arXiv:2305.07759, 2023.

9) File-Level Reference

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
