# Phase 2 Development Guide (LoRA + Offline Physics Corpus)

Date: July 8, 2026

## 1) Scope

Phase 2 transitions from Phase 1 base training to **parameter-efficient fine-tuning** using LoRA.

Base checkpoint from Phase 1:
- checkpoints/production_sml_v1.pt

Tokenizer (frozen):
- subword_tokenizer/model_32k.json

Primary targets:
- Build offline corpus (train.bin / val.bin / test.bin)
- Fine-tune with LoRA adapters
- Evaluate and rank adapter checkpoints

---

## 2) New Files Added

Configuration:
- config/phase2_lora_config.yaml

Scripts:
- scripts/prepare_offline_corpus.py
- scripts/phase2_lora_finetune.py
- scripts/evaluate_lora_checkpoints.py
- scripts/run_phase2_training.sh

Data/Checkpoint folders:
- data/offline_physics/
- checkpoints/phase2_lora/

---

## 3) End-to-End Workflow

### Step A — Prepare Offline Corpus

Run:

python scripts/prepare_offline_corpus.py \
  --category physics \
  --max-papers 50000 \
  --seq-len 256 \
  --output-dir data/offline_physics \
  --tokenizer-model subword_tokenizer/model_32k.json

Outputs:
- data/offline_physics/raw_papers.jsonl
- data/offline_physics/train.bin
- data/offline_physics/val.bin
- data/offline_physics/test.bin
- data/offline_physics/corpus_stats.json

Notes:
- Uses arXiv stream + existing retry/backoff path from stream_train.py
- Stores packed token IDs as uint16

### Step B — LoRA Fine-tune

Run:

python scripts/phase2_lora_finetune.py --config config/phase2_lora_config.yaml

Behavior:
- Loads base model weights from Phase 1 checkpoint
- Injects LoRA adapters into configured linear layers
- Freezes base model params
- Trains only LoRA params
- Saves:
  - checkpoints/phase2_lora/lora_adapter_step*.pt
  - checkpoints/phase2_lora/best_lora_adapter.pt
  - checkpoints/phase2_lora/lora_adapter_final.pt
  - checkpoints/phase2_lora/phase2_train_summary.json

### Step C — Evaluate LoRA Checkpoints

Run:

python scripts/evaluate_lora_checkpoints.py \
  --config config/phase2_lora_config.yaml \
  --checkpoints-dir checkpoints/phase2_lora \
  --pattern "lora_adapter_*.pt" \
  --eval-split val \
  --eval-steps 200 \
  --output checkpoints/phase2_lora/phase2_evaluation_report.json

Output:
- checkpoints/phase2_lora/phase2_evaluation_report.json

---

## 4) One-Command Runner

scripts/run_phase2_training.sh runs all three stages in sequence:
1. Corpus preparation
2. LoRA fine-tuning
3. Adapter evaluation

---

## 5) Recommended First Dry Run (Fast)

Before full 50K papers, do a quick validation:

python scripts/prepare_offline_corpus.py \
  --category physics \
  --max-papers 300 \
  --seq-len 256 \
  --max-sequences 2000 \
  --output-dir data/offline_physics

Then run a shorter fine-tune by lowering max_steps in config/phase2_lora_config.yaml (e.g., 200).

---

## 6) Tuning Knobs

In config/phase2_lora_config.yaml:

LoRA:
- lora.r
- lora.alpha
- lora.dropout
- lora.target_modules

Training:
- training.max_steps
- training.learning_rate
- training.batch_size
- training.grad_accum_steps
- training.eval_every
- training.save_every

Data:
- corpus_prep.max_papers
- training.seq_len

---

## 7) Success Criteria

- Corpus files are generated and non-empty
- LoRA checkpoints save every configured interval
- Validation loss decreases over time
- best_lora_adapter.pt is produced
- phase2_evaluation_report.json ranks checkpoints

---

## 8) Known Constraints

- Training uses TinyLM architecture from stream_train.py (not HF GPT model)
- LoRA injection uses substring matching for target linear modules
- YAML config requires pyyaml installed in environment

---

## 9) Next After Phase 2

- Merge best adapter into base model (optional)
- Move to Phase 3: RAG integration
- Build benchmark prompts and physics QA eval set
