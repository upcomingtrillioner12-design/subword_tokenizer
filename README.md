# Physics Research Assistant SLM

A physics-domain Small Language Model built end-to-end on Apple Silicon:
custom Rust BPE tokenizer, base model training, LoRA fine-tuning,
retrieval-augmented generation (BM25 + dense + hybrid).

## Paper
Full write-up (arXiv-style): [paper/physics_slm_paper.pdf](paper/physics_slm_paper.pdf)
(LaTeX source: [paper/physics_slm_paper.tex](paper/physics_slm_paper.tex))

## Results Summary
| Phase | Milestone | Key Result |
|---|---|---|
| 1 | Tokenizer + data pipeline | 32K BPE vocab, frozen |
| 2 | LoRA fine-tuning | 0.0107 to 0.0060 eval loss (44% down) |
| 3 | Inference + sampling profiles | 1.21-1.39x LoRA speedup |
| 4 | RAG (BM25 + dense + hybrid) | Recall@5: 75% to 87.5% (hybrid) |

## Authors
Jaydip Singh, Linkan Kumbhar
