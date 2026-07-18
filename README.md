# Physics Research Assistant SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Paper](https://github.com/upcomingtrillioner12-design/subword_tokenizer/actions/workflows/build-paper.yml/badge.svg)](https://github.com/upcomingtrillioner12-design/subword_tokenizer/actions/workflows/build-paper.yml)

A physics-domain Small Language Model built end-to-end on Apple Silicon:
custom Rust BPE tokenizer, base model training, LoRA fine-tuning,
retrieval-augmented generation (BM25 + dense + hybrid).

## Paper

Full write-up (arXiv-style): [paper/physics_slm_paper.pdf](paper/physics_slm_paper.pdf)
LaTeX source: [paper/physics_slm_paper.tex](paper/physics_slm_paper.tex)
arXiv-ready submission bundle: [paper/arxiv_submission.zip](paper/arxiv_submission.zip)

### Citation
See [CITATION.cff](CITATION.cff), or cite directly:

```bibtex
@misc{singh2026physicsslm,
  title  = {A Physics-Specialized Small Language Model: Tokenizer Design,
            LoRA Fine-Tuning, and Retrieval-Augmented Generation on
            Consumer Hardware},
  author = {Singh, Jaydip and Kumbhar, Linkan},
  year   = {2026},
  url    = {https://github.com/upcomingtrillioner12-design/subword_tokenizer}
}
```

## Results Summary

| Phase | Milestone | Key Result |
|---|---|---|
| 1 | Tokenizer + data pipeline | 32K BPE vocab, frozen |
| 2 | LoRA fine-tuning | 0.0107 to 0.0060 eval loss (44% down) |
| 3 | Inference + sampling profiles | 1.21-1.39x LoRA speedup |
| 4 | RAG (BM25 + dense + hybrid) | Recall@5: 75% to 87.5% (hybrid) |

## Repository Structure

```
scripts/        Training, evaluation, and retrieval scripts
config/         YAML configs for LoRA, RAG, inference
data/           Corpus metadata and stats (large binaries excluded)
checkpoints/    Model checkpoint metadata (weights excluded, see .gitignore)
results/        Evaluation reports (JSON/Markdown)
subword_tokenizer/  Rust BPE tokenizer
paper/          Paper source, PDF, Makefile, arXiv submission bundle
.github/        CI workflow (auto-rebuilds the paper PDF)
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/phase2_lora_finetune.py --config config/phase2_lora_config.yaml --resume auto
python scripts/inference_lora.py --adapter checkpoints/phase2_lora/lora_adapter_step9000.pt
```

## Building the Paper

```bash
cd paper
make            # compile PDF
make arxiv      # build arXiv submission zip
make clean      # remove build artifacts
```

## License
MIT — see [LICENSE](LICENSE).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Authors
Jaydip Singh, Linkan Kumbhar
