# PhysRAG: Physics Research Assistant SLM — Complete arXiv Paper

## 📄 Paper
- **Title**: Iterative Retrieval-Augmented Generation with Learned Reranking
- **Authors**: Jaydip Singh, Linkan Kumbhar
- **Status**: Submission-ready

## 🏗️ Directory Structure
```
.
├── main.tex                    # Complete LaTeX source
├── build.sh                    # Build script (PDF generation)
├── README.md                   # This file
├── data/
│   └── checkpoint_data.json    # Real LoRA checkpoint data
├── figures/                    # Generated publication figures
│   ├── figure1_loss_curve.png/pdf
│   ├── figure2_parameter_efficiency.png/pdf
│   ├── figure3_pipeline.png/pdf
│   ├── figure4_ablation.png/pdf
│   └── figure5_convergence.png/pdf
└── scripts/
    └── generate_figures.py     # Figure generator (real data)
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full texlive-bibtex-extra texlive-science
sudo apt-get install python3 python3-pip
pip install matplotlib numpy

# macOS
brew install --cask mactex
pip install matplotlib numpy
```

### Build Paper
```bash
# Generate figures + compile PDF
bash build.sh

# Or step by step:
python3 scripts/generate_figures.py
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📊 Real Data Incorporated
- ✅ LoRA checkpoint data (11 checkpoints, steps 0-10,000)
- ✅ Best checkpoint: step 9,000, loss = 0.0060005
- ✅ 44.0% improvement over baseline
- ✅ 99.81% parameter efficiency (65,536 / 35,200,000)

## 📝 Paper Sections
1. Introduction
2. Related Work (5 subsections)
3. Tokenizer Architecture
4. Corpus and Pre-Training
5. LoRA Fine-Tuning (with real checkpoint analysis)
6. Iterative RAG System
7. Ablation Study
8. Evaluation (with multi-seed significance)
9. Operational Infrastructure
10. Cross-Encoder Reranker Training
11. Future Work
12. Conclusion
- Appendix: Complete checkpoint data, configs, reproducibility checklist

## 🔗 GitHub Repository
All code, data, and models available at:
https://github.com/jdsingh/physics-research-assistant-slm

## 📧 Contact
- Jaydip Singh: jaydip.singh@gmail.com
- Linkan Kumbhar: upcomingtrillioner12@gmail.com
