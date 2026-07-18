#!/bin/bash
set -u

PROJECT_DIR="/workspaces/subword_tokenizer"
REPO_URL="https://github.com/upcomingtrillioner12-design/subword_tokenizer.git"
GH_USER="upcomingtrillioner12-design"
REPO_NAME="subword_tokenizer"

log()  { echo -e "\n▶ $1"; }
ok()   { echo "✅ $1"; }
fail() { echo "❌ $1"; }

cd "$PROJECT_DIR" || { fail "Cannot cd into $PROJECT_DIR"; exit 1; }

if [ ! -f "paper/physics_slm_paper.tex" ]; then
  fail "paper/physics_slm_paper.tex not found. Run publish_paper.sh first."
  exit 1
fi
ok "Found existing paper/physics_slm_paper.tex"

# ============================================
# 1. Makefile for reproducible LaTeX builds
# ============================================
log "Writing paper/Makefile"
cat > paper/Makefile << 'MAKE_EOF'
# Makefile for building the arXiv-style paper
TEX=physics_slm_paper
PDFLATEX=pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean arxiv

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	$(PDFLATEX) $(TEX).tex
	$(PDFLATEX) $(TEX).tex
	$(PDFLATEX) $(TEX).tex
	@$(MAKE) clean-aux

clean-aux:
	rm -f $(TEX).aux $(TEX).out $(TEX).toc $(TEX).bbl $(TEX).blg $(TEX).log

clean: clean-aux
	rm -f $(TEX).pdf

# Builds the arXiv submission bundle: tex source + any figures, zipped
arxiv: $(TEX).pdf
	rm -rf arxiv_submission
	mkdir -p arxiv_submission
	cp $(TEX).tex arxiv_submission/
	cp $(TEX).pdf arxiv_submission/
	@if ls *.png *.jpg *.eps >/dev/null 2>&1; then cp *.png *.jpg *.eps arxiv_submission/ 2>/dev/null || true; fi
	cd arxiv_submission && zip -r ../arxiv_submission.zip . -x ".*"
	@echo "arXiv submission bundle ready: paper/arxiv_submission.zip"
MAKE_EOF
ok "Makefile written"

# ============================================
# 2. CITATION.cff (GitHub's native citation format)
# ============================================
log "Writing CITATION.cff"
cat > CITATION.cff << 'CITE_EOF'
cff-version: 1.2.0
message: "If you use this software or paper, please cite it as below."
title: "A Physics-Specialized Small Language Model: Tokenizer Design, LoRA Fine-Tuning, and Retrieval-Augmented Generation on Consumer Hardware"
authors:
  - family-names: Singh
    given-names: Jaydip
  - family-names: Kumbhar
    given-names: Linkan
date-released: 2026-07-15
url: "https://github.com/upcomingtrillioner12-design/subword_tokenizer"
repository-code: "https://github.com/upcomingtrillioner12-design/subword_tokenizer"
license: MIT
keywords:
  - small language models
  - LoRA
  - retrieval-augmented generation
  - BM25
  - physics NLP
CITE_EOF
ok "CITATION.cff written"

# ============================================
# 3. LICENSE (MIT — adjust if you want something else)
# ============================================
log "Writing LICENSE"
YEAR=$(date +%Y)
cat > LICENSE << LICENSE_EOF
MIT License

Copyright (c) $YEAR Jaydip Singh, Linkan Kumbhar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSE_EOF
ok "LICENSE written"

# ============================================
# 4. CONTRIBUTING.md
# ============================================
log "Writing CONTRIBUTING.md"
cat > CONTRIBUTING.md << 'CONTRIB_EOF'
# Contributing

Thanks for your interest in this project.

## Reporting issues
Open a GitHub Issue with a clear description, reproduction steps, and
environment details (OS, Python version, PyTorch version, device).

## Development setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Rebuilding the paper
```bash
cd paper
make            # builds physics_slm_paper.pdf
make arxiv      # builds the arXiv submission zip
make clean      # removes build artifacts
```

## Pull requests
1. Fork the repo and create a feature branch.
2. Keep commits focused and messages descriptive.
3. Ensure the paper still compiles (`make` in `paper/`) if you touch `.tex` files.
4. Open a PR against `main`.
CONTRIB_EOF
ok "CONTRIBUTING.md written"

# ============================================
# 5. GitHub Actions CI — auto-rebuild PDF on push
# ============================================
log "Writing .github/workflows/build-paper.yml"
mkdir -p .github/workflows
cat > .github/workflows/build-paper.yml << 'CI_EOF'
name: Build arXiv Paper

on:
  push:
    paths:
      - 'paper/**.tex'
      - '.github/workflows/build-paper.yml'
  pull_request:
    paths:
      - 'paper/**.tex'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install TeX Live
        run: |
          sudo apt-get update
          sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

      - name: Build PDF
        working-directory: paper
        run: make

      - name: Upload PDF artifact
        uses: actions/upload-artifact@v4
        with:
          name: physics_slm_paper
          path: paper/physics_slm_paper.pdf

      - name: Commit compiled PDF back to repo
        if: github.event_name == 'push'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add paper/physics_slm_paper.pdf
          git diff --staged --quiet || git commit -m "CI: recompile paper PDF [skip ci]"
          git push
CI_EOF
ok "GitHub Actions workflow written"

# ============================================
# 6. requirements.txt (enterprise repos always have one)
# ============================================
if [ ! -f "requirements.txt" ]; then
  log "Writing requirements.txt"
  cat > requirements.txt << 'REQ_EOF'
torch>=2.0
transformers>=4.35
datasets
accelerate
peft>=0.7.1
pydantic
loguru
arxiv
tiktoken
numpy
pandas
sentence-transformers
faiss-cpu
REQ_EOF
  ok "requirements.txt written"
else
  ok "requirements.txt already exists — leaving as-is"
fi

# ============================================
# 7. Build the paper locally (PDF + arXiv bundle)
# ============================================
cd paper || exit 1
if command -v pdflatex &> /dev/null; then
  if command -v make &> /dev/null; then
    log "Running make (compiles PDF)"
    make
    log "Running make arxiv (builds submission bundle)"
    if command -v zip &> /dev/null; then
      make arxiv
      ok "arXiv submission bundle: paper/arxiv_submission.zip"
    else
      fail "zip not installed — run: sudo apt-get install -y zip"
    fi
  else
    fail "make not installed — run: sudo apt-get install -y make"
  fi
else
  fail "pdflatex not found — run: sudo apt-get update && sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
fi
cd "$PROJECT_DIR" || exit 1

# ============================================
# 8. Update README with badges + formal sections
# ============================================
log "Updating README.md with badges and formal metadata"
cat > README.md << README_EOF
# Physics Research Assistant SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Paper](https://github.com/${GH_USER}/${REPO_NAME}/actions/workflows/build-paper.yml/badge.svg)](https://github.com/${GH_USER}/${REPO_NAME}/actions/workflows/build-paper.yml)

A physics-domain Small Language Model built end-to-end on Apple Silicon:
custom Rust BPE tokenizer, base model training, LoRA fine-tuning,
retrieval-augmented generation (BM25 + dense + hybrid).

## Paper

Full write-up (arXiv-style): [paper/physics_slm_paper.pdf](paper/physics_slm_paper.pdf)
LaTeX source: [paper/physics_slm_paper.tex](paper/physics_slm_paper.tex)
arXiv-ready submission bundle: [paper/arxiv_submission.zip](paper/arxiv_submission.zip)

### Citation
See [CITATION.cff](CITATION.cff), or cite directly:

\`\`\`bibtex
@misc{singh2026physicsslm,
  title  = {A Physics-Specialized Small Language Model: Tokenizer Design,
            LoRA Fine-Tuning, and Retrieval-Augmented Generation on
            Consumer Hardware},
  author = {Singh, Jaydip and Kumbhar, Linkan},
  year   = {2026},
  url    = {https://github.com/${GH_USER}/${REPO_NAME}}
}
\`\`\`

## Results Summary

| Phase | Milestone | Key Result |
|---|---|---|
| 1 | Tokenizer + data pipeline | 32K BPE vocab, frozen |
| 2 | LoRA fine-tuning | 0.0107 to 0.0060 eval loss (44% down) |
| 3 | Inference + sampling profiles | 1.21-1.39x LoRA speedup |
| 4 | RAG (BM25 + dense + hybrid) | Recall@5: 75% to 87.5% (hybrid) |

## Repository Structure

\`\`\`
scripts/        Training, evaluation, and retrieval scripts
config/         YAML configs for LoRA, RAG, inference
data/           Corpus metadata and stats (large binaries excluded)
checkpoints/    Model checkpoint metadata (weights excluded, see .gitignore)
results/        Evaluation reports (JSON/Markdown)
subword_tokenizer/  Rust BPE tokenizer
paper/          Paper source, PDF, Makefile, arXiv submission bundle
.github/        CI workflow (auto-rebuilds the paper PDF)
\`\`\`

## Quick Start

\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/phase2_lora_finetune.py --config config/phase2_lora_config.yaml --resume auto
python scripts/inference_lora.py --adapter checkpoints/phase2_lora/lora_adapter_step9000.pt
\`\`\`

## Building the Paper

\`\`\`bash
cd paper
make            # compile PDF
make arxiv      # build arXiv submission zip
make clean      # remove build artifacts
\`\`\`

## License
MIT — see [LICENSE](LICENSE).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Authors
Jaydip Singh, Linkan Kumbhar
README_EOF
ok "README.md updated"

# ============================================
# 9. Commit and push everything
# ============================================
log "Committing enterprise formalities"
git add .
if git commit -m "Add enterprise formalities: Makefile, CI, CITATION.cff, LICENSE, arXiv bundle" 2>/tmp/commit_err.log; then
  ok "commit created"
else
  ok "nothing new to commit"
fi

log "Pushing to origin/main"
if git push -u origin main 2>/tmp/push_err.log; then
  ok "pushed successfully"
else
  fail "push failed:"
  cat /tmp/push_err.log
fi

echo ""
echo "Done. Repo now has: Makefile, CI auto-build, CITATION.cff, LICENSE, CONTRIBUTING.md, arXiv submission bundle, badged README."
