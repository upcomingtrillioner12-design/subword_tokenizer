#!/bin/bash
set -euo pipefail

echo "=================================="
echo "  BUILDING ARXIV PAPER"
echo "=================================="

if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Install TeX Live:"
    echo "   sudo apt-get install texlive-full texlive-bibtex-extra texlive-science"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

echo ""
echo "[INFO] Generating figures..."
python3 scripts/generate_figures.py || {
    echo "[WARN] Python figure generation failed"
    echo "[INFO] Install: pip install matplotlib numpy"
}

echo ""
echo "[INFO] Compiling LaTeX..."

pdflatex -interaction=nonstopmode main.tex || true

if [ -f main.aux ]; then
    bibtex main || true
fi

pdflatex -interaction=nonstopmode main.tex || true
pdflatex -interaction=nonstopmode main.tex || true

if [ -f main.pdf ]; then
    echo ""
    echo "=================================="
    echo "  PAPER BUILD COMPLETE"
    echo "=================================="
    ls -lh main.pdf
    echo ""
    echo "Output: main.pdf"
else
    echo "ERROR: Build failed."
    exit 1
fi
