#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║     BUILDING ARXIV PAPER                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

# Check for required tools
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Install TeX Live:"
    echo "   sudo apt-get install texlive-full texlive-bibtex-extra texlive-science"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found."
    exit 1
fi

# Generate figures first
echo ""
echo "📊 Generating figures from real checkpoint data..."
python3 scripts/generate_figures.py

# Build LaTeX
echo ""
echo "📄 Compiling LaTeX..."

# First pass
pdflatex -interaction=nonstopmode main.tex || true

# Bibliography
if [ -f main.aux ]; then
    bibtex main || true
fi

# Second pass
pdflatex -interaction=nonstopmode main.tex || true

# Third pass (resolve references)
pdflatex -interaction=nonstopmode main.tex || true

if [ -f main.pdf ]; then
    echo ""
    echo "✅ PAPER BUILD COMPLETE"
    ls -lh main.pdf
    echo ""
    echo "📄 Output: main.pdf"
else
    echo "❌ Build failed. Check logs above."
    exit 1
fi
