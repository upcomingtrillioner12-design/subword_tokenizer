#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║     BUILDING ARXIV PAPER                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

# Generate figures
echo "Generating figures..."
python3 generate_loss_curve.py || echo "⚠️  Loss curve generation failed (matplotlib may be missing)"
python3 generate_architecture.py || echo "⚠️  Architecture diagram generation failed"

# First pass
pdflatex -interaction=nonstopmode arxiv_paper.tex || true
# Bibliography
bibtex arxiv_paper || true
# Second pass
pdflatex -interaction=nonstopmode arxiv_paper.tex || true
# Third pass
pdflatex -interaction=nonstopmode arxiv_paper.tex || true

echo ""
echo "✅ PAPER BUILD COMPLETE"
ls -lh arxiv_paper.pdf
