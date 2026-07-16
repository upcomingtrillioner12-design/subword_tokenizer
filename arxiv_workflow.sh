#!/bin/bash

PROJECT_DIR="/workspaces/subword_tokenizer"
SUBMISSION_DIR="$PROJECT_DIR/arxiv_submission_$(date +%Y%m%d)"
TEX_FILE="project_reference_arxiv.tex"
BUNDLE="arxiv_bundle_$(date +%Y%m%d).tar.gz"

echo "=== arXiv + Git Workflow ==="

cd "$PROJECT_DIR"
echo "📥 Pulling latest changes..."
git pull origin main 2>/dev/null || echo "⚠️ No remote or already up to date"

if [ ! -f "$SUBMISSION_DIR/$TEX_FILE" ]; then
    mkdir -p "$SUBMISSION_DIR"
    if [ -f ".project_reference.tex" ]; then
        cp ".project_reference.tex" "$SUBMISSION_DIR/$TEX_FILE"
        echo "✅ Copied .project_reference.tex"
    elif [ -f "project_reference.tex" ]; then
        cp "project_reference.tex" "$SUBMISSION_DIR/$TEX_FILE"
        echo "✅ Copied project_reference.tex"
    else
        echo "❌ No .tex source found"
        exit 1
    fi
fi

cd "$SUBMISSION_DIR"
sed -i 's|.*doc/architecture\.png.*|% FIGURE REMOVED|' "$TEX_FILE" 2>/dev/null
sed -i 's|.*architecture\.png.*|% FIGURE REMOVED|' "$TEX_FILE" 2>/dev/null
sed -i 's|.*architecture\.svg.*|% FIGURE REMOVED|' "$TEX_FILE" 2>/dev/null

echo "📄 Compiling LaTeX..."
pdflatex -interaction=nonstopmode "$TEX_FILE" >/dev/null 2>&1
pdflatex -interaction=nonstopmode "$TEX_FILE" >/dev/null 2>&1

tar -czf "$PROJECT_DIR/$BUNDLE" *
echo "✅ Bundle created: $PROJECT_DIR/$BUNDLE"

echo ""
echo "=== PDF Info ==="
ls -la "$SUBMISSION_DIR/${TEX_FILE%.tex}.pdf"

cd "$PROJECT_DIR"
git add -A
git commit -m "arxiv: update submission bundle $(date +%Y%m%d)" 2>/dev/null || echo "⚠️ Nothing to commit"
git push origin main 2>/dev/null || echo "⚠️ Push failed or no remote"

echo ""
echo "=== NEXT STEPS ==="
echo "1. Go to https://arxiv.org/submit"
echo "2. Upload: $PROJECT_DIR/$BUNDLE"
echo "3. Fill metadata (title, authors, abstract)"
echo "4. Category: cs.CL or cs.LG"
echo "5. Submit!"
echo ""
echo "Bundle: $PROJECT_DIR/$BUNDLE"
