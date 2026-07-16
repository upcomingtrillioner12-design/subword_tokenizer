#!/bin/bash

PROJECT_DIR="/workspaces/subword_tokenizer"
DATE=$(date +%Y%m%d)
SUBMISSION_DIR="$PROJECT_DIR/arxiv_submission_$DATE"
TEX_FILE="project_reference_arxiv.tex"
BUNDLE="$PROJECT_DIR/arxiv_bundle_$DATE.tar.gz"

echo "=== arXiv Submission Prep ==="

cd "$PROJECT_DIR"
git pull origin main 2>/dev/null || true

rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

if [ -f ".project_reference.tex" ]; then
    cp ".project_reference.tex" "$SUBMISSION_DIR/$TEX_FILE"
elif [ -f "project_reference.tex" ]; then
    cp "project_reference.tex" "$SUBMISSION_DIR/$TEX_FILE"
else
    echo "❌ No .tex found"
    exit 1
fi

cd "$SUBMISSION_DIR"
sed -i 's|.*doc/architecture\.png.*|% FIGURE REMOVED|' "$TEX_FILE"
sed -i 's|.*architecture\.png.*|% FIGURE REMOVED|' "$TEX_FILE"
sed -i 's|.*architecture\.svg.*|% FIGURE REMOVED|' "$TEX_FILE"

pdflatex -interaction=nonstopmode "$TEX_FILE" >/dev/null 2>&1
pdflatex -interaction=nonstopmode "$TEX_FILE" >/dev/null 2>&1

tar -czf "$BUNDLE" *
echo "✅ Bundle: $BUNDLE ($(ls -la "$BUNDLE" | awk '{print $5}') bytes)"

echo ""
echo "=== COPY-PASTE THIS INTO arXiv ==="
echo ""
echo "----- TITLE -----"
grep -m1 '\\title{' "$TEX_FILE" | sed 's/\\title{//;s/}$//;s/\\//g'
echo ""
echo "----- AUTHORS -----"
grep -m1 '\\author{' "$TEX_FILE" | sed 's/\\author{//;s/}$//;s/\\//g'
echo ""
echo "----- ABSTRACT -----"
sed -n '/\\begin{abstract}/,/\\end{abstract}/p' "$TEX_FILE" | sed 's/\\begin{abstract}//;s/\\end{abstract}//;s/\\//g'
echo ""
echo "----- CATEGORY -----"
echo "cs.CL (Computation and Language) or cs.LG (Machine Learning)"
echo ""
echo "----- COMMENTS -----"
echo "16 pages. Code: https://github.com/upcomingtrillioner12-design/subword_tokenizer"
echo ""
echo "===== UPLOAD FILE ====="
echo "$BUNDLE"
echo ""

cd "$PROJECT_DIR"
git add -A
git commit -m "arxiv: submission prep $DATE" 2>/dev/null || true
git push origin main 2>/dev/null || true

echo "Opening https://arxiv.org/submit ..."
python3 -c "import webbrowser; webbrowser.open('https://arxiv.org/submit')" 2>/dev/null || echo "Open manually: https://arxiv.org/submit"

echo ""
echo "🚀 DONE. Paste the metadata above into arXiv and upload the bundle."
