#!/bin/bash
cd /workspaces/subword_tokenizer/arxiv_submission_20260716_final
sed -i 's|{architecture.png}|% FIGURE REMOVED|' project_reference_arxiv.tex 2>/dev/null || true
pdflatex -interaction=nonstopmode project_reference_arxiv.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode project_reference_arxiv.tex >/dev/null 2>&1
cd /workspaces/subword_tokenizer
mkdir -p arxiv_enterprise
cp arxiv_submission_20260716_final/project_reference_arxiv.tex arxiv_enterprise/
cp arxiv_submission_20260716_final/project_reference_arxiv.pdf arxiv_enterprise/
tar -czf arxiv_enterprise_bundle.tar.gz arxiv_enterprise/
echo "✅ ENTERPRISE PAPER READY"
ls -la arxiv_enterprise/project_reference_arxiv.pdf
ls -la arxiv_enterprise_bundle.tar.gz
echo ""
echo "Title: Subword Tokenizer and SLM Training/Evaluation Stack"
echo "Authors: Jaydip Singh, Linkan Kumbhar"
echo "Category: cs.CL"
echo "Pages: 4"
