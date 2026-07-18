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
