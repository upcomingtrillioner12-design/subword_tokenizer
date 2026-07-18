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
